"""
voice/listener.py
Microphone input → text using faster-whisper (Whisper, local).

Two recording modes:
  1. listen()                 — VAD-based: starts immediately, stops ~3s after
                                speech ends. No fixed timer.
  2. listen_with_wake_word()  — waits for "Aida / Аида" (or custom keywords)
                                using lightweight Whisper transcription on
                                1-second chunks, then switches to VAD recording.

Tuning (env vars):
  WHISPER_MODEL        = small          faster-whisper model size
  AIDA_SILENCE_TIMEOUT = 3.0            seconds of silence before stopping
  AIDA_ENERGY_THRESHOLD = 0.01          RMS threshold that counts as speech
  AIDA_MAX_RECORD      = 30.0           hard cap on recording length (seconds)
"""
import asyncio
import logging
import os
import tempfile
from typing import List

import numpy as np

log = logging.getLogger("aida.voice.listener")

# ── Constants ────────────────────────────────────────────────────────────────
SAMPLE_RATE      = 16_000
CHUNK_DURATION   = 0.1                                                    # seconds per audio chunk
SILENCE_TIMEOUT  = float(os.getenv("AIDA_SILENCE_TIMEOUT",  "3.0"))
ENERGY_THRESHOLD = float(os.getenv("AIDA_ENERGY_THRESHOLD", "0.01"))
MAX_DURATION     = float(os.getenv("AIDA_MAX_RECORD",        "30.0"))

# Keywords that trigger wake-word mode (lower-case, checked via substring)
WAKE_KEYWORDS_DEFAULT: List[str] = ["aida", "аида", "aide", "айда"]


class VoiceListener:
    def __init__(self):
        self.model_size = os.getenv("WHISPER_MODEL", "small")
        self._model     = None
        self._available = False
        self._init()

    # ── Init ─────────────────────────────────────────────────────────────────

    def _init(self):
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            log.warning("faster-whisper not installed — voice input disabled.")
            return

        # Try compute types from most efficient to most compatible.
        # CTranslate2 can crash at C level (not a Python exception) if the CPU
        # doesn't support int8 or onnxruntime is missing, so we try fallbacks.
        compute_types = ["int8", "float32", "auto"]
        last_err = None
        for ct in compute_types:
            try:
                log.debug("Trying WhisperModel compute_type=%s ...", ct)
                self._model     = WhisperModel(self.model_size, device="cpu", compute_type=ct)
                self._available = True
                log.info("Whisper loaded (model=%s, compute_type=%s).", self.model_size, ct)
                return
            except Exception as e:
                log.warning("WhisperModel(compute_type=%s) failed: %s", ct, e)
                last_err = e

        log.warning(
            "All compute_type options failed — voice disabled. "
            "Try: pip install onnxruntime  (last error: %s)", last_err
        )

    def is_available(self) -> bool:
        return self._available

    # ── Public API ───────────────────────────────────────────────────────────

    async def listen(self, timeout: float = None) -> str:
        """
        Record audio with VAD: stops automatically when silence exceeds
        AIDA_SILENCE_TIMEOUT seconds after speech is detected.
        `timeout` kept for backward compat but ignored.
        """
        if not self._available:
            raise RuntimeError("Voice input not available.")
        loop = asyncio.get_running_loop()
        audio_path = await loop.run_in_executor(None, self._record_vad)
        return await loop.run_in_executor(None, self._transcribe, audio_path)

    async def listen_with_wake_word(
        self,
        wake_keywords: List[str] = None,
    ) -> str:
        """
        Phase 1 — Wake word detection:
            Continuously transcribes 1-second audio chunks with Whisper.
            Returns as soon as any keyword in `wake_keywords` appears in the
            transcription (case-insensitive substring match).

        Phase 2 — VAD recording:
            Records until AIDA_SILENCE_TIMEOUT seconds of silence,
            then transcribes the full utterance and returns it.

        The wake word itself is stripped from the returned transcript.
        """
        if not self._available:
            raise RuntimeError("Voice input not available.")

        keywords = [k.lower() for k in (wake_keywords or WAKE_KEYWORDS_DEFAULT)]
        loop     = asyncio.get_running_loop()

        log.info("Waiting for wake word: %s", keywords)
        await loop.run_in_executor(None, self._wait_for_wake, keywords)

        log.info("Wake word detected — starting VAD recording.")
        audio_path = await loop.run_in_executor(None, self._record_vad)
        text       = await loop.run_in_executor(None, self._transcribe, audio_path)

        # Strip the wake word from the beginning of the transcript if present
        text_lower = text.lower()
        for kw in keywords:
            if text_lower.startswith(kw):
                text = text[len(kw):].lstrip(" ,.")
                break

        return text

    # ── Wake-word detection (Whisper-based) ──────────────────────────────────

    def _wait_for_wake(self, keywords: List[str]) -> None:
        """
        Blocks until one of `keywords` is found in a Whisper transcription.
        Uses 1-second non-overlapping audio chunks for low latency.
        Silent chunks (RMS below ENERGY_THRESHOLD) are skipped without
        transcription to save CPU.
        """
        import sounddevice as sd
        import soundfile   as sf

        chunk_size = int(SAMPLE_RATE * 1.0)   # 1-second window

        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32") as stream:
            while True:
                audio, _ = stream.read(chunk_size)
                flat      = audio.flatten()
                rms       = float(np.sqrt(np.mean(flat ** 2)))

                if rms < ENERGY_THRESHOLD:
                    continue   # silence — skip transcription

                fd, tmp = tempfile.mkstemp(suffix=".wav")
                os.close(fd)
                try:
                    sf.write(tmp, flat, SAMPLE_RATE)
                    segments, _ = self._model.transcribe(
                        tmp,
                        beam_size=1,        # fastest setting for keyword spotting
                        language=None,      # auto-detect (handles ru + en)
                    )
                    chunk_text = " ".join(s.text for s in segments).lower()
                    log.debug("Wake check: %r", chunk_text)

                    if any(kw in chunk_text for kw in keywords):
                        log.info("Wake word detected in: %r", chunk_text)
                        return
                except Exception as exc:
                    log.debug("Wake chunk error (skipping): %s", exc)
                finally:
                    try:
                        os.unlink(tmp)
                    except OSError:
                        pass

    # ── VAD recording ────────────────────────────────────────────────────────

    def _record_vad(self) -> str:
        """
        Records audio chunks until AIDA_SILENCE_TIMEOUT seconds of consecutive
        sub-threshold audio follow at least one speech chunk.
        Returns the path to a temporary WAV file; caller must NOT delete it —
        _transcribe() handles cleanup.
        """
        import sounddevice as sd
        import soundfile   as sf

        chunk_size     = int(SAMPLE_RATE * CHUNK_DURATION)
        max_chunks     = int(MAX_DURATION / CHUNK_DURATION)
        silence_needed = int(SILENCE_TIMEOUT / CHUNK_DURATION)

        log.info(
            "VAD recording started (silence=%.1fs, max=%.0fs, rms_thr=%.4f).",
            SILENCE_TIMEOUT, MAX_DURATION, ENERGY_THRESHOLD,
        )

        frames:         List[np.ndarray] = []
        silent_chunks:  int              = 0
        speech_detected: bool            = False

        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32") as stream:
            for _ in range(max_chunks):
                chunk, _ = stream.read(chunk_size)
                flat      = chunk.flatten()
                frames.append(flat)
                rms       = float(np.sqrt(np.mean(flat ** 2)))

                if rms >= ENERGY_THRESHOLD:
                    speech_detected = True
                    silent_chunks   = 0
                elif speech_detected:
                    silent_chunks += 1
                    if silent_chunks >= silence_needed:
                        log.info(
                            "Silence detected (%.1fs) — stopping recording.",
                            silent_chunks * CHUNK_DURATION,
                        )
                        break

        audio   = np.concatenate(frames, axis=0)
        fd, tmp = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        sf.write(tmp, audio, SAMPLE_RATE)
        return tmp

    # ── Transcription ─────────────────────────────────────────────────────────

    def _transcribe(self, audio_path: str) -> str:
        """Transcribes the WAV at audio_path and deletes it afterwards."""
        try:
            segments, _ = self._model.transcribe(audio_path, beam_size=5)
            text        = " ".join(seg.text for seg in segments).strip()
            log.info("Transcribed: %s", text)
            return text
        finally:
            try:
                os.unlink(audio_path)
            except OSError:
                pass
