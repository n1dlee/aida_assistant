"""
voice/listener.py
Microphone input → text using OpenAI Whisper (local, via faster-whisper).
Falls back to text input if no mic/whisper available.
"""
import asyncio
import logging
import os
import tempfile

log = logging.getLogger("aida.voice.listener")


class VoiceListener:
    def __init__(self):
        self.model_size = os.getenv("WHISPER_MODEL", "small")
        self._model = None
        self._available = False
        self._init()

    def _init(self):
        try:
            from faster_whisper import WhisperModel
            self._model = WhisperModel(
                self.model_size, device="cpu", compute_type="int8"
            )
            self._available = True
            log.info("Whisper loaded (model: %s).", self.model_size)
        except ImportError:
            log.warning("faster-whisper not installed — voice input disabled.")
        except Exception as e:
            log.warning("Whisper init failed: %s", e)

    def is_available(self) -> bool:
        return self._available

    async def listen(self, timeout: float = 5.0) -> str:
        """Record from mic and transcribe. Returns transcribed text."""
        if not self._available:
            raise RuntimeError("Voice input not available.")

        # get_running_loop() replaces the deprecated get_event_loop()
        # (Python 3.10+ emits DeprecationWarning when no loop is running)
        loop = asyncio.get_running_loop()
        audio_path = await loop.run_in_executor(None, self._record, timeout)
        text = await loop.run_in_executor(None, self._transcribe, audio_path)
        return text

    def _record(self, duration: float) -> str:
        """Records audio to a secure temp WAV file (mkstemp, not mktemp)."""
        import sounddevice as sd
        import soundfile as sf
        import numpy as np

        sample_rate = 16000
        log.info("Recording for %.1fs...", duration)
        audio = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
        )
        sd.wait()

        # tempfile.mktemp() is deprecated since Python 2.3 (unsafe race condition).
        # Use mkstemp() which atomically creates the file and returns an fd.
        fd, tmp = tempfile.mkstemp(suffix=".wav")
        os.close(fd)   # soundfile will open the path itself
        sf.write(tmp, audio, sample_rate)
        return tmp

    def _transcribe(self, audio_path: str) -> str:
        try:
            segments, _ = self._model.transcribe(audio_path, beam_size=5)
            text = " ".join(seg.text for seg in segments).strip()
            log.info("Transcribed: %s", text)
            return text
        finally:
            # Clean up the temp WAV we created in _record
            try:
                os.unlink(audio_path)
            except OSError:
                pass
