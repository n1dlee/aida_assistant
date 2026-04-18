"""
voice/voice_worker.py
─────────────────────
Run as a standalone subprocess from ui.py to record audio and transcribe it.
Isolation purpose: CTranslate2 / faster-whisper can crash at C level; running
here means the crash kills only this subprocess, not the main AIDA app.

Usage (spawned by ui.py):
    python voice_worker.py
stdout: transcript text (on success)
stderr: error message  (on failure)
exit 0 = success, exit 1 = error
"""

import os
import sys
import tempfile

# ── env-configurable constants ────────────────────────────────────────────────
SAMPLE_RATE      = 16_000
CHUNK_DURATION   = 0.1                                         # seconds per chunk
SILENCE_TIMEOUT  = float(os.getenv("AIDA_SILENCE_TIMEOUT",  "3.0"))
ENERGY_THRESHOLD = float(os.getenv("AIDA_ENERGY_THRESHOLD", "0.01"))
MAX_DURATION     = float(os.getenv("AIDA_MAX_RECORD",        "30.0"))
WHISPER_MODEL    = os.getenv("WHISPER_MODEL", "small")


def _record_vad() -> str:
    """Record with VAD; returns path to a temp WAV file."""
    import sounddevice as sd
    import soundfile   as sf
    import numpy       as np

    chunk_size     = int(SAMPLE_RATE * CHUNK_DURATION)
    max_chunks     = int(MAX_DURATION / CHUNK_DURATION)
    silence_needed = int(SILENCE_TIMEOUT / CHUNK_DURATION)

    frames: list         = []
    silent_chunks: int   = 0
    speech_detected: bool = False

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
                    break

    import numpy as np
    audio   = np.concatenate(frames, axis=0)
    fd, tmp = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    sf.write(tmp, audio, SAMPLE_RATE)
    return tmp


def _transcribe(audio_path: str) -> str:
    """Transcribe WAV file; tries int8 → float32 compute types."""
    from faster_whisper import WhisperModel

    last_err = None
    for ct in ["int8", "float32", "auto"]:
        try:
            model    = WhisperModel(WHISPER_MODEL, device="cpu", compute_type=ct)
            segments, _ = model.transcribe(audio_path, beam_size=5)
            return " ".join(seg.text for seg in segments).strip()
        except Exception as e:
            last_err = e

    raise RuntimeError(f"All compute_types failed: {last_err}")


def main():
    audio_path = None
    try:
        print("[voice_worker] Recording...", file=sys.stderr, flush=True)
        audio_path = _record_vad()
        print("[voice_worker] Transcribing...", file=sys.stderr, flush=True)
        text = _transcribe(audio_path)
        print(text, flush=True)          # ← transcript on stdout
        sys.exit(0)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)
    finally:
        if audio_path:
            try:
                os.unlink(audio_path)
            except OSError:
                pass


if __name__ == "__main__":
    main()
