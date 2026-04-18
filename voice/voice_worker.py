"""
voice/voice_worker.py
─────────────────────
Records audio with VAD. NO Whisper loading here.
Parent process (ui.py) sends the WAV path to the persistent
transcription_server for fast inference.

stdout: /path/to/recorded.wav
exit 0 = success, exit 1 = error
"""
import argparse, os, sys, tempfile
import numpy as np

SAMPLE_RATE      = 16_000
ENERGY_THRESHOLD = float(os.getenv("AIDA_ENERGY_THRESHOLD", "0.015"))
MAX_DURATION     = float(os.getenv("AIDA_MAX_RECORD", "30.0"))


def _record(device=None, silence_timeout=1.0) -> str:
    import sounddevice as sd, soundfile as sf
    chunk    = int(SAMPLE_RATE * 0.05)          # 50ms chunks
    max_c    = int(MAX_DURATION / 0.05)
    sil_need = int(silence_timeout / 0.05)
    frames   = []; silent = 0; speech = False

    print(f"[voice_worker] Recording device={device} silence={silence_timeout}s",
          file=sys.stderr, flush=True)

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                        dtype="float32", device=device) as stream:
        for _ in range(max_c):
            data, _ = stream.read(chunk)
            flat     = data.flatten()
            frames.append(flat)
            rms      = float(np.sqrt(np.mean(flat ** 2)))
            if rms >= ENERGY_THRESHOLD:
                speech = True;  silent = 0
            elif speech:
                silent += 1
                if silent >= sil_need:
                    break

    if not frames:
        raise RuntimeError("No audio recorded")

    audio    = np.concatenate(frames)
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    sf.write(path, audio, SAMPLE_RATE)
    print(f"[voice_worker] Saved {len(audio)/SAMPLE_RATE:.1f}s → {path}",
          file=sys.stderr, flush=True)
    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device",  type=int,   default=None)
    p.add_argument("--silence", type=float, default=1.0)
    args = p.parse_args()
    try:
        path = _record(device=args.device, silence_timeout=args.silence)
        print(path, flush=True)   # ← WAV path to stdout
        sys.exit(0)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
