"""
voice/transcription_server.py
──────────────────────────────
Persistent subprocess: loads WhisperModel ONCE, stays alive.
Each transcription call costs only inference time (~1-3s on GPU),
not model-load time (which is 20-50s per subprocess spawn).

Protocol (line-buffered JSON):
  stdin  ← {"path": "/tmp/x.wav", "lang": "ru"}   (parent sends)
  stdout → {"text": "...", "lang": "ru"}            (server responds)
  stdout → "READY\n"                                (emitted once on startup)

The server deletes the WAV file after transcribing (caller need not clean up).
"""

import json, os, sys, traceback

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")


def _get_device():
    try:
        import torch
        if torch.cuda.is_available():
            print(f"[transcription_server] CUDA device: {torch.cuda.get_device_name(0)}",
                  file=sys.stderr, flush=True)
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def _load_model(device: str):
    from faster_whisper import WhisperModel
    compute_order = ["float16", "int8"] if device == "cuda" else ["int8", "float32"]
    last_err = None
    for ct in compute_order:
        try:
            print(f"[transcription_server] Loading {WHISPER_MODEL} "
                  f"device={device} compute_type={ct} ...",
                  file=sys.stderr, flush=True)
            m = WhisperModel(WHISPER_MODEL, device=device, compute_type=ct)
            print(f"[transcription_server] Model ready ({device}/{ct})",
                  file=sys.stderr, flush=True)
            return m
        except Exception as exc:
            print(f"[transcription_server] {ct} failed: {exc}", file=sys.stderr)
            last_err = exc
    raise RuntimeError(f"Cannot load WhisperModel: {last_err}")


def main():
    device = _get_device()
    try:
        model = _load_model(device)
    except Exception as exc:
        print(f"[transcription_server] FATAL: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)

    # Signal parent that model is ready
    print("READY", flush=True)

    # Process requests indefinitely
    for raw_line in sys.stdin:
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        path = None
        try:
            req  = json.loads(raw_line)
            path = req.get("path", "")
            lang = req.get("lang")
            if lang in (None, "auto", ""):
                lang = None

            segs, info = model.transcribe(path, beam_size=5, language=lang)
            text = " ".join(s.text for s in segs).strip()
            print(json.dumps({"text": text, "lang": info.language}), flush=True)

        except Exception as exc:
            tb = traceback.format_exc()
            print(f"[transcription_server] Error: {exc}\n{tb}", file=sys.stderr, flush=True)
            print(json.dumps({"text": "", "lang": "en", "error": str(exc)}), flush=True)
        finally:
            if path:
                try:
                    os.unlink(path)
                except OSError:
                    pass


if __name__ == "__main__":
    main()
