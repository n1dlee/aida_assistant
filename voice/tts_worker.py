"""
voice/tts_worker.py
───────────────────
TTS chain (best quality first):
  1. Silero TTS  — offline, GPU-accelerated, excellent Russian (v3_1_ru / xenia)
                   also used for English (v3_en / en_0..en_111)
                   pip install torch  (already installed)
  2. edge-tts    — Microsoft neural TTS (online), used as Silero fallback
                   pip install edge-tts
  3. pyttsx3     — SAPI5 local (Windows), last resort
                   pip install pyttsx3

stdin : UTF-8 text to speak
"""
import asyncio, os, sys


def _is_russian(text: str, threshold: float = 0.20) -> bool:
    if not text: return False
    cyr = sum(1 for c in text if "\u0400" <= c <= "\u04ff")
    return (cyr / len(text)) >= threshold


def _get_device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


# ── Method 1: Silero TTS ─────────────────────────────────────────────────────
def _silero_speak(text: str, is_ru: bool) -> None:
    import torch, sounddevice as sd

    device = torch.device(_get_device())
    language = "ru" if is_ru else "en"
    speaker_id = "v3_1_ru" if is_ru else "v3_en"
    speaker    = "xenia"   if is_ru else "en_56"   # good quality voices

    print(f"[tts_worker] Silero lang={language} speaker={speaker} device={device}",
          file=sys.stderr, flush=True)

    model, _ = torch.hub.load(
        repo_or_dir="snakers4/silero-models",
        model="silero_tts",
        language=language,
        speaker=speaker_id,
        trust_repo=True,
    )
    model.to(device)

    sample_rate = 48_000
    audio = model.apply_tts(text=text, speaker=speaker, sample_rate=sample_rate)

    # audio is a 1D float tensor on GPU or CPU
    data = audio.cpu().numpy()
    sd.play(data, samplerate=sample_rate)
    sd.wait()
    print("[tts_worker] Silero done", file=sys.stderr, flush=True)


# ── Method 2: edge-tts ───────────────────────────────────────────────────────
async def _edge_tts_async(text: str, is_ru: bool) -> None:
    import edge_tts, numpy as np, sounddevice as sd

    voice = "ru-RU-SvetlanaNeural" if is_ru else "en-US-AriaNeural"
    print(f"[tts_worker] edge-tts voice={voice}", file=sys.stderr, flush=True)

    communicate = edge_tts.Communicate(text, voice,
                                       output_format="raw-24khz-16bit-mono-pcm")
    pcm_chunks: list[bytes] = []
    async for chunk in communicate.stream():
        if chunk.get("type") == "audio":
            pcm_chunks.append(chunk["data"])

    if not pcm_chunks:
        raise RuntimeError("edge-tts returned no audio")

    pcm   = b"".join(pcm_chunks)
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32_768.0
    sd.play(audio, samplerate=24_000)
    sd.wait()


def _edge_tts_speak(text: str, is_ru: bool) -> None:
    asyncio.run(_edge_tts_async(text, is_ru))


# ── Method 3: pyttsx3 ────────────────────────────────────────────────────────
def _pyttsx3_speak(text: str, is_ru: bool) -> None:
    import pyttsx3
    engine = pyttsx3.init()
    engine.setProperty("rate", 165)
    if is_ru:
        for v in (engine.getProperty("voices") or []):
            if any(k in (v.name + v.id).lower()
                   for k in ("russian", "irina", "pavel", "svetlana", "ru-ru")):
                engine.setProperty("voice", v.id)
                break
    engine.say(text)
    engine.runAndWait()


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    text = sys.stdin.read().strip()
    if not text:
        sys.exit(0)

    is_ru = _is_russian(text)
    print(f"[tts_worker] is_ru={is_ru} len={len(text)}", file=sys.stderr, flush=True)

    # 1. Silero
    try:
        _silero_speak(text, is_ru)
        sys.exit(0)
    except ImportError as e:
        print(f"[tts_worker] Silero import error: {e}", file=sys.stderr)
    except Exception as e:
        print(f"[tts_worker] Silero failed: {e}", file=sys.stderr)

    # 2. edge-tts
    try:
        _edge_tts_speak(text, is_ru)
        sys.exit(0)
    except ImportError:
        print("[tts_worker] edge-tts not installed", file=sys.stderr)
    except Exception as e:
        print(f"[tts_worker] edge-tts failed: {e}", file=sys.stderr)

    # 3. pyttsx3
    try:
        _pyttsx3_speak(text, is_ru)
        sys.exit(0)
    except Exception as e:
        print(f"[tts_worker] pyttsx3 failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
