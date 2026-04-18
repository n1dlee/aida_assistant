"""
voice/tts_server.py
────────────────────
Persistent TTS subprocess: loads Silero model ONCE, speaks on demand.
Avoids torch.hub re-download and model reload on every TTS call.

stdin:  plain UTF-8 text lines (one utterance per line)
stderr: progress messages
Plays audio directly via sounddevice.
"""
import sys, os


SAMPLE_RATE = 48_000
RU_SPEAKER  = "xenia"    # best Russian Silero voice
EN_SPEAKER  = "en_56"    # natural English Silero voice


def _is_russian(text: str, thr: float = 0.20) -> bool:
    cyr = sum(1 for c in text if "\u0400" <= c <= "\u04ff")
    return bool(text) and (cyr / len(text)) >= thr


def _get_device():
    try:
        import torch
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    except ImportError:
        return "cpu"


def _load_silero(language: str, speaker_id: str, device):
    import torch
    print(f"[tts_server] Loading Silero {language}/{speaker_id} on {device} ...",
          file=sys.stderr, flush=True)
    model, _ = torch.hub.load(
        repo_or_dir="snakers4/silero-models",
        model="silero_tts",
        language=language,
        speaker=speaker_id,
        trust_repo=True,
    )
    model.to(device)
    model.eval()
    print(f"[tts_server] Silero {language} ready.", file=sys.stderr, flush=True)
    return model


def _speak_silero(model, text: str, speaker: str, device) -> None:
    import sounddevice as sd
    audio = model.apply_tts(text=text, speaker=speaker, sample_rate=SAMPLE_RATE)
    sd.play(audio.cpu().numpy(), samplerate=SAMPLE_RATE)
    sd.wait()


def _speak_pyttsx3(text: str, is_ru: bool) -> None:
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


def main():
    device = _get_device()

    # Pre-load both language models at startup
    ru_model = en_model = None
    try:
        ru_model = _load_silero("ru", "v3_1_ru", device)
    except Exception as e:
        print(f"[tts_server] Russian Silero failed: {e}", file=sys.stderr, flush=True)
    try:
        en_model = _load_silero("en", "v3_en", device)
    except Exception as e:
        print(f"[tts_server] English Silero failed: {e}", file=sys.stderr, flush=True)

    print("READY", flush=True)  # signal parent

    for raw in sys.stdin:
        text = raw.strip()
        if not text:
            continue

        is_ru  = _is_russian(text)
        model  = ru_model if is_ru else en_model
        speaker = RU_SPEAKER if is_ru else EN_SPEAKER

        try:
            if model is not None:
                _speak_silero(model, text, speaker, device)
            else:
                _speak_pyttsx3(text, is_ru)
        except Exception as e:
            print(f"[tts_server] speak error: {e}", file=sys.stderr, flush=True)
            try:
                _speak_pyttsx3(text, is_ru)
            except Exception as e2:
                print(f"[tts_server] pyttsx3 fallback error: {e2}", file=sys.stderr)


if __name__ == "__main__":
    main()
