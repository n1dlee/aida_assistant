"""
voice/tts_server.py
────────────────────
Persistent TTS subprocess. Loads once, speaks on demand.

Chain:
  1. gTTS (Google TTS) — good quality, requires internet
     Russian: lang='ru', English: lang='en'
  2. Silero TTS — offline, GPU, excellent quality
  3. pyttsx3 — local SAPI5, last resort

stdin:  plain UTF-8 text lines
stdout: "READY\n" on startup
stderr: progress messages
"""
import os, sys, tempfile


def _is_russian(text: str, thr: float = 0.20) -> bool:
    cyr = sum(1 for c in text if "\u0400" <= c <= "\u04ff")
    return bool(text) and (cyr / len(text)) >= thr


def _play_mp3(path: str) -> None:
    """Play mp3 via sounddevice (no external player needed)."""
    try:
        from pydub import AudioSegment
        import sounddevice as sd
        import numpy as np
        audio = AudioSegment.from_mp3(path)
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        samples /= 2 ** (audio.sample_width * 8 - 1)
        if audio.channels == 2:
            samples = samples.reshape((-1, 2))
        sd.play(samples, samplerate=audio.frame_rate)
        sd.wait()
        return
    except Exception:
        pass
    # fallback: os.system
    if sys.platform == "win32":
        os.system(f'start /wait "" "{path}"')
    else:
        os.system(f'mpg123 -q "{path}" 2>/dev/null || ffplay -nodisp -autoexit -loglevel quiet "{path}"')


def _speak_gtts(text: str, is_ru: bool) -> None:
    from gtts import gTTS
    lang = "ru" if is_ru else "en"
    fd, path = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)
    try:
        tts = gTTS(text, lang=lang)
        tts.save(path)
        _play_mp3(path)
    finally:
        try: os.unlink(path)
        except: pass


def _speak_silero(text: str, is_ru: bool) -> None:
    import torch, sounddevice as sd
    try:
        import omegaconf  # noqa
    except ImportError:
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "omegaconf", "-q"],
                       capture_output=True)

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    language = "ru" if is_ru else "en"
    speaker_id = "v3_1_ru" if is_ru else "v3_en"
    speaker    = "xenia"   if is_ru else "en_56"

    model, _ = torch.hub.load(
        repo_or_dir="snakers4/silero-models",
        model="silero_tts",
        language=language,
        speaker=speaker_id,
        trust_repo=True,
    )
    model.to(device).eval()
    audio = model.apply_tts(text=text, speaker=speaker, sample_rate=48_000)
    sd.play(audio.cpu().numpy(), samplerate=48_000)
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


def _speak(text: str) -> None:
    is_ru = _is_russian(text)

    # 1. gTTS
    try:
        _speak_gtts(text, is_ru)
        return
    except ImportError:
        print("[tts_server] gtts not installed → pip install gtts", file=sys.stderr)
    except Exception as e:
        print(f"[tts_server] gTTS failed: {e}", file=sys.stderr)

    # 2. Silero
    try:
        _speak_silero(text, is_ru)
        return
    except Exception as e:
        print(f"[tts_server] Silero failed: {e}", file=sys.stderr)

    # 3. pyttsx3
    try:
        _speak_pyttsx3(text, is_ru)
    except Exception as e:
        print(f"[tts_server] pyttsx3 failed: {e}", file=sys.stderr)


def main():
    print("READY", flush=True)
    for raw in sys.stdin:
        text = raw.strip()
        if text:
            _speak(text)


if __name__ == "__main__":
    main()
