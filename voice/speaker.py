"""
voice/speaker.py
Text-to-Speech output using Piper TTS (local, fast, offline).
Falls back to pyttsx3 (cross-platform), then print-only if nothing available.
"""
import asyncio
import logging
import os

log = logging.getLogger("aida.voice.speaker")


class Speaker:
    def __init__(self):
        self._engine = None
        self._mode = "none"
        self._init()

    def _init(self):
        # Try pyttsx3 first (easiest cross-platform install)
        try:
            import pyttsx3
            self._engine = pyttsx3.init()
            rate = int(os.getenv("AIDA_TTS_RATE", "175"))
            self._engine.setProperty("rate", rate)
            self._mode = "pyttsx3"
            log.info("TTS engine: pyttsx3")
            return
        except ImportError:
            pass
        except Exception as e:
            log.debug("pyttsx3 init failed: %s", e)

        # Try piper CLI
        try:
            import subprocess
            result = subprocess.run(
                ["piper", "--version"], capture_output=True, timeout=3
            )
            if result.returncode == 0:
                self._mode = "piper"
                log.info("TTS engine: piper")
                return
        except Exception:
            pass

        log.warning("No TTS engine available — responses will be text-only.")
        self._mode = "none"

    def is_available(self) -> bool:
        return self._mode != "none"

    async def speak(self, text: str):
        """Convert text to speech asynchronously."""
        # get_running_loop() is the non-deprecated replacement for
        # get_event_loop() in Python 3.10+
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._sync_speak, text)

    def _sync_speak(self, text: str):
        if self._mode == "pyttsx3":
            self._engine.say(text)
            self._engine.runAndWait()

        elif self._mode == "piper":
            import subprocess
            import tempfile

            voice = os.getenv("PIPER_VOICE", "en_US-lessac-medium")
            # mktemp() is deprecated since Python 2.3 — use mkstemp()
            fd, tmp = tempfile.mkstemp(suffix=".wav")
            os.close(fd)   # close the OS-level fd; piper will overwrite the file
            try:
                proc = subprocess.run(
                    ["piper", "--model", voice, "--output_file", tmp],
                    input=text.encode(),
                    capture_output=True,
                )
                if proc.returncode == 0:
                    self._play_wav(tmp)
            finally:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass

    def _play_wav(self, path: str):
        try:
            import sounddevice as sd
            import soundfile as sf
            data, sr = sf.read(path)
            sd.play(data, sr)
            sd.wait()
        except Exception as e:
            log.debug("Audio playback failed: %s", e)
