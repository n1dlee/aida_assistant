"""
voice/wake_word.py
Wake word detection using OpenWakeWord.
Listens continuously; fires callback when wake word is detected.
Default wake word: "hey aida" — configure in settings.yaml.
"""
import asyncio
import logging
import os

log = logging.getLogger("aida.voice.wakeword")


class WakeWordDetector:
    def __init__(self, wake_word: str = None):
        self.wake_word = wake_word or os.getenv("AIDA_WAKE_WORD", "hey aida")
        self._model = None
        self._available = False
        self._init()

    def _init(self):
        try:
            from openwakeword.model import Model
            self._model = Model(inference_framework="onnx")
            self._available = True
            log.info("Wake word detector ready (word: '%s').", self.wake_word)
        except ImportError:
            log.warning("openwakeword not installed — wake word disabled.")
        except Exception as e:
            log.warning("Wake word init failed: %s", e)

    def is_available(self) -> bool:
        return self._available

    async def wait_for_wake_word(self) -> bool:
        """
        Blocks until wake word is detected.
        Returns True when triggered.
        """
        if not self._available:
            log.debug("Wake word not available — skipping.")
            return True  # Pass-through so text mode still works

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._blocking_listen)

    def _blocking_listen(self) -> bool:
        import numpy as np
        try:
            import sounddevice as sd
            chunk_size = 1280
            sample_rate = 16000
            log.info("Waiting for wake word '%s'...", self.wake_word)
            with sd.InputStream(samplerate=sample_rate, channels=1, dtype="int16") as stream:
                while True:
                    audio_chunk, _ = stream.read(chunk_size)
                    audio_flat = np.squeeze(audio_chunk)
                    predictions = self._model.predict(audio_flat)
                    for model_name, score in predictions.items():
                        if score > 0.5:
                            log.info("Wake word detected! (score=%.2f)", score)
                            return True
        except Exception as e:
            log.error("Wake word detection error: %s", e)
            return True
