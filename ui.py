"""
ui.py — AIDA Desktop App
Нативное окно через pywebview + Gradio
Размер окна: 430×900, без скроллинга, минималистичный дизайн
Запуск: python ui.py

Wake Word режим:
  Нажми "🔊 Wake Word ON" → фоновый поток слушает "Aida / Аида".
  Обнаружив слово — автоматически записывает до тишины (VAD) и отвечает.
  Переключи обратно в "Text" или "🔴 Wake Word OFF" чтобы остановить.
"""

import asyncio
import json
import logging
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

# ── ell: must be initialised BEFORE it is used ──────────────────────────────
ell = None  # type: ignore[assignment]
ELL_AVAILABLE = False
try:
    import ell as _ell_module  # noqa: F401
    ell = _ell_module
    ELL_AVAILABLE = True
except ImportError:
    pass

import gradio as gr
import logging as _early_log
_early_log.basicConfig(level=_early_log.DEBUG, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
_early_log.getLogger("aida.ui").info("Gradio version: %s", gr.__version__)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.orchestrator import Orchestrator
from voice.listener import VoiceListener
from config.feature_flags import Flags

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("aida.ui")

# ── Global exception hook — catches ALL unhandled exceptions incl. module-level ──
import traceback as _tb

def _global_exc_hook(exc_type, exc_value, exc_traceback):
    log.critical("=" * 60)
    log.critical("UNHANDLED EXCEPTION — AIDA crashed during startup")
    log.critical("=" * 60)
    for line in _tb.format_exception(exc_type, exc_value, exc_traceback):
        for subline in line.splitlines():
            log.critical(subline)
    log.critical("=" * 60)

sys.excepthook = _global_exc_hook
log.debug("Global exception hook installed.")

if ELL_AVAILABLE and ell is not None:
    log.debug("Initialising ell...")
    ell.init(store="./data/ell_store", autocommit=True, verbose=False)
    log.debug("ell OK.")

# ── Shared state ─────────────────────────────────────────────────────────────
log.info("Creating Orchestrator...")
try:
    orchestrator = Orchestrator()
    log.info("Orchestrator OK.")
except Exception as _e:
    log.critical("Orchestrator failed: %s", _e, exc_info=True)
    raise

# VoiceListener is initialised LAZILY on first voice use.
# WhisperModel (CTranslate2) can crash at C level if onnxruntime/AVX is missing —
# a module-level crash would kill the whole app before the UI ever opens.
voice_listener: "VoiceListener | None" = None
_voice_init_done = False

def _get_voice_listener() -> "VoiceListener | None":
    """Thread-safe lazy init: creates VoiceListener exactly once on first call."""
    global voice_listener, _voice_init_done
    if _voice_init_done:
        return voice_listener
    _voice_init_done = True
    log.info("Initialising VoiceListener (first voice use)...")
    try:
        voice_listener = VoiceListener()
        log.info("VoiceListener ready (available=%s).", voice_listener.is_available())
    except Exception as _e:
        log.error("VoiceListener init failed: %s", _e, exc_info=True)
        voice_listener = None
    return voice_listener

# Queue: wake-word thread puts (transcript, response) here; UI polls it
WAKE_EVENTS: "queue.Queue[tuple]" = queue.Queue()
WAKE_RUNNING  = False
WAKE_THREAD: threading.Thread | None = None


# ── Core visual markup (defined ONCE) ────────────────────────────────────────

def core_markup(glow: bool = False) -> str:
    active = " core-active" if glow else ""
    return f"""
    <div class="jarvis-core{active}">
      <div class="jarvis-core-ring">
        <div class="jarvis-core-dot"></div>
      </div>
    </div>
    """


# ── Helpers ──────────────────────────────────────────────────────────────────

def _run_async(coro):
    """Run coroutine from a sync context, whether a loop is running or not."""
    try:
        loop = asyncio.get_running_loop()
        return asyncio.run_coroutine_threadsafe(coro, loop).result(timeout=60)
    except RuntimeError:
        return asyncio.run(coro)

def _stream_llm(user_input: str):
    """
    Bridge between sync Gradio generator and async orchestrator.stream_process.
    Yields string tokens. Raises on error.
    """
    import queue as _q
    token_q: "_q.Queue[object]" = _q.Queue()
    _DONE = object()

    async def _run():
        try:
            async for tok in orchestrator.stream_process(user_input):
                token_q.put(tok)
        except Exception as exc:
            token_q.put(exc)
        finally:
            token_q.put(_DONE)

    t = threading.Thread(target=lambda: asyncio.run(_run()), daemon=True, name="LLMStream")
    t.start()
    try:
        while True:
            item = token_q.get(timeout=120)
            if item is _DONE:
                break
            if isinstance(item, Exception):
                raise item
            yield str(item)
    finally:
        t.join(timeout=2)





def get_microphones() -> list[str]:
    try:
        import sounddevice as sd
        devices    = sd.query_devices()
        default_in = sd.default.device[0]
        seen: set[str] = set()
        mics: list[str] = []
        if isinstance(default_in, int) and default_in >= 0:
            d0: dict = sd.query_devices(default_in)  # type: ignore[assignment]
            if d0.get("max_input_channels", 0) > 0:
                name0 = str(d0["name"]).strip()
                seen.add(name0.lower())
                mics.append(f"{default_in}: {name0} [Default]")
        for i, dev in enumerate(devices):
            d: dict = dev  # type: ignore[assignment]
            if d.get("max_input_channels", 0) <= 0:
                continue
            name = str(d["name"]).strip()
            if name.lower() in seen:
                continue
            seen.add(name.lower())
            mics.append(f"{i}: {name}")
            if len(mics) >= 10:
                break
        return mics if mics else ["Default microphone"]
    except Exception:
        return ["Default microphone"]


def get_status() -> str:
    groq_ok  = orchestrator.selector.groq.is_available()
    local_ok = orchestrator.selector.local.is_available()
    mem      = orchestrator.vector_store.count()
    mode     = orchestrator.modes.current
    parts: list[str] = [f"{mode.emoji} {mode.label}"]
    if groq_ok:
        parts.append("⚡ Groq")
    elif local_ok:
        parts.append(f"⬡ Local")
    else:
        parts.append("⚠ No LLM")
    active_skills = orchestrator.skills.active_skills
    if active_skills:
        parts.append(" ".join(s.emoji for s in active_skills))
    parts.append(f"🧠 {mem}")
    goal_status = orchestrator.goal_engine.status()
    if goal_status:
        parts.append(goal_status[:40])
    return "  ·  ".join(parts) if parts else "⚠ No model"


def speak_text(text: str) -> None:
    """TTS — fire-and-forget. Does nothing if pyttsx3 unavailable."""
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", int(os.getenv("AIDA_TTS_RATE", "170")))
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        log.debug("TTS skipped: %s", e)


# ── TTS / device helpers ─────────────────────────────────────────────────────

_TTS_WORKER  = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "voice", "tts_worker.py")
_TTS_SERVER_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "voice", "tts_server.py")

def _get_device_index(mic_str: str):
    """Extract integer device index from '7: Mic Name [Default]' → 7, or None."""
    try:
        return int(mic_str.split(":")[0])
    except (ValueError, IndexError):
        return None


# ── Persistent TTS server ────────────────────────────────────────────────────
_TTS_PROC: "subprocess.Popen[bytes] | None" = None
_TTS_PROC_LOCK = threading.Lock()


def _ensure_tts_server() -> "subprocess.Popen[bytes] | None":
    global _TTS_PROC
    if _TTS_PROC and _TTS_PROC.poll() is None:
        return _TTS_PROC
    if not os.path.exists(_TTS_SERVER_SCRIPT):
        return None
    log.info("Starting TTS server …")
    try:
        _TTS_PROC = subprocess.Popen(
            [sys.executable, _TTS_SERVER_SCRIPT],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        # drain stderr in background
        def _log_stderr():
            proc = _TTS_PROC
            if proc and proc.stderr:
                for line in proc.stderr:
                    log.debug("[tts-srv] %s", line.decode(errors="replace").rstrip())
        threading.Thread(target=_log_stderr, daemon=True, name="TTSSrvStderr").start()
        # wait for READY
        rq: queue.Queue = queue.Queue()
        def _wait(): rq.put(_TTS_PROC.stdout.readline() if _TTS_PROC else b"")
        threading.Thread(target=_wait, daemon=True).start()
        try:
            line = rq.get(timeout=120)
            if b"READY" not in line:
                raise RuntimeError(f"TTS server unexpected: {line!r}")
            log.info("TTS server READY.")
        except Exception as exc:
            log.warning("TTS server not ready: %s — falling back to tts_worker", exc)
            if _TTS_PROC: _TTS_PROC.kill()
            _TTS_PROC = None
        return _TTS_PROC
    except Exception as exc:
        log.error("Cannot start TTS server: %s", exc)
        return None


def _preload_tts_server() -> None:
    threading.Thread(target=_ensure_tts_server, daemon=True, name="TTSServerLoader").start()


def _speak_via_server(text: str) -> bool:
    """Send text to persistent TTS server. Returns True on success."""
    with _TTS_PROC_LOCK:
        proc = _ensure_tts_server()
        if not proc:
            return False
        try:
            proc.stdin.write((text.replace("\n", " ") + "\n").encode("utf-8", errors="replace"))
            proc.stdin.flush()
            return True
        except Exception as exc:
            log.debug("TTS server write error: %s", exc)
            return False


def _speak_async(text: str) -> None:
    """Non-blocking TTS — uses persistent server, falls back to subprocess."""
    def _run():
        if not _speak_via_server(text):
            try:
                subprocess.run([sys.executable, _TTS_WORKER],
                               input=text.encode("utf-8", errors="replace"),
                               capture_output=True, timeout=90)
            except Exception as exc:
                log.debug("TTS fallback error: %s", exc)
    threading.Thread(target=_run, daemon=True, name="TTS").start()


def _speak_sync(text: str) -> None:
    """Blocking TTS — used in wake word loop."""
    if not _speak_via_server(text):
        try:
            subprocess.run([sys.executable, _TTS_WORKER],
                           input=text.encode("utf-8", errors="replace"),
                           capture_output=True, timeout=90)
        except Exception as exc:
            log.debug("TTS sync fallback error: %s", exc)


# ── Persistent transcription server ──────────────────────────────────────────
# Loads WhisperModel ONCE on startup. Each TALK call costs only inference
# time (~1-3 s on CUDA) instead of model-load time (20-50 s/spawn).

_TRANS_PROC: "subprocess.Popen[bytes] | None" = None
_TRANS_LOCK  = threading.Lock()
_TRANS_SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "voice", "transcription_server.py"
)


def _ensure_transcription_server() -> "subprocess.Popen[bytes] | None":
    global _TRANS_PROC
    if _TRANS_PROC and _TRANS_PROC.poll() is None:
        return _TRANS_PROC

    if not os.path.exists(_TRANS_SCRIPT):
        log.error("transcription_server.py not found")
        return None

    log.info("Starting transcription server …")
    try:
        _TRANS_PROC = subprocess.Popen(
            [sys.executable, _TRANS_SCRIPT],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
    except Exception as exc:
        log.error("Cannot spawn transcription server: %s", exc)
        return None

    # Drain stderr in background so it doesn't block
    def _log_stderr():
        proc = _TRANS_PROC
        if proc and proc.stderr:
            for line in proc.stderr:
                log.debug("[trans-srv] %s", line.decode(errors="replace").rstrip())
    threading.Thread(target=_log_stderr, daemon=True, name="TransSrvStderr").start()

    # Wait for "READY" signal (model load can take up to 90 s first time)
    ready_q: queue.Queue = queue.Queue()
    def _wait_ready():
        if _TRANS_PROC and _TRANS_PROC.stdout:
            ready_q.put(_TRANS_PROC.stdout.readline())
    threading.Thread(target=_wait_ready, daemon=True, name="TransSrvReady").start()
    try:
        line = ready_q.get(timeout=120)
        if b"READY" not in line:
            raise RuntimeError(f"unexpected startup line: {line!r}")
        log.info("Transcription server READY.")
        return _TRANS_PROC
    except Exception as exc:
        log.error("Transcription server failed to start: %s", exc)
        _TRANS_PROC.kill()
        return None


def transcribe_audio_file(wav_path: str, lang: str | None = None) -> tuple[str, str]:
    """Send WAV path to persistent server → (transcript, detected_lang)."""
    with _TRANS_LOCK:
        proc = _ensure_transcription_server()
        if not proc:
            return "", "en"
        try:
            req = json.dumps({"path": wav_path, "lang": lang or "auto"}) + "\n"
            proc.stdin.write(req.encode())   # type: ignore[union-attr]
            proc.stdin.flush()               # type: ignore[union-attr]
            resp = proc.stdout.readline()    # type: ignore[union-attr]
            data = json.loads(resp.decode().strip())
            return data.get("text", ""), data.get("lang", "en")
        except Exception as exc:
            log.error("transcribe_audio_file error: %s", exc)
            return "", "en"


def _preload_transcription_server() -> None:
    """Called once at startup — pre-warms the model so first TALK is fast."""
    threading.Thread(target=_ensure_transcription_server,
                     daemon=True, name="TransServerLoader").start()


# ── History helpers — Gradio 6.x messages format ─────────────────────────────
# Gradio 6.x dropped the type= parameter from gr.Chatbot but requires
# {"role": "user"/"assistant", "content": str} dicts — tuples [[u,b]] no longer work.

def _append_exchange(history: list, user_text: str, bot_text: str) -> list:
    """Append a complete user+bot pair."""
    return history + [
        {"role": "user",      "content": user_text},
        {"role": "assistant", "content": bot_text},
    ]

def _append_user(history: list, user_text: str) -> list:
    """Append a user message only (assistant reply comes next yield)."""
    return history + [{"role": "user", "content": user_text}]

def _append_bot(history: list, bot_text: str) -> list:
    """Append an assistant reply."""
    return history + [{"role": "assistant", "content": bot_text}]


# ── Wake Word background thread ──────────────────────────────────────────────
# Uses a long-running subprocess (wake_listener.py) that:
#   1. Loads WhisperModel ONCE (amortises startup cost)
#   2. Records 3-second chunks and checks for wake keywords
#   3. On match: records command (VAD), sends JSON line to stdout
#   4. Main thread reads stdout and dispatches to orchestrator
#
# Because the subprocess is isolated, CTranslate2 crashes won't kill the app.

_WAKE_PROC: "subprocess.Popen[bytes] | None" = None


def _wake_word_loop() -> None:
    """Daemon thread: manages the wake_listener subprocess lifecycle."""
    global WAKE_RUNNING, _WAKE_PROC

    log.info("Wake word thread started.")
    listener_script = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "voice", "wake_listener.py",
    )
    if not os.path.exists(listener_script):
        log.error("wake_listener.py not found — wake word disabled.")
        WAKE_RUNNING = False
        return

    while WAKE_RUNNING:
        # ── build command ────────────────────────────────────────────────────
        cmd = [sys.executable, listener_script]
        dev_idx = _get_device_index(MIC_LIST[0]) if MIC_LIST else None
        if dev_idx is not None:
            cmd += ["--device", str(dev_idx)]
        if WAKE_LANG not in (None, "auto", ""):
            cmd += ["--lang", WAKE_LANG]

        log.info("Spawning wake_listener: %s", " ".join(str(c) for c in cmd))

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=1,              # line-buffered
            )
            _WAKE_PROC = proc

            # ── read JSON lines from subprocess stdout ────────────────────
            while WAKE_RUNNING and proc.poll() is None:
                raw = proc.stdout.readline()  # type: ignore[union-attr]
                if not raw:
                    break
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    data       = json.loads(line)
                    transcript = data.get("transcript", "").strip()
                    if transcript:
                        log.info("Wake command: %r", transcript)
                        response = _run_async(orchestrator.process(transcript))
                        WAKE_EVENTS.put((transcript, response))
                        _speak_sync(response)
                except json.JSONDecodeError:
                    log.debug("wake_listener non-JSON: %r", line)

            ret = proc.poll()
            if ret is not None and ret != 0:
                log.warning("wake_listener exited (code=%d) — restarting in 2 s", ret)
                time.sleep(2)

        except Exception as exc:
            log.error("Wake word loop error: %s", exc)
            time.sleep(2)
        finally:
            if _WAKE_PROC and _WAKE_PROC.poll() is None:
                _WAKE_PROC.terminate()
            _WAKE_PROC = None

    log.info("Wake word thread stopped.")


def start_wake_word() -> tuple[str, str]:
    """Start the background wake word detection thread."""
    global WAKE_RUNNING, WAKE_THREAD
    if WAKE_RUNNING:
        return "🔴 Wake Word OFF", "⚠ Already running"
    listener = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "voice", "wake_listener.py")
    if not os.path.exists(listener):
        return "🔊 Wake Word ON", "⚠ wake_listener.py not found"
    WAKE_RUNNING = True
    WAKE_THREAD  = threading.Thread(target=_wake_word_loop, daemon=True, name="WakeWord")
    WAKE_THREAD.start()
    return "🔴 Wake Word OFF", "✅ Listening — say 'Aida / Jarvis / Assistant …'"


def stop_wake_word() -> tuple[str, str]:
    """Stop the background wake word detection thread and kill subprocess."""
    global WAKE_RUNNING, _WAKE_PROC
    WAKE_RUNNING = False
    if _WAKE_PROC and _WAKE_PROC.poll() is None:
        _WAKE_PROC.terminate()
    return "🔊 Wake Word ON", "⏹ Wake word stopped"


def toggle_wake_word(btn_label: str) -> tuple[str, str]:
    if "ON" in btn_label:
        return start_wake_word()
    return stop_wake_word()


def poll_wake_events(history: list) -> list:
    """Called by gr.Timer every 500 ms — drains WAKE_EVENTS and appends to chat."""
    updated = list(history)
    try:
        while True:
            transcript, response = WAKE_EVENTS.get_nowait()
            updated = _append_exchange(updated, f"🎙️ {transcript}", response)
    except queue.Empty:
        pass
    return updated


# ── Text & voice handlers ────────────────────────────────────────────────────

def handle_text(user_message: str, history: list, mode: str):
    if not user_message.strip():
        yield history, "", core_markup(False)
        return

    history = _append_user(history, user_message)
    # Add empty assistant placeholder with streaming cursor
    history = history + [{"role": "assistant", "content": "▌"}]
    yield history, "", core_markup(False)

    full_text = ""
    try:
        for token in _stream_llm(user_message):
            full_text += token
            history[-1]["content"] = full_text + "▌"
            yield history, "", core_markup(False)
    except Exception as exc:
        log.error("handle_text stream error: %s", exc, exc_info=True)
        full_text = f"[Error: {exc}]"

    history[-1]["content"] = full_text
    if full_text and not full_text.startswith("[Error"):
        _speak_async(full_text)
    yield history, "", core_markup(True)


def handle_voice_capture(history: list, mode: str, muted: bool,
                          mic_choice: str = "Default microphone",
                          lang_choice: str = "auto"):
    """
    TALK button handler.
    Step 1: voice_worker subprocess records audio (fast, no model load).
    Step 2: persistent transcription_server transcribes (model always in GPU).
    Total latency: ~2-5 s instead of 40-55 s.
    """
    if mode != "Voice":
        yield history, "Voice mode is disabled.", core_markup(False)
        return
    if muted:
        history = _append_exchange(history, "🔇 [muted]", "Microphone muted. Unmute to speak.")
        yield history, "Muted.", core_markup(False)
        return

    worker = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "voice", "voice_worker.py")
    if not os.path.exists(worker):
        yield history, "⚠ voice_worker.py not found.", core_markup(False)
        return

    # Build record-only subprocess command (no --lang needed here)
    cmd = [sys.executable, worker, "--silence", "1.0"]
    dev_idx = _get_device_index(mic_choice)
    if dev_idx is not None:
        cmd += ["--device", str(dev_idx)]

    try:
        # ── Step 1: record audio (fast, ~1-3 s) ─────────────────────────────
        yield history, "⏺ Recording...", core_markup(False)
        log.info("Recording: %s", " ".join(str(c) for c in cmd))
        rec = subprocess.run(cmd, capture_output=True, timeout=60)

        if rec.returncode != 0:
            err = rec.stderr.decode("utf-8", errors="replace").strip() or "recorder error"
            log.error("voice_worker error: %s", err)
            yield history, f"Recording error: {err}", core_markup(False)
            return

        wav_path = rec.stdout.decode("utf-8", errors="replace").strip()
        if not wav_path or not os.path.exists(wav_path):
            yield history, "Nothing heard (no audio file).", core_markup(False)
            return

        # ── Step 2: transcribe via persistent server (~1-2 s on GPU) ────────
        yield history, "🔍 Transcribing...", core_markup(False)
        lang_arg = None if lang_choice in (None, "auto", "") else lang_choice
        log.info("Transcribing %s lang=%s ...", wav_path, lang_arg)
        transcript, _ = transcribe_audio_file(wav_path, lang_arg)
        log.info("Transcript: %r", transcript)

        if not transcript.strip():
            yield history, "Nothing heard.", core_markup(False)
            return

        history = _append_user(history, f"🎤 {transcript}")
        # Streaming assistant reply
        history = history + [{"role": "assistant", "content": "▌"}]
        yield history, "💭 Generating...", core_markup(False)

        full_text = ""
        try:
            for token in _stream_llm(transcript):
                full_text += token
                history[-1]["content"] = full_text + "▌"
                yield history, "💭 Generating...", core_markup(False)
        except Exception as exc:
            log.error("voice stream error: %s", exc)
            full_text = f"[Error: {exc}]"

        history[-1]["content"] = full_text
        if full_text and not full_text.startswith("[Error"):
            _speak_async(full_text)
        yield history, "✅ Done.", core_markup(True)
        return

    except subprocess.TimeoutExpired:
        yield history, "⚠ Voice recording timed out.", core_markup(False)
    except Exception as exc:
        log.error("handle_voice_capture error: %s", exc, exc_info=True)
        yield history, f"Voice error: {exc}", core_markup(False)


def clear_chat():
    orchestrator.buffer.clear()
    return []


def toggle_voice_ui(m: str):
    show = m == "Voice"
    return (
        gr.update(visible=show),          # voice_group
        gr.update(),                       # chatbox — no height change in wide layout
    )


def _set_env_value(env_path: Path, key: str, value: str):
    env_path.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if env_path.exists():
        existing = env_path.read_text(encoding="utf-8").splitlines()
    prefix   = f"{key}="
    replaced = False
    out      = []
    for line in existing:
        if line.startswith(prefix):
            out.append(f"{prefix}{value}")
            replaced = True
        else:
            out.append(line)
    if not replaced:
        out.append(f"{prefix}{value}")
    env_path.write_text("\n".join(out).strip() + "\n", encoding="utf-8")


def save_settings(gemini_key: str, openai_key: str):
    gemini_key = (gemini_key or "").strip()
    openai_key = (openai_key or "").strip()
    os.environ["GEMINI_API_KEY"] = gemini_key
    os.environ["OPENAI_API_KEY"] = openai_key
    orchestrator.selector.cloud.gemini_key = gemini_key
    orchestrator.selector.cloud.openai_key = openai_key
    orchestrator.selector.cloud.model_name = (
        "gemini-2.0-flash" if gemini_key else "gpt-4o-mini"
    )
    env_path = Path(__file__).resolve().parent / ".env"
    _set_env_value(env_path, "GEMINI_API_KEY", gemini_key)
    _set_env_value(env_path, "OPENAI_API_KEY", openai_key)
    return get_status(), "✅ Settings saved."




def handle_assistant_mode_change(label: str) -> str:
    """Switch AIDA personality mode from UI."""
    key = orchestrator.modes.from_label(label) or label.split()[-1].lower()
    profile = orchestrator.modes.set_mode(key)
    return get_status()


def get_plan_display() -> str:
    """Return current plan for brain panel."""
    if not orchestrator.planner.has_active_plan:
        return ""
    return orchestrator.planner.current_plan.to_display()


def get_suggestions_display() -> str:
    """Return predictive suggestions as compact string."""
    from config.feature_flags import Flags
    if not Flags.PREDICTIVE_SUGGESTIONS:
        return ""
    sugs = orchestrator.get_suggestions()
    if not sugs:
        return ""
    return "💡 " + "\n\n💡 ".join(s.text for s in sugs[:3])



# ── CSS ───────────────────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@300;400;500;600&family=Share+Tech+Mono&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Reset ─────────────────────────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
    --bg-void:      #05080d;
    --bg-base:      #080c12;
    --bg-panel:     #0c1018;
    --bg-surface:   #111827;
    --bg-surface2:  #161f2e;
    --bg-hover:     #1a2436;
    --cyan:         #00e5ff;
    --cyan-dim:     rgba(0,229,255,0.12);
    --cyan-glow:    rgba(0,229,255,0.35);
    --cyan-soft:    rgba(0,229,255,0.07);
    --indigo:       #7c8fff;
    --indigo-dim:   rgba(124,143,255,0.12);
    --green:        #00ffaa;
    --green-dim:    rgba(0,255,170,0.1);
    --amber:        #ffcc00;
    --red:          #ff4466;
    --text-hi:      #e8f0f8;
    --text-mid:     #8fa8c0;
    --text-lo:      #3d5470;
    --border:       rgba(0,229,255,0.08);
    --border-mid:   rgba(0,229,255,0.18);
    --border-hi:    rgba(0,229,255,0.4);
    --radius-sm:    6px;
    --radius-md:    10px;
    --radius-lg:    16px;
    --font-ui:      'DM Sans', system-ui, sans-serif;
    --font-mono:    'Share Tech Mono', monospace;
    --font-display: 'Rajdhani', sans-serif;
    --shadow-panel: 0 0 0 1px var(--border), 0 8px 32px rgba(0,0,0,0.6);
}

html, body {
    background: var(--bg-void) !important;
    height: 100%;
    overflow: hidden;
    font-family: var(--font-ui);
}

footer, .built-with, .svelte-1ed2p3z { display: none !important; }

/* ── Container ──────────────────────────────────────────────────────────────── */
.gradio-container {
    max-width: 100% !important;
    width: 100% !important;
    min-height: 100vh !important;
    background: var(--bg-void) !important;
    padding: 0 !important;
    margin: 0 !important;
    overflow: hidden !important;
}
.gradio-container > .main > .wrap { padding: 0 !important; gap: 0 !important; }

/* ── Scanline overlay ───────────────────────────────────────────────────────── */
body::after {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(0,0,0,0.03) 2px,
        rgba(0,0,0,0.03) 4px
    );
    z-index: 9999;
}

/* ── TOP BAR ────────────────────────────────────────────────────────────────── */
.topbar {
    height: 54px !important;
    min-height: 54px !important;
    max-height: 54px !important;
    background: linear-gradient(90deg, var(--bg-panel) 0%, var(--bg-base) 50%, var(--bg-panel) 100%) !important;
    border-bottom: 1px solid var(--border-mid) !important;
    display: flex !important;
    align-items: center !important;
    padding: 0 20px !important;
    gap: 0 !important;
}
.topbar > .wrap { display: flex; align-items: center; width: 100%; gap: 0 !important; padding: 0 !important; }

/* ── TOP BAR HTML ───────────────────────────────────────────────────────────── */
.topbar-html {
    flex: 0 0 auto;
}
.aida-wordmark {
    font-family: var(--font-display);
    font-size: 1.5rem;
    font-weight: 600;
    letter-spacing: 8px;
    color: var(--cyan);
    text-shadow: 0 0 20px var(--cyan-glow), 0 0 40px rgba(0,229,255,0.15);
    display: inline-flex;
    align-items: center;
    gap: 10px;
    white-space: nowrap;
}
.aida-wordmark .dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--cyan);
    box-shadow: 0 0 10px var(--cyan), 0 0 20px var(--cyan-glow);
    animation: pulse-dot 2.4s ease-in-out infinite;
}
@keyframes pulse-dot {
    0%, 100% { opacity: 1; box-shadow: 0 0 10px var(--cyan), 0 0 20px var(--cyan-glow); }
    50%       { opacity: 0.4; box-shadow: 0 0 4px var(--cyan); }
}
.aida-tagline {
    font-family: var(--font-mono);
    font-size: 0.6rem;
    color: var(--text-lo);
    letter-spacing: 2px;
    text-transform: uppercase;
    display: block;
    margin-top: 1px;
}

/* status in topbar */
.topbar-status { flex: 1; }
.topbar-status p {
    text-align: center !important;
    font-family: var(--font-mono) !important;
    font-size: 0.65rem !important;
    color: rgba(0,229,255,0.45) !important;
    letter-spacing: 1.5px !important;
    margin: 0 !important;
}

/* ── THREE-PANEL LAYOUT ─────────────────────────────────────────────────────── */
.aida-layout > .wrap {
    display: flex !important;
    height: calc(100vh - 54px) !important;
    overflow: hidden !important;
    padding: 0 !important;
    gap: 0 !important;
    align-items: stretch !important;
}

/* ── LEFT PANEL ─────────────────────────────────────────────────────────────── */
.panel-left {
    width: 220px !important;
    min-width: 220px !important;
    max-width: 220px !important;
    flex-shrink: 0 !important;
    background: var(--bg-panel) !important;
    border-right: 1px solid var(--border) !important;
    display: flex !important;
    flex-direction: column !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    padding: 14px 10px !important;
    gap: 14px !important;
}
.panel-left > .wrap { display: flex; flex-direction: column; gap: 14px; padding: 0; }

.panel-section-label {
    font-family: var(--font-mono);
    font-size: 0.55rem;
    letter-spacing: 2.5px;
    color: var(--text-lo);
    text-transform: uppercase;
    padding-bottom: 4px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2px;
}

/* Mode pills in left panel */
.mode-pills .wrap { display: flex !important; flex-direction: column !important; gap: 3px !important; padding: 0 !important; }
.mode-pills label {
    width: 100% !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    background: transparent !important;
    padding: 5px 8px !important;
    transition: all 0.15s ease !important;
    cursor: pointer !important;
}
.mode-pills label:hover { background: var(--cyan-soft) !important; border-color: var(--border-mid) !important; }
.mode-pills label.selected {
    background: var(--cyan-dim) !important;
    border-color: var(--border-hi) !important;
    box-shadow: 0 0 12px var(--cyan-glow), inset 0 0 8px rgba(0,229,255,0.05) !important;
}
.mode-pills label span {
    font-size: 0.7rem !important;
    font-family: var(--font-ui) !important;
    color: var(--text-mid) !important;
    font-weight: 400 !important;
    letter-spacing: 0.3px !important;
}
.mode-pills label.selected span { color: var(--cyan) !important; }

/* Skill dropdown */
.skill-drop select, .skill-drop .wrap-inner {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-mid) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.68rem !important;
    border-radius: var(--radius-sm) !important;
    padding: 5px 8px !important;
}
.skill-drop label span { font-size: 0.6rem !important; color: var(--text-lo) !important; letter-spacing: 1px !important; }

/* Suggestions area */
.suggestions-area p, .suggestions-area span {
    font-family: var(--font-ui) !important;
    font-size: 0.68rem !important;
    color: var(--text-mid) !important;
    line-height: 1.5 !important;
}

/* Shadow checkbox */
.shadow-toggle label { font-size: 0.68rem !important; color: var(--text-lo) !important; font-family: var(--font-mono) !important; }
.shadow-toggle input[type=checkbox] { accent-color: var(--indigo) !important; }

/* ── CENTER PANEL ────────────────────────────────────────────────────────────── */
.panel-center {
    flex: 1 !important;
    min-width: 0 !important;
    background: var(--bg-base) !important;
    display: flex !important;
    flex-direction: column !important;
    overflow: hidden !important;
    position: relative !important;
}
.panel-center > .wrap { display: flex; flex-direction: column; height: 100%; padding: 0; gap: 0; }

/* Chat area */
.aida-chat {
    flex: 1 !important;
    background: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    min-height: 0 !important;
    overflow-y: auto !important;
    padding: 12px 18px !important;
}
.aida-chat .message-wrap { gap: 10px !important; }
.aida-chat [data-testid="user"] .bubble-wrap {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-mid) !important;
    color: var(--text-hi) !important;
    font-family: var(--font-ui) !important;
    font-size: 0.84rem !important;
    border-radius: var(--radius-md) !important;
    padding: 10px 14px !important;
    box-shadow: none !important;
}
.aida-chat [data-testid="bot"] .bubble-wrap {
    background: var(--bg-panel) !important;
    border: 1px solid var(--border) !important;
    color: #c8daea !important;
    font-family: var(--font-ui) !important;
    font-size: 0.84rem !important;
    border-radius: var(--radius-md) !important;
    padding: 10px 14px !important;
    border-left: 2px solid rgba(0,229,255,0.3) !important;
}

/* Input area */
.input-area {
    background: var(--bg-panel) !important;
    border-top: 1px solid var(--border) !important;
    padding: 12px 16px !important;
    flex-shrink: 0 !important;
}
.input-area > .wrap { padding: 0 !important; gap: 8px !important; }
.aida-input textarea {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-mid) !important;
    border-radius: var(--radius-md) !important;
    color: var(--text-hi) !important;
    font-family: var(--font-ui) !important;
    font-size: 0.84rem !important;
    resize: none !important;
    padding: 10px 14px !important;
    min-height: 44px !important;
}
.aida-input textarea:focus {
    border-color: var(--border-hi) !important;
    box-shadow: 0 0 0 2px var(--cyan-dim), 0 0 16px var(--cyan-soft) !important;
    outline: none !important;
}
.aida-input textarea::placeholder { color: var(--text-lo) !important; }
.send-btn button {
    background: var(--cyan-dim) !important;
    border: 1px solid var(--border-hi) !important;
    border-radius: var(--radius-md) !important;
    color: var(--cyan) !important;
    font-size: 0.8rem !important;
    font-family: var(--font-mono) !important;
    height: 44px !important;
    min-width: 52px !important;
    letter-spacing: 1px !important;
    transition: all 0.15s !important;
}
.send-btn button:hover {
    background: rgba(0,229,255,0.22) !important;
    box-shadow: 0 0 16px var(--cyan-glow) !important;
}

/* Core orb (small, centered in chat header) */
.core-orb-wrap { display: flex; justify-content: center; padding: 10px 0 4px; }
.core-orb-outer {
    width: 56px; height: 56px;
    border-radius: 50%;
    border: 1px solid rgba(0,229,255,0.2);
    background: radial-gradient(circle, rgba(0,229,255,0.08) 0%, transparent 70%);
    display: flex; align-items: center; justify-content: center;
    position: relative;
    box-shadow: 0 0 20px rgba(0,229,255,0.08);
}
.core-orb-outer::before {
    content: "";
    position: absolute; inset: 8px;
    border-radius: 50%;
    border: 1px dashed rgba(0,229,255,0.12);
    animation: spin-ring 12s linear infinite;
}
@keyframes spin-ring { to { transform: rotate(360deg); } }
.core-orb-inner {
    width: 18px; height: 18px;
    border-radius: 50%;
    background: radial-gradient(circle, #fff 0%, var(--cyan) 50%, transparent 100%);
    box-shadow: 0 0 12px var(--cyan), 0 0 24px var(--cyan-glow);
}
.core-active .core-orb-outer { box-shadow: 0 0 30px var(--cyan-glow); border-color: rgba(0,229,255,0.5); }
.core-active .core-orb-inner { box-shadow: 0 0 20px var(--cyan), 0 0 40px var(--cyan-glow); animation: throb 0.8s ease-in-out infinite alternate; }
@keyframes throb { from { opacity: 0.7; } to { opacity: 1; } }

/* ── RIGHT PANEL ─────────────────────────────────────────────────────────────── */
.panel-right {
    width: 240px !important;
    min-width: 240px !important;
    max-width: 240px !important;
    flex-shrink: 0 !important;
    background: var(--bg-panel) !important;
    border-left: 1px solid var(--border) !important;
    display: flex !important;
    flex-direction: column !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    padding: 14px 10px !important;
    gap: 10px !important;
}
.panel-right > .wrap { display: flex; flex-direction: column; gap: 10px; padding: 0; }

/* Input mode toggle (Text/Voice) */
.input-mode .wrap { gap: 4px !important; justify-content: center !important; }
.input-mode label {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    background: transparent !important;
    padding: 4px 16px !important;
    flex: 1 !important;
    text-align: center !important;
    transition: all 0.15s !important;
}
.input-mode label:hover { background: var(--cyan-soft) !important; }
.input-mode label.selected { background: var(--indigo-dim) !important; border-color: rgba(124,143,255,0.5) !important; }
.input-mode label span { font-size: 0.7rem !important; color: var(--text-mid) !important; letter-spacing: 0.5px !important; }
.input-mode label.selected span { color: var(--indigo) !important; }

/* Voice panel */
.aida-voice-panel {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    padding: 10px !important;
}
.aida-voice-panel .markdown-body p { font-size: 0.65rem !important; color: var(--text-lo) !important; }
.aida-voice-panel select, .aida-voice-panel .wrap-inner {
    background: var(--bg-surface2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-mid) !important;
    font-size: 0.68rem !important;
    border-radius: var(--radius-sm) !important;
}
.aida-voice-panel button {
    background: var(--bg-surface2) !important;
    border: 1px solid var(--border-mid) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-mid) !important;
    font-size: 0.7rem !important;
    transition: all 0.15s !important;
}
.aida-voice-panel button:hover { background: var(--indigo-dim) !important; border-color: rgba(124,143,255,0.4) !important; color: var(--indigo) !important; }

.wake-btn button {
    width: 100% !important;
    background: rgba(129,140,248,0.07) !important;
    border: 1px solid rgba(129,140,248,0.3) !important;
    color: var(--indigo) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.68rem !important;
    letter-spacing: 1px !important;
    border-radius: var(--radius-sm) !important;
}
.wake-btn button:hover { background: var(--indigo-dim) !important; }

/* Talk button */
.aida-voice-panel .gr-button:first-of-type,
.talk-btn button {
    background: rgba(0,255,170,0.07) !important;
    border: 1px solid rgba(0,255,170,0.3) !important;
    color: var(--green) !important;
}
.talk-btn button:hover { background: var(--green-dim) !important; box-shadow: 0 0 12px rgba(0,255,170,0.2) !important; }

/* Plan display */
.plan-area p, .plan-area span, .plan-area li {
    font-family: var(--font-mono) !important;
    font-size: 0.65rem !important;
    color: var(--text-mid) !important;
    line-height: 1.6 !important;
}
.plan-area strong { color: var(--cyan) !important; }

/* Goal status */
.goal-status textarea, .goal-status input {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border) !important;
    color: rgba(0,255,170,0.7) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.65rem !important;
    border-radius: var(--radius-sm) !important;
}
.goal-status label span { font-size: 0.6rem !important; color: var(--text-lo) !important; }

/* Settings accordion */
.aida-settings {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
}
.aida-settings .label-wrap button { font-size: 0.7rem !important; color: var(--text-lo) !important; background: transparent !important; }
.aida-settings input, .aida-settings select {
    background: var(--bg-surface2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-mid) !important;
    font-size: 0.7rem !important;
    border-radius: var(--radius-sm) !important;
}
.aida-settings label span { font-size: 0.65rem !important; color: var(--text-lo) !important; }
.aida-settings-brain {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
}
.aida-settings-brain .label-wrap button { font-size: 0.7rem !important; color: var(--text-lo) !important; background: transparent !important; }
.aida-settings-status textarea { color: rgba(0,255,170,0.7) !important; font-size: 0.68rem !important; }

/* Clear button */
.aida-clear button {
    width: 100% !important;
    background: transparent !important;
    border: 1px solid rgba(255,68,102,0.15) !important;
    color: rgba(255,68,102,0.4) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.6rem !important;
    letter-spacing: 2px !important;
    border-radius: var(--radius-sm) !important;
    padding: 5px !important;
    transition: all 0.15s !important;
}
.aida-clear button:hover {
    border-color: rgba(255,68,102,0.5) !important;
    color: rgba(255,68,102,0.8) !important;
    background: rgba(255,68,102,0.05) !important;
}

/* Voice status + misc textboxes */
textarea[data-testid], .gr-textbox textarea {
    background: var(--bg-surface2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-mid) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.68rem !important;
    border-radius: var(--radius-sm) !important;
}

/* Lang radio inside voice panel */
.aida-lang-radio .wrap { gap: 4px !important; }
.aida-lang-radio label { padding: 2px 8px !important; border-radius: var(--radius-sm) !important; font-size: 0.65rem !important; }
.aida-lang-radio label span { color: var(--text-lo) !important; font-size: 0.65rem !important; }
.aida-lang-radio label.selected span { color: var(--indigo) !important; }
.aida-lang-radio label.selected { background: var(--indigo-dim) !important; border-color: rgba(124,143,255,0.35) !important; }

/* Hide Gradio labels on panels that don't need them */
.panel-left label > span.svelte-1gfkn6j,
.panel-right label > span.svelte-1gfkn6j { display: none !important; }

/* Scrollbars */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border-mid); border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: var(--border-hi); }

/* ── Hide label on specific elements ─────────────────────────────────────────── */
.hide-label label, .hide-label > div > label { display: none !important; }
"""


# ── Runtime globals ───────────────────────────────────────────────────────────
MIC_LIST     = get_microphones()
log.info("Microphones: %s", MIC_LIST)
LANG_CHOICES = ["auto", "ru", "en"]
LANG_DEFAULT = "auto"
WAKE_LANG    = "auto"   # updated by UI language selector

# Pre-warm transcription server and TTS server in background
_preload_transcription_server()
_preload_tts_server()

CHAT_HEIGHT = 460
PORT        = 7860

log.info("Building Gradio UI (gr.Blocks)...")
with gr.Blocks(title="AIDA", fill_width=True) as demo:

    # ── TOP BAR ───────────────────────────────────────────────────────────────
    with gr.Row(elem_classes=["topbar"]):
        gr.HTML("""
        <div class="topbar-html">
            <span class="aida-wordmark">
                <span class="dot"></span>AIDA
            </span>
            <span class="aida-tagline">AI DESKTOP ASSISTANT</span>
        </div>
        """)
        status_md = gr.Markdown(get_status(),
                                elem_classes=["topbar-status", "hide-label"])

    # ── THREE-PANEL MAIN AREA ─────────────────────────────────────────────────
    with gr.Row(elem_classes=["aida-layout"], equal_height=False):

        # ══════════════════════════════════════════════════════════════════════
        # LEFT PANEL — Context / Mode / Skill
        # ══════════════════════════════════════════════════════════════════════
        with gr.Column(scale=1, elem_classes=["panel-left"]):

            gr.HTML('<div class="panel-section-label">PERSONALITY</div>')
            assistant_mode = gr.Radio(
                choices    = orchestrator.modes.mode_labels(),
                value      = orchestrator.modes.mode_labels()[0],
                label      = "",
                show_label = False,
                container  = False,
                elem_classes=["mode-pills"],
            )

            gr.HTML('<div class="panel-section-label" style="margin-top:6px">SKILL</div>')
            skill_select = gr.Dropdown(
                choices     = orchestrator.skills.skill_labels(),
                value       = orchestrator.skills.skill_labels()[0],
                label       = "",
                show_label  = False,
                interactive = True,
                elem_classes= ["skill-drop"],
            )

            gr.HTML('<div class="panel-section-label" style="margin-top:6px">SUGGESTIONS</div>')
            suggest_display = gr.Markdown("", elem_classes=["suggestions-area"])

            gr.HTML('<div class="panel-section-label" style="margin-top:6px">OBSERVE</div>')
            shadow_chk = gr.Checkbox(
                label="Shadow Mode",
                value=False, interactive=True,
                elem_classes=["shadow-toggle"],
            )
            gr.Markdown(
                "<span style='font-size:0.55rem;color:#2a3a4a;line-height:1.4'>"
                "Logs app patterns locally.<br>Never captures screen.</span>"
            )

        # ══════════════════════════════════════════════════════════════════════
        # CENTER PANEL — Main Interaction
        # ══════════════════════════════════════════════════════════════════════
        with gr.Column(scale=3, elem_classes=["panel-center"]):

            core_view = gr.HTML(core_markup(False))

            chatbox = gr.Chatbot(
                label="",
                height=CHAT_HEIGHT,
                min_height=CHAT_HEIGHT,
                show_label=False,
                elem_classes=["aida-chat"],
                avatar_images=(None, None),
                autoscroll=False,
            )

            with gr.Row(equal_height=True, elem_classes=["input-area"]) as text_row:
                text_in = gr.Textbox(
                    placeholder="Message AIDA...",
                    show_label=False,
                    scale=9,
                    container=False,
                    elem_classes=["aida-input"],
                    lines=1,
                    max_lines=3,
                )
                send_btn = gr.Button("⏎", scale=1, variant="secondary",
                                     size="sm", elem_classes=["send-btn"])

        # ══════════════════════════════════════════════════════════════════════
        # RIGHT PANEL — Tools / Plan / Voice / Settings
        # ══════════════════════════════════════════════════════════════════════
        with gr.Column(scale=1, elem_classes=["panel-right"]):

            gr.HTML('<div class="panel-section-label">INPUT MODE</div>')
            mode = gr.Radio(
                choices=["Text", "Voice"],
                value="Voice",
                show_label=False,
                container=False,
                elem_classes=["input-mode"],
            )

            # ── Voice panel ──────────────────────────────────────────────────
            with gr.Group(visible=True, elem_classes=["aida-voice-panel"]) as voice_group:
                mic_device = gr.Dropdown(
                    choices=MIC_LIST,
                    value=MIC_LIST[0] if MIC_LIST else "Default microphone",
                    label="🎙 Mic",
                    show_label=True,
                    interactive=True,
                    elem_classes=["aida-mic-select"],
                )
                lang_radio = gr.Radio(
                    choices=LANG_CHOICES, value=LANG_DEFAULT,
                    label="🌐 Lang", interactive=True,
                    elem_classes=["aida-lang-radio"],
                )
                with gr.Row(equal_height=True):
                    talk_btn  = gr.Button("🎤 TALK", variant="secondary",
                                          size="sm", elem_classes=["talk-btn"])
                    mic_muted = gr.Checkbox(label="Mute", value=False)
                voice_state = gr.Textbox(label="Status", value="Idle",
                                         interactive=False, lines=1)
                wake_btn = gr.Button("🔊 Wake ON", variant="secondary",
                                     size="sm", elem_classes=["wake-btn"])
                wake_status_txt = gr.Textbox(label="", value="",
                                              interactive=False, lines=1,
                                              elem_classes=["hide-label"])

            # ── Plan / Goal ───────────────────────────────────────────────────
            gr.HTML('<div class="panel-section-label" style="margin-top:4px">PLAN</div>')
            plan_display = gr.Markdown("", elem_classes=["plan-area"])

            with gr.Row():
                goal_status_txt = gr.Textbox(
                    label="Goal", value="", interactive=False,
                    lines=1, placeholder="No active goal",
                    elem_classes=["goal-status"],
                )
                abort_goal_btn = gr.Button("✗", size="sm", variant="secondary",
                                           elem_classes=["aida-clear"])

            # ── Settings ─────────────────────────────────────────────────────
            with gr.Accordion("⚙ Config", open=False, elem_classes=["aida-settings"]):
                gemini_key = gr.Textbox(
                    label="Gemini Key", type="password",
                    value=os.getenv("GEMINI_API_KEY", ""), placeholder="AIza...",
                )
                openai_key = gr.Textbox(
                    label="OpenAI Key", type="password",
                    value=os.getenv("OPENAI_API_KEY", ""), placeholder="sk-...",
                )
                save_settings_btn = gr.Button("Save", variant="secondary", size="sm")
                settings_status = gr.Textbox(
                    label="", value="", interactive=False, lines=1,
                    elem_classes=["aida-settings-status"],
                )

            # ── Clear ─────────────────────────────────────────────────────────
            clear_btn = gr.Button("CLEAR", variant="secondary",
                                   size="sm", elem_classes=["aida-clear"])


    # ── Timer: polls WAKE_EVENTS every 500 ms ─────────────────────────────────
    timer = gr.Timer(0.5)

    # ── Wiring ───────────────────────────────────────────────────────────────

    submit_cfg = dict(
        fn=handle_text,
        inputs=[text_in, chatbox, mode],
        outputs=[chatbox, text_in, core_view],
    )
    text_in.submit(**submit_cfg)
    send_btn.click(**submit_cfg)

    talk_btn.click(
        fn=handle_voice_capture,
        inputs=[chatbox, mode, mic_muted, mic_device, lang_radio],
        outputs=[chatbox, voice_state, core_view],
    )

    def _update_wake_lang(lang):
        global WAKE_LANG
        WAKE_LANG = lang
        return lang
    lang_radio.change(fn=_update_wake_lang, inputs=lang_radio, outputs=lang_radio)

    wake_btn.click(
        fn=toggle_wake_word,
        inputs=[wake_btn],
        outputs=[wake_btn, wake_status_txt],
    )

    timer.tick(
        fn=poll_wake_events,
        inputs=[chatbox],
        outputs=[chatbox],
    )

    clear_btn.click(fn=clear_chat, outputs=[chatbox])

    mode.change(
        fn=toggle_voice_ui,
        inputs=mode,
        outputs=[voice_group, chatbox],
    )

    save_settings_btn.click(
        fn=save_settings,
        inputs=[gemini_key, openai_key],
        outputs=[status_md, settings_status],
    )

    assistant_mode.change(
        fn=handle_assistant_mode_change,
        inputs=[assistant_mode],
        outputs=[status_md],
    )

    def _toggle_shadow(enabled: bool) -> None:
        if enabled:
            orchestrator.shadow.enable()
        else:
            orchestrator.shadow.disable()
    shadow_chk.change(fn=_toggle_shadow, inputs=[shadow_chk], outputs=[])

    def _change_skill(label: str) -> str:
        key = orchestrator.skills.from_label(label)
        if key:
            orchestrator.skills.clear()
            if key != "none":
                orchestrator.skills.activate(key)
        return get_status()
    skill_select.change(fn=_change_skill, inputs=[skill_select], outputs=[status_md])

    def _abort_goal() -> str:
        msg = orchestrator.goal_engine.abort()
        return msg
    abort_goal_btn.click(fn=_abort_goal, outputs=[goal_status_txt])

    # Refresh plan + suggestions + goal status every 2 s
    plan_timer = gr.Timer(2.0)
    plan_timer.tick(fn=get_plan_display,          outputs=[plan_display])
    plan_timer.tick(fn=get_suggestions_display,    outputs=[suggest_display])
    plan_timer.tick(fn=orchestrator.get_goal_status, outputs=[goal_status_txt])

log.info("Gradio UI built successfully.")


# ── Launch ─────────────────────────────────────────────────────────────────────

def _start_gradio():
    log.info("Gradio server thread starting on 127.0.0.1:%d ...", PORT)
    try:
        demo.queue().launch(
            server_name="127.0.0.1",
            server_port=PORT,
            prevent_thread_lock=True,
            show_error=True,
            quiet=False,
            inbrowser=False,
            css=CSS,
        )
        log.info("Gradio server launched (non-blocking).")
    except Exception as _e:
        log.critical("Gradio launch() FAILED: %s", _e, exc_info=True)


if __name__ == "__main__":
    log.info("=" * 50)
    log.info("AIDA __main__ reached — startup proceeding.")
    log.info("Python %s | Platform: %s", sys.version.split()[0], sys.platform)
    log.info("Gradio version: %s", gr.__version__)
    log.info("=" * 50)

    log.info("Attempting to import pywebview...")
    try:
        import webview  # pywebview ≥ 4.x
        log.info("pywebview found (version: %s). Using native window.", getattr(webview, '__version__', '?'))

        log.info("Starting Gradio server thread...")
        t = threading.Thread(target=_start_gradio, daemon=True)
        t.start()
        log.info("Waiting 2s for Gradio to bind to port %d...", PORT)
        time.sleep(2)

        log.info("Opening native window 1100×720 at http://127.0.0.1:%d", PORT)
        try:
            webview.create_window(
                title="AIDA",
                url=f"http://127.0.0.1:{PORT}",
                width=1100,
                height=720,
                resizable=True,
                background_color="#0a0c0f",
            )
            log.info("webview.create_window() OK. Calling webview.start()...")
            webview.start(debug=False)
            log.info("webview.start() returned — window closed.")
        except Exception as _wv_err:
            log.critical("pywebview error: %s", _wv_err, exc_info=True)
            log.warning("Falling back to browser launch...")
            demo.queue().launch(
                server_name="0.0.0.0",
                server_port=PORT,
                inbrowser=True,
                show_error=True,
                quiet=False,
                )

    except ImportError:
        log.warning("pywebview not installed — launching in browser instead.")
        log.info("Calling demo.queue().launch() on 0.0.0.0:%d ...", PORT)
        try:
            demo.queue().launch(
                server_name="0.0.0.0",
                server_port=PORT,
                inbrowser=True,
                show_error=True,
                quiet=False,
                css=CSS,
            )
        except Exception as _launch_err:
            log.critical("demo.launch() FAILED: %s", _launch_err, exc_info=True)
            raise
