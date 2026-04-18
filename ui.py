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
import logging
import os
import queue
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


def get_microphones() -> list[str]:
    try:
        import sounddevice as sd
        devices   = sd.query_devices()
        default_in, _ = sd.default.device
        seen: set[str] = set()
        mics: list[str] = []
        if isinstance(default_in, int) and default_in >= 0:
            d0: dict = sd.query_devices(default_in)  # type: ignore[assignment]
            if d0.get("max_input_channels", 0) > 0:
                name0 = str(d0["name"]).strip()
                seen.add(name0.lower())
                mics.append(f"{default_in}: {name0}")
        for i, dev in enumerate(devices):
            d: dict = dev  # type: ignore[assignment]
            if d.get("max_input_channels", 0) <= 0:
                continue
            name = str(d["name"]).strip()
            if name.lower() in seen:
                continue
            seen.add(name.lower())
            mics.append(f"{i}: {name}")
            if len(mics) >= 5:
                break
        return mics if mics else ["Default microphone"]
    except Exception:
        return ["Default microphone"]


def get_status() -> str:
    local_ok = orchestrator.selector.local.is_available()
    cloud_ok = orchestrator.selector.cloud.is_available()
    mem      = orchestrator.vector_store.count()
    parts: list[str] = []
    if local_ok:
        parts.append(f"⬡ {orchestrator.selector.local.model_name}")
    if cloud_ok:
        parts.append(f"☁ {orchestrator.selector.cloud.model_name}")
    parts.append(f"🧠 {mem}")
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

def _wake_word_loop():
    """
    Runs in a daemon thread.
    1. Waits for "Aida / Аида" via Whisper-based keyword spotting (VoiceListener).
    2. Records the following utterance with VAD (stops on silence).
    3. Sends to orchestrator.
    4. Puts (transcript, response) into WAKE_EVENTS queue for UI polling.
    5. Optionally speaks the response via TTS.
    """
    global WAKE_RUNNING
    log.info("Wake word thread started.")

    _vl = _get_voice_listener()
    if not _vl or not _vl.is_available():
        log.warning("VoiceListener not available — wake word thread exiting.")
        WAKE_RUNNING = False
        return

    while WAKE_RUNNING:
        try:
            transcript = _run_async(
                _vl.listen_with_wake_word(
                    wake_keywords=["aida", "аида", "aide", "айда"]
                )
            )
            if not transcript.strip():
                continue

            log.info("Wake utterance: %r", transcript)
            response = _run_async(orchestrator.process(transcript))
            WAKE_EVENTS.put((transcript, response))

            # Optional TTS output
            if os.getenv("AIDA_WAKE_TTS", "1") == "1":
                speak_text(response)

        except Exception as exc:
            if WAKE_RUNNING:
                log.error("Wake word loop error: %s", exc)
                time.sleep(1)   # brief pause before retrying

    log.info("Wake word thread stopped.")


def start_wake_word() -> tuple[str, str]:
    """Start the background wake word detection thread."""
    global WAKE_RUNNING, WAKE_THREAD
    if WAKE_RUNNING:
        return "🔴 Wake Word OFF", "⚠ Already running"
    _vl = _get_voice_listener()
    if not _vl or not _vl.is_available():
        return "🔊 Wake Word ON", "⚠ faster-whisper not installed"
    WAKE_RUNNING = True
    WAKE_THREAD  = threading.Thread(target=_wake_word_loop, daemon=True, name="WakeWord")
    WAKE_THREAD.start()
    return "🔴 Wake Word OFF", "✅ Wake word listening active — say 'Aida'"


def stop_wake_word() -> tuple[str, str]:
    """Stop the background wake word detection thread."""
    global WAKE_RUNNING
    WAKE_RUNNING = False
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
    yield history, "", core_markup(False)
    response = _run_async(orchestrator.process(user_message))
    history = _append_bot(history, response)
    yield history, "", core_markup(True)


def handle_voice_capture(history: list, mode: str, muted: bool):
    """
    TALK button handler.
    Runs recording + transcription in a SEPARATE SUBPROCESS so that
    CTranslate2/faster-whisper C-level crashes cannot kill the main app.
    """
    if mode != "Voice":
        return history, "Voice mode is disabled.", core_markup(False)
    if muted:
        history = _append_exchange(history, "🔇 [muted]", "Microphone muted. Unmute to speak.")
        return history, "Muted.", core_markup(False)

    import subprocess
    worker = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voice", "voice_worker.py")

    if not os.path.exists(worker):
        return history, "⚠ voice_worker.py not found.", core_markup(False)

    try:
        log.info("Spawning voice_worker subprocess...")
        result = subprocess.run(
            [sys.executable, worker],
            capture_output=True,
            timeout=120,
        )
        stderr_txt = result.stderr.decode("utf-8", errors="replace").strip()
        if result.returncode != 0:
            err = stderr_txt or "voice_worker exited with non-zero code"
            log.error("voice_worker failed: %s", err)
            history = _append_exchange(history, "🎤 [error]", f"Voice error: {err}")
            return history, f"Voice error: {err}", core_markup(False)

        transcript = result.stdout.decode("utf-8", errors="replace").strip()
        log.info("voice_worker transcript: %r", transcript)

        if not transcript:
            return history, "Nothing heard.", core_markup(False)

        history  = _append_user(history, f"🎤 {transcript}")
        response = _run_async(orchestrator.process(transcript))
        history  = _append_bot(history, response)
        return history, "✅ Done.", core_markup(True)

    except subprocess.TimeoutExpired:
        return history, "⚠ Voice recording timed out.", core_markup(False)
    except Exception as exc:
        log.error("handle_voice_capture error: %s", exc, exc_info=True)
        return history, f"Voice error: {exc}", core_markup(False)


def clear_chat():
    orchestrator.buffer.clear()
    return []


def toggle_voice_ui(m: str):
    show = m == "Voice"
    chat_h = 150 if show else 320
    return (
        gr.update(visible=show),                       # voice_group
        gr.update(height=chat_h, min_height=chat_h),  # chatbox
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


# ── Custom CSS ────────────────────────────────────────────────────────────────

CSS = """
/* ── Reset / root ──────────────────────────────────────── */
html, body {
    overflow: hidden !important;
    height: 100% !important;
    margin: 0;
    padding: 0;
    background: #120c0c !important;
    font-family: 'Inter', 'Segoe UI', sans-serif;
}

.gradio-container {
    max-width: 430px !important;
    width: 430px !important;
    min-height: 900px !important;
    max-height: 900px !important;
    overflow-x: hidden !important;
    overflow-y: auto !important;
    background: linear-gradient(180deg, #1a1010 0%, #0f0808 100%) !important;
    margin: 0 !important;
    padding: 0 10px !important;
    box-sizing: border-box;
}
.gradio-container::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    background-image: linear-gradient(#9bd9b208 1px, transparent 1px),
                      linear-gradient(90deg, #9bd9b208 1px, transparent 1px);
    background-size: 28px 28px;
    mask-image: radial-gradient(circle at center, black 40%, transparent 95%);
}

footer { display: none !important; }
.built-with { display: none !important; }

/* ── Header ─────────────────────────────────────────────── */
.aida-header {
    text-align: center;
    padding: 14px 0 4px;
    letter-spacing: 6px;
    font-size: 1.45rem;
    font-weight: 800;
    color: #c87a7a;
    text-shadow: 0 0 18px #c87a7a55;
}
.aida-sub {
    text-align: center;
    font-size: 0.6rem;
    letter-spacing: 3px;
    color: #ffffff28;
    margin-bottom: 6px;
}

.jarvis-core {
    display: flex;
    justify-content: center;
    margin: 8px 0 10px;
}
.jarvis-core-ring {
    width: 160px;
    height: 160px;
    border-radius: 999px;
    border: 1px solid #a44f4f6a;
    background: radial-gradient(circle, #8f3a3a35 0%, #1a0d0d00 72%);
    box-shadow: inset 0 0 24px #8f3a3a2b, 0 0 26px #00000055;
    position: relative;
}
.jarvis-core-ring::before {
    content: "";
    position: absolute;
    inset: 18px;
    border-radius: 999px;
    border: 1px dashed #b35a5a3d;
}
.jarvis-core-dot {
    position: absolute;
    inset: 58px;
    border-radius: 999px;
    background: radial-gradient(circle, #ffdede 0%, #d37474 52%, #d3747400 100%);
}
.core-active .jarvis-core-ring {
    box-shadow: inset 0 0 34px #c24d4d5f, 0 0 34px #c24d4d57;
}
.core-active .jarvis-core-dot {
    box-shadow: 0 0 22px #ff8f8f99;
}

/* ── Status bar ─────────────────────────────────────────── */
.aida-status p {
    text-align: center;
    font-size: 0.65rem;
    color: #c87a7a75;
    letter-spacing: 1px;
    margin: 0 !important;
    padding: 2px 0 4px !important;
}

/* ── Chatbot ─────────────────────────────────────────────── */
.aida-chat {
    background: #1a171acc !important;
    border: 1px solid #ffffff1e !important;
    border-radius: 18px !important;
    backdrop-filter: blur(10px);
    box-shadow: inset 0 1px 0 #ffffff14, 0 8px 20px #00000038;
}
.aida-chat .message-bubble-border {
    border-radius: 8px !important;
}
.aida-chat [data-testid="user"] .bubble-wrap {
    background: #c06c6c1f !important;
    border: 1px solid #c06c6c44 !important;
    color: #f5dbdb !important;
    font-size: 0.82rem !important;
}
.aida-chat [data-testid="bot"] .bubble-wrap {
    background: #131a17 !important;
    border: 1px solid #ffffff14 !important;
    color: #d5dbd8 !important;
    font-size: 0.82rem !important;
}

/* ── Mode radio ─────────────────────────────────────────── */
.aida-mode .wrap {
    gap: 6px !important;
    justify-content: center;
}
.aida-mode label span {
    font-size: 0.7rem !important;
    letter-spacing: 1px;
    color: #8892a4 !important;
}
.aida-mode label.selected span {
    color: #c87a7a !important;
}
.aida-mode label {
    border: 1px solid #c87a7a26 !important;
    border-radius: 8px !important;
    background: #0f171400 !important;
    padding: 3px 12px !important;
}
.aida-mode label.selected {
    border-color: #c87a7a66 !important;
    background: #c87a7a1a !important;
}

/* ── Input row ──────────────────────────────────────────── */
.aida-input textarea {
    background: #181315 !important;
    border: 1px solid #c87a7a26 !important;
    border-radius: 10px !important;
    color: #e4d4d4 !important;
    font-size: 0.82rem !important;
    resize: none !important;
}
.aida-input textarea:focus {
    border-color: #c87a7a55 !important;
    box-shadow: 0 0 0 2px #c87a7a1f !important;
}
.aida-send button {
    background: #2a1f22 !important;
    border: 1px solid #c87a7a44 !important;
    border-radius: 10px !important;
    color: #f1c9c9 !important;
    font-size: 0.75rem !important;
    letter-spacing: 1px;
    height: 42px !important;
}
.aida-send button:hover {
    background: #3a2a2e !important;
}

/* ── Voice panel ────────────────────────────────────────── */
.aida-voice-panel {
    border: 1px solid #c87a7a25 !important;
    border-radius: 10px !important;
    padding: 6px 8px !important;
    background: #151012 !important;
}
.aida-voice-panel select, .aida-voice-panel .wrap-inner,
.aida-settings input, .aida-settings select {
    background: #1b1416 !important;
    border: 1px solid #c87a7a26 !important;
    color: #d9b7b7 !important;
    font-size: 0.72rem !important;
    border-radius: 8px !important;
}

/* ── Wake word button ───────────────────────────────────── */
.wake-btn button {
    width: 100% !important;
    background: #1a1020 !important;
    border: 1px solid #9b72dd66 !important;
    border-radius: 8px !important;
    color: #c4a8f5 !important;
    font-size: 0.72rem !important;
    letter-spacing: 1px;
    padding: 5px 0 !important;
}
.wake-btn button:hover {
    background: #231530 !important;
}

.aida-settings {
    border: 1px solid #c87a7a25 !important;
    border-radius: 10px !important;
    background: #151012 !important;
    box-shadow: inset 0 0 0 1px #c87a7a1a;
}
.aida-settings button {
    border-radius: 8px !important;
}
.aida-settings-status textarea {
    color: #c87a7a !important;
}
.aida-voice-panel button {
    background: #2a1f22 !important;
    border: 1px solid #c87a7a44 !important;
    color: #f1c9c9 !important;
}
.aida-voice-panel button:hover {
    background: #3a2a2e !important;
}

/* ── Clear button ───────────────────────────────────────── */
.aida-clear button {
    width: 100% !important;
    background: transparent !important;
    border: 1px solid #ffffff0a !important;
    border-radius: 6px !important;
    color: #ffffff28 !important;
    font-size: 0.68rem !important;
    letter-spacing: 1px;
    padding: 4px 0 !important;
}
.aida-clear button:hover {
    border-color: #ff444428 !important;
    color: #ff444488 !important;
}

/* Hide all Gradio labels we don't need */
.aida-status label,
.aida-voice-panel > .wrap > label:first-child,
.hide-label label { display: none !important; }
"""


# ── UI layout ─────────────────────────────────────────────────────────────────

log.info("Querying microphones...")
MIC_LIST   = get_microphones()
log.info("Microphones: %s", MIC_LIST)
CHAT_HEIGHT = 150

PORT = 7860

log.info("Building Gradio UI (gr.Blocks)...")
with gr.Blocks(title="AIDA", fill_width=True) as demo:

    # ── Header ──────────────────────────────────────────────────────────────
    gr.HTML("""
    <div class="aida-header">◈ AIDA</div>
    <div class="aida-sub">MINIMAL VOICE ASSISTANT</div>
    """)

    # ── Status ───────────────────────────────────────────────────────────────
    status_md = gr.Markdown(get_status(), elem_classes=["aida-status", "hide-label"])

    # ── Mode ─────────────────────────────────────────────────────────────────
    mode = gr.Radio(
        choices=["Text", "Voice"],
        value="Voice",
        show_label=False,
        container=False,
        elem_classes=["aida-mode"],
    )

    core_view = gr.HTML(core_markup(False))

    # ── Settings ─────────────────────────────────────────────────────────────
    with gr.Accordion("⚙ Settings", open=False, elem_classes=["aida-settings"]):
        gemini_key = gr.Textbox(
            label="Gemini API Key",
            type="password",
            value=os.getenv("GEMINI_API_KEY", ""),
            placeholder="AIza...",
        )
        openai_key = gr.Textbox(
            label="OpenAI API Key",
            type="password",
            value=os.getenv("OPENAI_API_KEY", ""),
            placeholder="sk-...",
        )
        save_settings_btn = gr.Button("Save settings", variant="secondary", size="sm")
        settings_status = gr.Textbox(
            label="", value="", interactive=False, lines=1,
            elem_classes=["aida-settings-status"],
        )

    # ── Chat ─────────────────────────────────────────────────────────────────
    chatbox = gr.Chatbot(
        label="",
        height=CHAT_HEIGHT,
        min_height=CHAT_HEIGHT,
        show_label=False,
        elem_classes=["aida-chat"],
        avatar_images=(None, None),
    )

    # ── Input ────────────────────────────────────────────────────────────────
    with gr.Row(equal_height=True) as text_row:
        text_in = gr.Textbox(
            placeholder="Message AIDA...",
            show_label=False,
            scale=8,
            container=False,
            elem_classes=["aida-input"],
            lines=1,
            max_lines=1,
        )
        send_btn = gr.Button("▶", scale=1, variant="secondary", size="sm",
                             elem_classes=["aida-send"])

    # ── Voice panel ───────────────────────────────────────────────────────────
    with gr.Group(visible=True, elem_classes=["aida-voice-panel"]) as voice_group:
        gr.Markdown("**Voice mode** — press TALK (VAD: auto-stops on silence).")
        with gr.Row(equal_height=True):
            talk_btn   = gr.Button("🎤 TALK", variant="secondary", size="sm")
            mic_muted  = gr.Checkbox(label="Mute mic", value=False)
        voice_state = gr.Textbox(label="Voice status", value="Idle",
                                 interactive=False, lines=1)

        gr.Markdown("**Wake Word mode** — say *'Aida'* to trigger automatically.")
        wake_btn        = gr.Button("🔊 Wake Word ON", variant="secondary", size="sm",
                                    elem_classes=["wake-btn"])
        wake_status_txt = gr.Textbox(label="", value="Inactive",
                                     interactive=False, lines=1,
                                     elem_classes=["hide-label"])

    # ── Clear ────────────────────────────────────────────────────────────────
    clear_btn = gr.Button("CLEAR CONVERSATION", variant="secondary", size="sm",
                          elem_classes=["aida-clear"])

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
        inputs=[chatbox, mode, mic_muted],
        outputs=[chatbox, voice_state, core_view],
    )

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

        log.info("Opening native window 430×900 at http://127.0.0.1:%d", PORT)
        try:
            webview.create_window(
                title="AIDA",
                url=f"http://127.0.0.1:{PORT}",
                width=430,
                height=900,
                resizable=False,
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
                css=CSS,
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
