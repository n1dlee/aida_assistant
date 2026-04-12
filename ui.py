"""
ui.py — AIDA Desktop App
Нативное окно через pywebview + Gradio 6.12
Размер окна: 420×870, без скроллинга, минималистичный дизайн
Запуск: python ui.py
"""

import asyncio
import logging
import os
import queue
import sys
import tempfile
import threading
from pathlib import Path

# ── ell: must be initialised BEFORE it is used ──────────────────────────────
# Assign None first so Pylance never sees it as "possibly unbound"
ell = None  # type: ignore[assignment]
ELL_AVAILABLE = False
try:
    import ell as _ell_module  # noqa: F401
    ell = _ell_module
    ELL_AVAILABLE = True
except ImportError:
    pass

import gradio as gr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.orchestrator import Orchestrator
from voice.wake_word import WakeWordDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("aida.ui")

orchestrator = Orchestrator()
WAKE_EVENTS: "queue.Queue[tuple]" = queue.Queue()
WAKE_RUNNING = False
WAKE_THREAD: threading.Thread | None = None

if ELL_AVAILABLE and ell is not None:
    ell.init(store="./data/ell_store", autocommit=True, verbose=False)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _run_async(coro):
    """Run coroutine from a sync context, whether a loop is running or not."""
    try:
        # Python 3.10+: get_running_loop() (non-deprecated)
        loop = asyncio.get_running_loop()
        return asyncio.run_coroutine_threadsafe(coro, loop).result(timeout=60)
    except RuntimeError:
        # No running loop — safe to use asyncio.run()
        return asyncio.run(coro)


def get_microphones() -> list[str]:
    """Returns a list of available input devices."""
    try:
        import sounddevice as sd
        num_devices = len(sd.query_devices())
        mics: list[str] = []
        for i in range(num_devices):
            # sd.query_devices(i) returns a proper dict — avoids Pylance
            # subscript error that occurs when iterating DeviceList (a tuple).
            d: dict = sd.query_devices(i)  # type: ignore[assignment]
            if d["max_input_channels"] > 0:
                mics.append(f"{i}: {d['name']}")
        return mics if mics else ["Default microphone"]
    except Exception:
        return ["Default microphone"]


def transcribe_audio(audio_path: str, mic_index: str) -> str:
    try:
        from faster_whisper import WhisperModel
        m = WhisperModel(
            os.getenv("WHISPER_MODEL", "small"), device="cpu", compute_type="int8"
        )
        segs, _ = m.transcribe(audio_path, beam_size=5)
        return " ".join(s.text for s in segs).strip()
    except ImportError:
        return "[faster-whisper not installed — pip install faster-whisper]"
    except Exception as e:
        return f"[STT error: {e}]"


def record_from_mic(mic_choice: str, duration: float = 5.0) -> str:
    """Record from selected input device and return temporary WAV path."""
    import sounddevice as sd
    import soundfile as sf

    sample_rate = 16000
    device_idx = None
    if mic_choice and ":" in mic_choice:
        try:
            device_idx = int(mic_choice.split(":", 1)[0].strip())
        except ValueError:
            device_idx = None

    log.info("Recording %.1fs from mic: %s", duration, mic_choice or "default")
    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        device=device_idx,
    )
    sd.wait()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fh:
        tmp = fh.name
    sf.write(tmp, audio, sample_rate)
    return tmp


def handle_voice_capture(
    history: list, mode: str, mic_choice: str, muted: bool, duration: float
):
    if mode not in ("Voice", "Hybrid"):
        return history, None, "Voice mode is disabled.", ""
    if muted:
        history = history + [make_aida_msg("🔇 Microphone muted. Unmute to speak.")]
        return history, None, "Muted. Unmute microphone to record.", ""

    audio_path = None
    try:
        audio_path = record_from_mic(mic_choice, duration=max(2.0, float(duration)))
        transcript = transcribe_audio(audio_path, mic_choice)
        history = history + [make_user_msg(f"🎤 {transcript}")]
        if transcript.startswith("["):
            return history, None, "Speech-to-text failed.", transcript
        response = _run_async(orchestrator.process(transcript))
        history = history + [make_aida_msg(response)]
        return history, speak_text(response), "✅ Voice captured and processed.", transcript
    except Exception as e:
        history = history + [make_aida_msg(f"[Voice capture error: {e}]")]
        return history, None, f"Voice capture error: {e}", ""
    finally:
        if audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except OSError:
                pass


def speak_text(text: str) -> str | None:
    """TTS → WAV file path, or None if TTS unavailable."""
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", int(os.getenv("AIDA_TTS_RATE", "170")))
        # mktemp() is deprecated since Python 2.3 — use NamedTemporaryFile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fh:
            tmp = fh.name
        engine.save_to_file(text, tmp)
        engine.runAndWait()
        return tmp if os.path.exists(tmp) else None
    except Exception as e:
        log.debug("TTS skipped: %s", e)
        return None


def get_status() -> str:
    local_ok  = orchestrator.selector.local.is_available()
    cloud_ok  = orchestrator.selector.cloud.is_available()
    mem       = orchestrator.vector_store.count()
    parts: list[str] = []
    if local_ok:
        parts.append(f"⬡ {orchestrator.selector.local.model_name}")
    if cloud_ok:
        parts.append(f"☁ {orchestrator.selector.cloud.model_name}")
    parts.append(f"🧠 {mem}")
    return "  ·  ".join(parts) if parts else "⚠ No model"


# ─── Message helpers (Gradio 6 requires role/content dicts) ──────────────────

def make_user_msg(text: str) -> dict:
    return {"role": "user", "content": text}

def make_aida_msg(text: str) -> dict:
    return {"role": "assistant", "content": text}


# ─── Event handlers ───────────────────────────────────────────────────────────

def handle_text(user_message: str, history: list, mode: str):
    if not user_message.strip():
        yield history, "", None
        return

    history = history + [make_user_msg(user_message)]
    yield history, "", None

    response  = _run_async(orchestrator.process(user_message))
    history   = history + [make_aida_msg(response)]
    audio_out = speak_text(response) if mode in ("Voice", "Hybrid") else None
    yield history, "", audio_out


def handle_voice(audio_path, history: list, mode: str, mic_choice: str):
    if audio_path is None:
        return history, None
    transcript = transcribe_audio(audio_path, mic_choice)
    history    = history + [make_user_msg(f"🎤 {transcript}")]
    if transcript.startswith("["):
        return history, None
    response = _run_async(orchestrator.process(transcript))
    history  = history + [make_aida_msg(response)]
    return history, speak_text(response)


def clear_chat():
    orchestrator.buffer.clear()
    return [], None


def toggle_voice_ui(m: str):
    show = m in ("Voice", "Hybrid")
    text_enabled = m != "Voice"
    chat_h = 260 if show else 320
    return (
        gr.update(visible=show),
        gr.update(visible=show),
        gr.update(visible=text_enabled),
        gr.update(height=chat_h, min_height=chat_h),
    )


def _set_env_value(env_path: Path, key: str, value: str):
    env_path.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if env_path.exists():
        existing = env_path.read_text(encoding="utf-8").splitlines()

    prefix = f"{key}="
    replaced = False
    out = []
    for line in existing:
        if line.startswith(prefix):
            out.append(f"{prefix}{value}")
            replaced = True
        else:
            out.append(line)
    if not replaced:
        out.append(f"{prefix}{value}")

    env_path.write_text("\n".join(out).strip() + "\n", encoding="utf-8")


def save_settings(gemini_key: str, openai_key: str, mic_choice: str):
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

    return (
        get_status(),
        f"✅ Settings saved. Mic: {mic_choice or 'default'}",
    )


def _wake_loop(mic_choice: str, duration: float, wake_word: str):
    global WAKE_RUNNING
    detector = WakeWordDetector(wake_word=wake_word)
    WAKE_EVENTS.put(("status", f"Listening 24/7 for wake word: '{wake_word}'"))

    if not detector.is_available():
        WAKE_EVENTS.put(("status", "Wake word module unavailable (install openwakeword)."))
        WAKE_RUNNING = False
        return

    while WAKE_RUNNING:
        try:
            triggered = detector._blocking_listen()
            if not WAKE_RUNNING:
                break
            if not triggered:
                continue

            WAKE_EVENTS.put(("status", "Wake word detected. Capturing voice..."))
            audio_path = record_from_mic(mic_choice, duration=max(2.0, float(duration)))
            transcript = transcribe_audio(audio_path, mic_choice)
            try:
                os.remove(audio_path)
            except OSError:
                pass

            if transcript.startswith("["):
                WAKE_EVENTS.put(("status", f"STT failed: {transcript}"))
                continue

            response = asyncio.run(orchestrator.process(transcript))
            WAKE_EVENTS.put(("message", transcript, response))
            WAKE_EVENTS.put(("status", "Listening for wake word..."))
        except Exception as e:
            WAKE_EVENTS.put(("status", f"Wake loop error: {e}"))


def start_wake_mode(mic_choice: str, duration: float, wake_word: str):
    global WAKE_RUNNING, WAKE_THREAD
    if WAKE_RUNNING:
        return "Wake mode already running."
    WAKE_RUNNING = True
    WAKE_THREAD = threading.Thread(
        target=_wake_loop,
        args=(mic_choice, duration, (wake_word or "aida").strip().lower()),
        daemon=True,
    )
    WAKE_THREAD.start()
    return "Wake mode started."


def stop_wake_mode():
    global WAKE_RUNNING
    WAKE_RUNNING = False
    return "Wake mode stopped."


def poll_wake_events(history: list):
    status = "Listening..." if WAKE_RUNNING else "Idle"
    transcript = ""
    while not WAKE_EVENTS.empty():
        event = WAKE_EVENTS.get()
        if not event:
            continue
        if event[0] == "status":
            status = event[1]
        elif event[0] == "message":
            user_text, bot_text = event[1], event[2]
            history = history + [make_user_msg(f"🎤 {user_text}"), make_aida_msg(bot_text)]
            transcript = user_text
    return history, status, transcript


# ─── Custom CSS ── 420×870 window, no page-level scroll ──────────────────────
# All sizing is tuned so every element fits inside 870 px total height.

CSS = """
/* ── Reset / root ──────────────────────────────────────── */
html, body {
    overflow: hidden !important;
    height: 100% !important;
    margin: 0;
    padding: 0;
    background: #0b1210 !important;
    font-family: 'Inter', 'Segoe UI', sans-serif;
}

.gradio-container {
    max-width: 430px !important;
    width: 430px !important;
    min-height: 900px !important;
    max-height: 900px !important;
    overflow: hidden !important;
    background: linear-gradient(180deg, #0f1412 0%, #0a0f0d 100%) !important;
    margin: 0 !important;
    padding: 0 10px !important;
    box-sizing: border-box;
}

/* Hide Gradio footer */
footer { display: none !important; }
.built-with { display: none !important; }

/* ── Header ─────────────────────────────────────────────── */
.aida-header {
    text-align: center;
    padding: 14px 0 4px;
    letter-spacing: 6px;
    font-size: 1.45rem;
    font-weight: 800;
    color: #9bd9b2;
    text-shadow: 0 0 18px #9bd9b255;
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
    border: 1px solid #9bd9b24a;
    background: radial-gradient(circle, #9bd9b235 0%, #0f171400 72%);
    box-shadow: inset 0 0 24px #9bd9b22b, 0 0 26px #00000055;
    position: relative;
}
.jarvis-core-ring::before {
    content: "";
    position: absolute;
    inset: 18px;
    border-radius: 999px;
    border: 1px dashed #9bd9b23d;
}
.jarvis-core-dot {
    position: absolute;
    inset: 58px;
    border-radius: 999px;
    background: radial-gradient(circle, #e8fffa 0%, #96ddd3 52%, #96ddd300 100%);
}
/* ── Status bar ─────────────────────────────────────────── */
.aida-status p {
    text-align: center;
    font-size: 0.65rem;
    color: #9bd9b275;
    letter-spacing: 1px;
    margin: 0 !important;
    padding: 2px 0 4px !important;
}

/* ── Chatbot ─────────────────────────────────────────────── */
.aida-chat {
    background: #141917cc !important;
    border: 1px solid #ffffff1e !important;
    border-radius: 18px !important;
    backdrop-filter: blur(10px);
    box-shadow: inset 0 1px 0 #ffffff14, 0 8px 20px #00000038;
}
.aida-chat .message-bubble-border {
    border-radius: 8px !important;
}
/* user bubble */
.aida-chat [data-testid="user"] .bubble-wrap {
    background: #9bd9b21a !important;
    border: 1px solid #9bd9b244 !important;
    color: #d7efe2 !important;
    font-size: 0.82rem !important;
}
/* assistant bubble */
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
    color: #9bd9b2 !important;
}
.aida-mode label {
    border: 1px solid #9bd9b226 !important;
    border-radius: 8px !important;
    background: #0f171400 !important;
    padding: 3px 12px !important;
}
.aida-mode label.selected {
    border-color: #9bd9b255 !important;
    background: #9bd9b214 !important;
}

/* ── Input row ──────────────────────────────────────────── */
.aida-input textarea {
    background: #111a16 !important;
    border: 1px solid #9bd9b226 !important;
    border-radius: 10px !important;
    color: #d6e2dc !important;
    font-size: 0.82rem !important;
    resize: none !important;
}
.aida-input textarea:focus {
    border-color: #9bd9b255 !important;
    box-shadow: 0 0 0 2px #9bd9b218 !important;
}
.aida-send button {
    background: #1a2d24 !important;
    border: 1px solid #9bd9b244 !important;
    border-radius: 10px !important;
    color: #b9e3ca !important;
    font-size: 0.75rem !important;
    letter-spacing: 1px;
    height: 42px !important;
}
.aida-send button:hover {
    background: #20372c !important;
}

/* ── Voice panel ────────────────────────────────────────── */
.aida-voice-panel {
    border: 1px solid #9bd9b220 !important;
    border-radius: 10px !important;
    padding: 6px 8px !important;
    background: #0f1714 !important;
}
.aida-voice-panel select, .aida-voice-panel .wrap-inner,
.aida-settings input, .aida-settings select {
    background: #121d18 !important;
    border: 1px solid #9bd9b226 !important;
    color: #a7b9b0 !important;
    font-size: 0.72rem !important;
    border-radius: 8px !important;
}

.aida-settings {
    border: 1px solid #9bd9b220 !important;
    border-radius: 10px !important;
    background: #0f1714 !important;
    box-shadow: inset 0 0 0 1px #9bd9b21a;
}
.aida-settings button {
    border-radius: 8px !important;
}
.aida-settings-status textarea {
    color: #9bd9b2 !important;
}
.aida-voice-panel button {
    background: #1d3027 !important;
    border: 1px solid #9bd9b244 !important;
    color: #ccead8 !important;
}
.aida-voice-panel button:hover {
    background: #264136 !important;
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


# ─── UI layout (fits exactly in 420×870, zero page scroll) ───────────────────

MIC_LIST = get_microphones()

# Chat height = 870px total
#  - header+sub:   ~62px
#  - status:       ~24px
#  - mode radio:   ~38px
#  - input row:    ~50px
#  - voice panel:  ~100px (visible by default for Hybrid)
#  - clear btn:    ~30px
#  - gaps/padding: ~36px
#  = dynamic, so controls at bottom stay visible even in Voice/Hybrid.
CHAT_HEIGHT = 260

with gr.Blocks(
    title="AIDA",
    fill_height=False,
    fill_width=True,
    # NOTE: In Gradio 6.x css is passed to launch(), not Blocks()
) as demo:

    # ── Header ──────────────────────────────────────────────────────────────
    gr.HTML("""
    <div class="aida-header">◈ AIDA</div>
    <div class="aida-sub">MINIMAL VOICE ASSISTANT</div>
    """)

    # ── Status ───────────────────────────────────────────────────────────────
    status_md = gr.Markdown(get_status(), elem_classes=["aida-status", "hide-label"])

    # ── Mode ─────────────────────────────────────────────────────────────────
    mode = gr.Radio(
        choices=["Text", "Voice", "Hybrid"],
        value="Hybrid",
        show_label=False,
        container=False,
        elem_classes=["aida-mode"],
    )

    gr.HTML("""
    <div class="jarvis-core">
      <div class="jarvis-core-ring">
        <div class="jarvis-core-dot"></div>
      </div>
    </div>
    """)

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
        mic_choice = gr.Dropdown(
            choices=MIC_LIST,
            value=MIC_LIST[0] if MIC_LIST else None,
            label="Microphone",
            interactive=True,
        )
        save_settings_btn = gr.Button("Save settings", variant="secondary", size="sm")
        settings_status = gr.Textbox(
            label="",
            value="",
            interactive=False,
            lines=1,
            elem_classes=["aida-settings-status"],
        )

    # ── Chat ─────────────────────────────────────────────────────────────────
    chatbox = gr.Chatbot(
        label="",
        height=CHAT_HEIGHT,
        min_height=CHAT_HEIGHT,
        layout="bubble",
        show_label=False,
        elem_classes=["aida-chat"],
        avatar_images=(None, None),   # no avatars — keeps layout tight
        autoscroll=True,
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
        send_btn = gr.Button(
            "▶",
            scale=1,
            variant="secondary",
            size="sm",
            elem_classes=["aida-send"],
        )

    # ── Voice panel (hidden in Text mode) ────────────────────────────────────
    with gr.Group(visible=True, elem_classes=["aida-voice-panel"]) as voice_group:
        voice_hint = gr.Markdown(
            "Voice mode: choose mic → set duration → press **TALK** and speak immediately."
        )
        wake_word = gr.Textbox(
            label="Wake word",
            value="aida",
            placeholder="aida",
            lines=1,
        )
        with gr.Row(equal_height=True):
            start_wake_btn = gr.Button("▶ Start 24/7 Wake Mode", variant="secondary", size="sm")
            stop_wake_btn = gr.Button("■ Stop", variant="secondary", size="sm")
        record_seconds = gr.Slider(
            minimum=2,
            maximum=12,
            step=1,
            value=5,
            label="Record duration (seconds)",
        )
        with gr.Row(equal_height=True):
            talk_btn = gr.Button("🎙 START VOICE CAPTURE", variant="secondary", size="sm")
            mic_muted = gr.Checkbox(label="Mute mic", value=False)
        voice_state = gr.Textbox(
            label="Voice status",
            value="Idle",
            interactive=False,
            lines=1,
        )
        last_transcript = gr.Textbox(
            label="Last transcript",
            value="",
            interactive=False,
            lines=2,
        )

    # ── Audio output (hidden element — autoplay only) ─────────────────────────
    audio_out = gr.Audio(
        autoplay=True,
        interactive=False,
        visible=False,   # hidden — we only need the autoplay, not UI chrome
    )

    # ── Clear ────────────────────────────────────────────────────────────────
    clear_btn = gr.Button(
        "CLEAR CONVERSATION",
        variant="secondary",
        size="sm",
        elem_classes=["aida-clear"],
    )

    # ── Wiring ───────────────────────────────────────────────────────────────

    submit_cfg = dict(
        fn=handle_text,
        inputs=[text_in, chatbox, mode],
        outputs=[chatbox, text_in, audio_out],
    )
    text_in.submit(**submit_cfg)
    send_btn.click(**submit_cfg)

    talk_btn.click(
        fn=handle_voice_capture,
        inputs=[chatbox, mode, mic_choice, mic_muted, record_seconds],
        outputs=[chatbox, audio_out, voice_state, last_transcript],
    )
    start_wake_btn.click(
        fn=start_wake_mode,
        inputs=[mic_choice, record_seconds, wake_word],
        outputs=[voice_state],
    )
    stop_wake_btn.click(
        fn=stop_wake_mode,
        outputs=[voice_state],
    )

    clear_btn.click(fn=clear_chat, outputs=[chatbox, audio_out])

    mode.change(
        fn=toggle_voice_ui,
        inputs=mode,
        outputs=[voice_group, audio_out, text_row, chatbox],
    )

    save_settings_btn.click(
        fn=save_settings,
        inputs=[gemini_key, openai_key, mic_choice],
        outputs=[status_md, settings_status],
    )

    wake_timer = gr.Timer(value=1.0, active=True)
    wake_timer.tick(
        fn=poll_wake_events,
        inputs=[chatbox],
        outputs=[chatbox, voice_state, last_transcript],
    )


# ─── Launch ── native window (pywebview) or browser fallback ─────────────────

PORT = 7860


def _start_gradio():
    demo.queue().launch(
        server_name="127.0.0.1",
        server_port=PORT,
        prevent_thread_lock=True,
        show_error=True,
        quiet=True,
        inbrowser=False,
        css=CSS,               # Gradio 6.x: css belongs in launch(), not Blocks()
    )


if __name__ == "__main__":
    log.info("Starting AIDA...")

    try:
        import webview  # pywebview ≥ 4.x

        t = threading.Thread(target=_start_gradio, daemon=True)
        t.start()

        import time
        time.sleep(2)   # wait for Gradio to bind

        log.info("Opening native window (430×900)...")
        webview.create_window(
            title="AIDA",
            url=f"http://127.0.0.1:{PORT}",
            width=430,
            height=900,
            resizable=False,        # fixed size — layout is tuned for this
            min_size=(430, 900),
            frameless=False,
            on_top=False,
            background_color="#0a0c0f",
        )
        webview.start(debug=False)

    except ImportError:
        log.warning("pywebview not installed — opening in browser instead.")
        log.warning("Install: pip install pywebview")
        demo.queue().launch(
            server_name="0.0.0.0",
            server_port=PORT,
            inbrowser=True,
            show_error=True,
            quiet=False,
            css=CSS,
        )
