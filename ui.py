"""
ui.py — AIDA Desktop App
Нативное окно через pywebview + Gradio 6.12
Размер окна: 420×870, без скроллинга, минималистичный дизайн
Запуск: python ui.py
"""

import asyncio
import logging
import os
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("aida.ui")

orchestrator = Orchestrator()


def core_markup(glow: bool = False) -> str:
    active = " core-active" if glow else ""
    return f"""
    <div class=\"jarvis-core{active}\">
      <div class=\"jarvis-core-ring\">
        <div class=\"jarvis-core-dot\"></div>
      </div>
    </div>
    """

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
    if mode != "Voice":
        return history, None, "Voice mode is disabled.", "", core_markup(False)
    if muted:
        history = history + [make_aida_msg("🔇 Microphone muted. Unmute to speak.")]
        return history, None, "Muted. Unmute microphone to record.", "", core_markup(False)

    audio_path = None
    try:
        audio_path = record_from_mic(mic_choice, duration=max(2.0, float(duration)))
        transcript = transcribe_audio(audio_path, mic_choice)
        history = history + [make_user_msg(f"🎤 {transcript}")]
        if transcript.startswith("["):
            return history, None, "Speech-to-text failed.", transcript, core_markup(False)
        response = _run_async(orchestrator.process(transcript))
        history = history + [make_aida_msg(response)]
        return history, speak_text(response), "✅ Voice captured and processed.", transcript, core_markup(True)
    except Exception as e:
        history = history + [make_aida_msg(f"[Voice capture error: {e}]")]
        return history, None, f"Voice capture error: {e}", "", core_markup(False)
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
        yield history, "", None, core_markup(False)
        return

    history = history + [make_user_msg(user_message)]
    yield history, "", None, core_markup(False)

    response  = _run_async(orchestrator.process(user_message))
    history   = history + [make_aida_msg(response)]
    audio_out = speak_text(response) if mode == "Voice" else None
    yield history, "", audio_out, core_markup(True)


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
    show = m == "Voice"
    text_enabled = m != "Voice"
    chat_h = 150 if show else 320
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


# ─── Custom CSS ── 420×870 window, no page-level scroll ──────────────────────
# All sizing is tuned so every element fits inside 870 px total height.

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
/* user bubble */
.aida-chat [data-testid="user"] .bubble-wrap {
    background: #c06c6c1f !important;
    border: 1px solid #c06c6c44 !important;
    color: #f5dbdb !important;
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
        gr.Markdown(
            "Voice mode is **output-only**: AIDA replies with voice (TTS). "
            "Speech capture/wake-word is disabled for stability."
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
        outputs=[chatbox, text_in, audio_out, core_view],
    )
    text_in.submit(**submit_cfg)
    send_btn.click(**submit_cfg)

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
