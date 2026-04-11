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
    return gr.update(visible=show), gr.update(visible=show)


# ─── Custom CSS ── 420×870 window, no page-level scroll ──────────────────────
# All sizing is tuned so every element fits inside 870 px total height.

CSS = """
/* ── Reset / root ──────────────────────────────────────── */
html, body {
    overflow: hidden !important;
    height: 100% !important;
    margin: 0;
    padding: 0;
    background: #0a0c0f !important;
    font-family: 'Inter', 'Segoe UI', sans-serif;
}

.gradio-container {
    max-width: 420px !important;
    width: 420px !important;
    min-height: 870px !important;
    max-height: 870px !important;
    overflow: hidden !important;
    background: #0a0c0f !important;
    margin: 0 !important;
    padding: 0 6px 0 6px !important;
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
    font-size: 1.35rem;
    font-weight: 800;
    color: #00d4ff;
    text-shadow: 0 0 18px #00d4ff88;
}
.aida-sub {
    text-align: center;
    font-size: 0.55rem;
    letter-spacing: 4px;
    color: #ffffff28;
    margin-bottom: 6px;
}

/* ── Status bar ─────────────────────────────────────────── */
.aida-status p {
    text-align: center;
    font-size: 0.65rem;
    color: #00d4ff88;
    letter-spacing: 1px;
    margin: 0 !important;
    padding: 2px 0 4px !important;
}

/* ── Chatbot ─────────────────────────────────────────────── */
.aida-chat {
    background: transparent !important;
    border: 1px solid #ffffff0f !important;
    border-radius: 10px !important;
}
.aida-chat .message-bubble-border {
    border-radius: 8px !important;
}
/* user bubble */
.aida-chat [data-testid="user"] .bubble-wrap {
    background: #00d4ff18 !important;
    border: 1px solid #00d4ff30 !important;
    color: #d0f4ff !important;
    font-size: 0.82rem !important;
}
/* assistant bubble */
.aida-chat [data-testid="bot"] .bubble-wrap {
    background: #1a1e26 !important;
    border: 1px solid #ffffff12 !important;
    color: #c8cdd8 !important;
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
    color: #00d4ff !important;
}
.aida-mode label {
    border: 1px solid #ffffff14 !important;
    border-radius: 6px !important;
    background: transparent !important;
    padding: 3px 12px !important;
}
.aida-mode label.selected {
    border-color: #00d4ff44 !important;
    background: #00d4ff0e !important;
}

/* ── Input row ──────────────────────────────────────────── */
.aida-input textarea {
    background: #111318 !important;
    border: 1px solid #ffffff14 !important;
    border-radius: 8px !important;
    color: #d0d8e8 !important;
    font-size: 0.82rem !important;
    resize: none !important;
}
.aida-input textarea:focus {
    border-color: #00d4ff44 !important;
    box-shadow: 0 0 0 2px #00d4ff14 !important;
}
.aida-send button {
    background: #00d4ff18 !important;
    border: 1px solid #00d4ff44 !important;
    border-radius: 8px !important;
    color: #00d4ff !important;
    font-size: 0.75rem !important;
    letter-spacing: 1px;
    height: 42px !important;
}
.aida-send button:hover {
    background: #00d4ff28 !important;
}

/* ── Voice panel ────────────────────────────────────────── */
.aida-voice-panel {
    border: 1px solid #ffffff0a !important;
    border-radius: 8px !important;
    padding: 6px 8px !important;
    background: #0e1117 !important;
}
.aida-voice-panel select, .aida-voice-panel .wrap-inner {
    background: #111318 !important;
    border: 1px solid #ffffff14 !important;
    color: #8892a4 !important;
    font-size: 0.72rem !important;
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
#  = 340px used → chat gets 870 - 340 = 530px
CHAT_HEIGHT = 530

with gr.Blocks(
    title="AIDA",
    fill_height=False,
    fill_width=True,
    # NOTE: In Gradio 6.x css is passed to launch(), not Blocks()
) as demo:

    # ── Header ──────────────────────────────────────────────────────────────
    gr.HTML("""
    <div class="aida-header">◈ AIDA</div>
    <div class="aida-sub">AI DESKTOP ASSISTANT</div>
    """)

    # ── Status ───────────────────────────────────────────────────────────────
    gr.Markdown(get_status, elem_classes=["aida-status", "hide-label"])

    # ── Mode ─────────────────────────────────────────────────────────────────
    mode = gr.Radio(
        choices=["Text", "Voice", "Hybrid"],
        value="Hybrid",
        show_label=False,
        container=False,
        elem_classes=["aida-mode"],
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
    with gr.Row(equal_height=True):
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
        mic_choice = gr.Dropdown(
            choices=MIC_LIST,
            value=MIC_LIST[0] if MIC_LIST else None,
            show_label=False,
            container=False,
            interactive=True,
        )
        mic_in = gr.Audio(
            sources=["microphone"],
            type="filepath",
            show_label=False,
            container=False,
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

    mic_in.stop_recording(
        fn=handle_voice,
        inputs=[mic_in, chatbox, mode, mic_choice],
        outputs=[chatbox, audio_out],
    )

    clear_btn.click(fn=clear_chat, outputs=[chatbox, audio_out])

    mode.change(
        fn=toggle_voice_ui,
        inputs=mode,
        outputs=[voice_group, audio_out],
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

        log.info("Opening native window (420×870)...")
        webview.create_window(
            title="AIDA",
            url=f"http://127.0.0.1:{PORT}",
            width=420,
            height=870,
            resizable=False,        # fixed size — layout is tuned for this
            min_size=(420, 870),
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
