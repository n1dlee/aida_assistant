# AIDA — AI Desktop Assistant

Personal Jarvis-style AI assistant. Hybrid local + cloud LLM, voice I/O, memory, tools, proactive triggers.

---

## Quick Start

### 1. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env — add at least one API key (Gemini or OpenAI)
```

### 3. (Optional) Install Ollama for local LLM
Download from https://ollama.ai, then:
```bash
ollama pull llama3.1:8b
```

### 4. Run AIDA
```bash
python main.py
```

AIDA starts in **text mode** by default (no mic required).  
Type your messages, press Enter. Type `quit` to exit.

---

## Project Structure

```
aida/
├── main.py                  # Entry point
├── core/
│   ├── orchestrator.py      # Central coordinator
│   ├── router.py            # Intent classifier
│   └── personality.py       # System prompt builder
├── brain/
│   ├── selector.py          # Local vs cloud routing logic
│   ├── local_llm.py         # Ollama wrapper
│   └── cloud_llm.py         # Gemini / OpenAI wrapper
├── memory/
│   ├── buffer.py            # Short-term conversation buffer
│   ├── vector_store.py      # Long-term semantic memory (ChromaDB)
│   └── episodic.py          # Timestamped event log
├── tools/
│   ├── base_tool.py         # Abstract base class for tools
│   ├── registry.py          # Auto-discovers and dispatches tools
│   ├── system_tool.py       # Open apps / OS commands
│   ├── web_tool.py          # DuckDuckGo web search
│   ├── calendar_tool.py     # Local calendar (JSON)
│   └── time_tool.py         # Current date/time
├── voice/
│   ├── listener.py          # Microphone → Whisper STT
│   ├── wake_word.py         # "Hey AIDA" detection
│   └── speaker.py           # TTS output (pyttsx3 / Piper)
├── proactive/
│   ├── scheduler.py         # APScheduler wrapper
│   └── trigger_engine.py    # Condition-based proactive messages
├── config/
│   ├── settings.yaml        # All configuration (no code changes needed)
│   └── prompts.yaml         # AIDA personality & system prompt
└── data/
    ├── chroma_db/           # Persistent vector memory (auto-created)
    └── episodes.json        # Episodic event log (auto-created)
```

---

## Brain Routing Logic

```
User input
    │
    ▼
complexity_score()
    │
    ├── score < 0.4  ──▶  Ollama (local) ──▶ response
    │
    └── score ≥ 0.4  ──▶  Cloud API
                              │
                        fail? └──▶  Ollama fallback
```

Adjust `AIDA_CLOUD_THRESHOLD` in `.env` to tune this (0.0 = always local).

---

## Adding a New Tool

1. Create `tools/my_tool.py`:
```python
from tools.base_tool import BaseTool

class MyTool(BaseTool):
    @property
    def name(self): return "my_tool"

    @property
    def description(self): return "Does something useful."

    async def run(self, query: str, **kwargs) -> str:
        return f"Result for: {query}"
```

2. Register it in `tools/registry.py`:
```python
from tools.my_tool import MyTool
# In _register_defaults():
self.register(MyTool())
```

That's it.

---

## Enabling Voice Mode

Install voice dependencies:
```bash
pip install faster-whisper sounddevice soundfile pyttsx3 openwakeword
```

Voice mode activates automatically when the packages are present.  
Say **"Hey AIDA"** to activate, then speak your command.

---

## Customising Personality

Edit `config/prompts.yaml` — change `system_prompt` to whatever character you want.  
No Python changes needed.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | — | Gemini API key |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `AIDA_LOCAL_MODEL` | `llama3.1:8b` | Ollama model name |
| `AIDA_CLOUD_THRESHOLD` | `0.4` | Complexity score to trigger cloud |
| `WHISPER_MODEL` | `small` | Whisper model size |
| `AIDA_WAKE_WORD` | `hey aida` | Wake word phrase |

---

## Web Interface (React)

### Start the backend server
```bash
pip install fastapi uvicorn
python server.py
```

### Start the React frontend (dev mode)
```bash
cd frontend
npm install
npm run dev
```
Open **http://localhost:5173** in your browser.

### Build for production (optional)
```bash
cd frontend
npm run build
# Then just run: python server.py
# Open http://localhost:8000
```

### Voice input
Click the microphone button (🎤) and speak.
Uses the browser's built-in Web Speech API — works in Chrome and Edge.
For Russian, change `recog.current.lang = 'ru-RU'` in `frontend/src/hooks/useVoice.js`.

---

## Gradio Voice Interface (рекомендуется)

Самый быстрый способ запустить голосовой интерфейс — только Python, никакого npm.

### Установка
```bash
pip install gradio faster-whisper pyttsx3 ell-ai
```

### Запуск
```bash
python ui.py
```
Открой **http://localhost:7860**

### Три режима
| Режим | Ввод | Вывод |
|---|---|---|
| Text | Текст с клавиатуры | Текст в чате |
| Voice | Микрофон (Whisper STT) | Голос (TTS) + текст |
| Hybrid | Оба | Голос + текст |

### Как пользоваться голосом
1. Выбери режим **Voice** или **Hybrid**
2. Нажми на микрофон → говори → отпусти
3. Whisper транскрибирует → AIDA отвечает → pyttsx3 озвучивает

### ell — версионирование промптов (опционально)
Если установлен `ell-ai`, все промпты автоматически версионируются в `data/ell_store`.
Запусти `ell studio` в отдельном терминале чтобы посмотреть историю:
```bash
ell studio --storage ./data/ell_store
```
