# AIDA Assistant

AIDA is a local-first AI assistant designed to run on your machine with minimal reliance on cloud services.  
It combines voice interaction, local language models, memory, and automation into a single orchestrated system.

---

## 🚀 Overview

AIDA is not just a chatbot. It is an orchestrated assistant that can:

- Understand voice input
- Process requests using local or cloud LLMs
- Store and retrieve memory
- Perform searches
- Respond via text or speech
- Run continuously with background scheduling

All components are coordinated through a central **Orchestrator**.

---

## 🧠 Core Architecture

```
User (Voice / Text)
        ↓
   STT (Whisper)
        ↓
   Orchestrator
   ├── LLM (Ollama / OpenAI / Gemini)
   ├── Memory (ChromaDB)
   ├── Tools (Search, Scheduler)
   └── Actions
        ↓
   TTS (pyttsx3)
        ↓
     Response
```

---

## ✨ Features

- 🧠 Hybrid LLM support  
  - Local: Ollama  
  - Cloud: OpenAI, Google Gemini  

- 🎙 Voice interaction  
  - STT: faster-whisper  
  - TTS: pyttsx3  
  - Wake word support (openwakeword)  

- 🗂 Persistent memory  
  - ChromaDB vector storage  

- 🌐 Built-in tools  
  - DuckDuckGo search  

- ⏱ Task scheduling  
  - APScheduler for background tasks  

- 🖥 Web interface  
  - FastAPI backend  
  - WebSocket communication  
  - Optional React frontend  

- 🎛 Modular architecture  
  - Orchestrator pattern for extensibility  

---

## 🛠 Tech Stack

- Python 3.10+
- FastAPI + WebSockets
- Ollama (local LLM)
- OpenAI / Google Generative AI
- ChromaDB
- faster-whisper
- pyttsx3
- openwakeword
- APScheduler
- Gradio (optional UI)

---

## ⚙️ Installation

### 1. Clone repository

```bash
git clone https://github.com/n1dlee/aida_assistant.git
cd aida_assistant
```

---

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 3. Setup environment variables

Copy example file:

```bash
cp .env.example .env
```

Fill in keys if using cloud models:

```
OPENAI_API_KEY=your_key
GOOGLE_API_KEY=your_key
```

---

### 4. Setup local LLM (optional)

Install Ollama:

👉 https://ollama.com/

Run a model:

```bash
ollama run llama3
```

---

## ▶️ Running AIDA

### Option 1 — Core assistant

```bash
python main.py
```

---

### Option 2 — Web server (for UI)

```bash
python server.py
```

Then open:

- http://localhost:8000
- or connect via WebSocket `/ws`

---

### Option 3 — Gradio UI (if enabled)

```bash
python ui.py
```

---

## 🧪 Usage

- Speak or type input
- AIDA processes it via the orchestrator
- Response is generated using LLM
- Output is returned as text and/or speech

---

## 📂 Project Structure

```
aida_assistant/
├── core/              # Orchestrator and system logic
├── main.py            # Entry point
├── server.py          # FastAPI server
├── ui.py              # UI layer (Gradio)
├── requirements.txt   # Dependencies
├── .env.example       # Environment config
└── README.md
```

---

## ⚠️ Notes

- Voice features require a working microphone
- Local models depend on hardware performance
- Some features require API keys

---

## 🧠 Design Philosophy

AIDA is built around one idea:

> AI should be modular, local-first, and under your control.

No hidden pipelines.  
No unnecessary cloud dependency.  
Just a system you can extend and understand.

---

## 🔮 Future Improvements

- Better UI (React integration)
- Improved memory retrieval
- Plugin system
- Mobile deployment
- Model optimization

---

## 🤝 Contributing

Contributions are welcome.

Open issues, suggest improvements, or extend the system.

---

## 📄 License

MIT License
