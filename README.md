# AIDA – AI Desktop Assistant

AIDA is a modular AI assistant designed to run locally or in hybrid mode, combining multiple LLM providers, memory systems, voice interaction, and tool execution.

It supports CLI, web interface (FastAPI + React), and an advanced Gradio UI with voice capabilities.

---

## Overview

AIDA is built around a central orchestration system that:

- routes user intent
- retrieves relevant memory
- optionally executes tools
- selects an appropriate LLM provider
- generates responses
- stores interaction history

---

## Core Architecture

### Orchestrator

Main logic:
core/orchestrator.py

Flow:

1. Detect intent using IntentRouter  
2. Attempt direct tool execution  
3. Retrieve memory from vector store  
4. Build system prompt (personality + memory)  
5. Select LLM provider  
6. Generate response  
7. Store interaction  

---

## Model Selection

brain/selector.py

### Providers

- Local LLM (Ollama)
- Cloud LLM (Gemini / OpenAI)
- Groq (OpenAI-compatible)

### Routing Priority

1. Groq (if available)
2. Cloud (complex queries)
3. Local (simple queries)

Automatic fallback if provider fails.

---

## Memory System

memory/

### Components

- buffer.py → short-term memory  
- vector_store.py → long-term memory (ChromaDB or keyword fallback)  
- episodic.py → logs to data/episodes.json  

---

## Tools System

tools/

### Available Tools

- System Tool → opens apps (notepad, browser, terminal)
- Web Tool → DuckDuckGo search
- Calendar Tool → local JSON (data/calendar.json)
- Time Tool → system time
- Filesystem Tool → file operations

Filesystem supports:
- read/write files
- create/delete folders
- rename
- list directories

Includes protection for system directories.

---

## Voice System

voice/

### Features

- microphone input
- Whisper transcription
- TTS (pyttsx3 / Piper)
- wake word (OpenWakeWord)
- transcription server
- TTS server

### Important

Two separate pipelines:

1. React UI → browser speech recognition  
2. Gradio UI → local Whisper + TTS  

---

## Interfaces

### CLI

python main.py

---

### Web (FastAPI + React)

Backend:
server.py

Endpoints:
- /ws
- /api/status
- /api/history
- DELETE /api/history

Frontend:
frontend/

---

### Gradio UI

python ui.py

Features:
- streaming responses
- voice input/output
- wake word
- persistent models
- chat history

---

## Configuration

### YAML

config/settings.yaml  
config/prompts.yaml  

### Environment Variables

Primary config is handled via env variables:

- AIDA_LOCAL_MODEL  
- AIDA_CLOUD_THRESHOLD  
- WHISPER_MODEL  
- AIDA_WAKE_WORD  
- AIDA_TTS_RATE  
- PIPER_VOICE  

---

## Proactive System (Experimental)

proactive/

Contains:
- scheduler
- trigger engine

Not fully integrated.

---

## Project Structure

aida_assistant/
├── brain/
├── core/
├── memory/
├── tools/
├── voice/
├── proactive/
├── config/
├── data/
├── frontend/
├── main.py
├── server.py
├── ui.py

---

## Key Files

- core/orchestrator.py  
- brain/selector.py  
- memory/vector_store.py  
- tools/registry.py  
- ui.py  
- server.py  
- tools/filesystem_tool.py  

---

## Limitations

- tool routing is keyword-based  
- config is split between YAML and env  
- proactive system incomplete  
- React and Gradio voice not unified  
- memory retrieval is basic  

---

## Summary

AIDA is a modular assistant framework with:

- multi-model routing  
- memory system  
- tool execution  
- voice interaction  
- multiple interfaces  

Suitable for experimentation and local AI workflows.
