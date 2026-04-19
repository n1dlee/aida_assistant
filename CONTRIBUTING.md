# Contributing to AIDA Assistant

Thank you for your interest in contributing to AIDA Assistant. We welcome contributions of all kinds, including bug fixes, new features, improvements, and documentation.

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/aida_assistant.git
   ```
3. Navigate into the project:
   ```bash
   cd aida_assistant
   ```
4. Create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Project Structure Overview

- `brain/` — core intelligence and decision-making logic  
- `memory/` — memory systems and state handling  
- `proactive/` — proactive behaviors and triggers  
- `tools/` — integrations and external tools  
- `voice/` — voice processing and interaction  
- `frontend/` — UI components  
- `core/`, `config/`, `data/` — system configuration and data handling  

Understanding this structure will help you contribute effectively.

## Development Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the main application:

```bash
python main.py
```

## Contribution Guidelines

- Write clear, readable, and maintainable code
- Follow existing project structure and naming conventions
- Keep functions small and focused
- Add comments where necessary, but avoid obvious comments
- Update documentation when your changes affect usage

## Commit Guidelines

Use meaningful commit messages:

```
feat: add proactive memory trigger system
fix: resolve crash in voice module
docs: update README with setup instructions
```

## Pull Requests

Before submitting a PR:

- Ensure your code runs without errors
- Test your changes
- Make sure your branch is up to date with `main`

When submitting:

- Provide a clear description of what you changed and why
- Link related issues if applicable

## Reporting Issues

If you find a bug or have a feature request:

- Open an issue
- Provide a clear description
- Include steps to reproduce (if applicable)

## Code Style

- Use consistent formatting
- Prefer readability over cleverness
- Keep logic simple and explicit

## Final Notes

AIDA Assistant is an evolving system. Contributions that improve clarity, modularity, and intelligence of the system are especially valuable.

Build carefully. Think deeply. Keep it clean.
