# GrothendieckAI

**GrothendieckAI** is an intelligent assistant for exploring advanced mathematics. It allows users to upload PDFs, ask context-aware math questions, chat with an LLM tutor, and run algebraic computations using Macaulay2.

---

## 🚀 Features

- 📄 **PDF Question Answering**: Upload mathematical documents and ask context-specific questions using retrieval-augmented generation.
- 🤖 **LLM Tutor**: Chat with a math-savvy AI assistant powered by Meta's LLaMA 3.2 via Groq API.
- 💻 **Macaulay2 Interface**: Run algebraic geometry and commutative algebra code directly in-browser.
- 🖼️ **Library Manager**: Upload, browse, and manage PDFs and whiteboard images.
- 🧠 **History Tracking**: View and edit your question-answer history.
- ✏️ **Whiteboard Save**: Save canvas-based whiteboard drawings (in progress).

---

## 🧰 Tech Stack

- **Backend**: Django (Python)
- **Frontend**: TailwindCSS (via Django templates)
- **AI Models**: Meta LLaMA 3.2 via Groq API
- **Computation**: Macaulay2 subprocess
- **Vector Search**: FAISS (PDF chunking & embedding)
- **Storage**: Local media files, Django ORM

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/Daviddavid-sudo/grothendieckai.git
cd grothendieckai
