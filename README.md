# MindEcho

MindEcho is a local journaling system built with **FastAPI**, **Streamlit**, and **FAISS**, using **Ollama** models for embeddings and reflection generation.  
It helps you record your thoughts, find related past entries, and get weekly summaries — all stored and processed locally.

---

## Tech Stack
- **Backend:** FastAPI  
- **Frontend:** Streamlit  
- **Database:** SQLite  
- **Embeddings:** `nomic-embed-text` (via Ollama)  
- **Language Model:** `llama3` or `qwen2.5:3b-instruct`  
- **Vector Index:** FAISS (cosine similarity)

Everything runs offline — no cloud API calls or external services.

---

## Setup

### 1. Create and activate a virtual environment
```bash 
python -m venv .venv
.venv\Scripts\activate           # Windows
# source .venv/bin/activate      # macOS / Linux
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Pull Ollama models
```bash
ollama pull nomic-embed-text
ollama pull llama3               # or ollama pull qwen2.5:3b-instruct
```

### 4. Run the app
Start the backend:
```bash
uvicorn app.main:app --reload --port 8000
```

Then launch the UI:
```bash
streamlit run app_ui.py
```

Access in browser:
- API → http://127.0.0.1:8000  
- UI → http://localhost:8501  

---

## Features
- Add and save journal entries  
- View similar past reflections using FAISS search  
- Chat-based and classic journaling modes  
- Weekly summaries of your recent entries  
- Fully local processing and storage (`.data/` directory)

---

## Folder Structure
```
app/
 ├── main.py         # FastAPI backend
 ├── db.py           # SQLite database operations
 ├── storage.py      # FAISS + embedding store
 ├── models.py       # Pydantic schemas
 ├── llm_client.py   # OpenAI-compatible client pointed at Ollama (/v1)
 ├── prompts.py      # Prompt templates
 └── settings.py     # Configuration

streamlit_app.py     # Streamlit frontend
.data/               # Local data storage
requirements.txt
```

---

## Requirements
```txt
fastapi==0.115.0
uvicorn[standard]==0.30.6
pydantic==2.9.2
streamlit==1.38.0
faiss-cpu==1.8.0.post1
sentence-transformers==3.0.1
numpy==1.26.4
pandas==2.2.2
scikit-learn==1.5.1
openai==1.43.0
httpx==0.27.2
python-dotenv==1.0.1
```
> Ollama must be installed separately → [https://ollama.ai](https://ollama.ai)

---