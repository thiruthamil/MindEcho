# app/main.py — MindEcho API (Ollama/OpenAI-compatible)
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

import os
import re
import logging
import requests
from collections import Counter
from threading import Thread

from openai import OpenAI

from .models import EntryIn, EntryOut, WeeklySummary, SearchOut, Snippet
from .settings import settings
from .storage import VectorStore
from .db import DB
from .prompts import simple_nudge

# Load env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Fallback prompt text if prompts module is missing required fields
try:
    from . import prompts as P
except Exception:
    class P:
        CHAT_SYSTEM = (
            "Respond concisely and supportively. "
            "Use past notes only as optional context. "
            "Avoid medical/therapy claims. Suggest one small next step only if relevant."
        )
        CHAT_USER = (
            "Today's note:\n{user_text}\n\n"
            "Context:\n{context_block}\n\n"
            "Tasks:\n- Reflect back simply.\n- Note any real pattern.\n- Offer one small next step (optional)."
        )

# Ollama (OpenAI-compatible /v1)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
LLM_MODEL = os.getenv("LLM_MODEL", os.getenv("OLLAMA_CHAT_MODEL", "llama3"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "512"))

os.environ.setdefault("OPENAI_API_KEY", "ollama-local")
_oai_client = None
try:
    _oai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "ollama-local"), base_url=OLLAMA_BASE_URL)
except Exception as e:
    logging.getLogger("uvicorn.error").warning(f"Ollama client init failed: {e}")

def _warm_ollama():
    try:
        _oai_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1,
            temperature=0.0,
        )
    except Exception:
        pass

_warm_ollama()

def _ollama_status():
    """Ping native /api/tags to list models (avoids client features that may vary)."""
    try:
        native_base = OLLAMA_BASE_URL.rstrip("/").removesuffix("/v1")
        r = requests.get(f"{native_base}/api/tags", timeout=2)
        r.raise_for_status()
        data = r.json() if r.content else {}
        models = [m.get("name") for m in data.get("models", [])] if isinstance(data, dict) else []
        return True, models
    except Exception as e:
        return False, [str(e)]

app = FastAPI(title="MindEcho API", description="Backend for the MindEcho journaling app.", version="1.2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# State
vs = VectorStore(settings.data_dir, settings.embeddings_model)
db = DB(os.path.join(settings.data_dir, "entries.sqlite"))

# ---- helpers ----
def _fmt_preview(text: str, maxlen: int = 140) -> str:
    t = " ".join(text.strip().split())
    return (t[: maxlen - 1] + "…") if len(t) > maxlen else t

def _call_llm(user_text: str, context_block: str) -> str:
    if _oai_client is None:
        raise RuntimeError("LLM client not initialized")
    messages = [
        {"role": "system", "content": P.CHAT_SYSTEM},
        {"role": "user", "content": P.CHAT_USER.format(
            user_text=user_text,
            context_block=context_block if context_block.strip() else "(no strong matches)",
        )},
    ]
    resp = _oai_client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    return resp.choices[0].message.content.strip()

def _fallback_reply(user_text: str, similars: List[Snippet]) -> str:
    past = [(s.date, s.preview) for s in similars[:3]]
    try:
        return simple_nudge(user_text, past)
    except Exception:
        msg = "Noted. "
        if similars:
            msg += "I also checked a few past notes. "
        msg += "We can view similar entries or run a semantic search."
        return msg

# ---- routes ----
@app.get("/")
def root():
    return {"app": "MindEcho", "message": "Welcome to MindEcho API", "docs": "/docs", "health": "/health"}

@app.get("/health")
def health():
    ok, models = _ollama_status()
    return {
        "app": "MindEcho",
        "status": "ok",
        "entries": vs.index.ntotal,
        "llm_ok": ok,
        "llm_models": models[:8],
        "model_in_use": LLM_MODEL,
        "base_url": OLLAMA_BASE_URL,
    }

@app.post("/entry", response_model=EntryOut)
def add_entry(payload: EntryIn):
    try:
        text = payload.text.strip()
        if len(text) < 3:
            raise HTTPException(400, "Entry too short")

        faiss_id, entry_id = vs.add(text)
        db.insert_entry(faiss_id, entry_id, text)

        ids, sims = vs.search(text, k=4)
        ids = [i for i in ids if isinstance(i, int) and i >= 0]
        sims = [s for s in sims[:len(ids)]]

        if ids and ids[0] == faiss_id:
            ids, sims = ids[1:], sims[1:]

        rows = db.fetch_by_ids(ids) if ids else []
        past, similar_snips = [], []

        for i, (fid, eid, ts, t, sent, tags) in enumerate(rows):
            date = ts.split("T")[0] if isinstance(ts, str) else str(ts)
            prev = _fmt_preview(t or "")
            past.append((date, prev))
            score = float(sims[i]) if i < len(sims) else 0.0
            similar_snips.append(Snippet(id=eid, date=date, preview=prev, score=score))

        reply = simple_nudge(text, past)
        return EntryOut(reply=reply, similar=similar_snips)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"/entry failed: {e}")

@app.get("/weekly-summary", response_model=WeeklySummary)
def weekly_summary():
    rows = db.fetch_recent(days=7)
    if not rows:
        return WeeklySummary(bullets=["No entries yet. Add notes through the week to see a summary."])

    text = "\n".join(r[3] for r in rows)
    words = [w.lower() for w in re.findall(r"[a-zA-Z]{5,}", text)]
    stop = {
        "about","there","which","would","could","should","other","their",
        "these","those","having","being","because","after","before","today","tomorrow"
    }
    words = [w for w in words if w not in stop]
    common = [w for w,_ in Counter(words).most_common(3)]

    bullets = [
        f"You wrote {len(rows)} reflections this week.",
        f"Top themes: {', '.join(common) if common else '—'}.",
        "Pick one small action to repeat from a good day.",
    ]
    return WeeklySummary(bullets=bullets)

@app.get("/search", response_model=SearchOut)
def search(q: str, k: int = 5):
    ids, sims = vs.search(q, k=k)
    rows = db.fetch_by_ids(ids)
    results = []
    for i, r in enumerate(rows):
        fid, eid, ts, t, sent, tags = r
        score = float(sims[i]) if i < len(sims) else 0.0
        results.append(Snippet(
            id=eid,
            date=ts.split('T')[0],
            preview=_fmt_preview(t),
            score=score
        ))
    return SearchOut(results=results)

# ---- chat (RAG) ----
class ChatIn(BaseModel):
    text: str
    top_k: int = 5
    save: bool = True

class ChatOut(BaseModel):
    reply: str
    similar: List[Snippet] = []
    sources: List[Snippet] = []
    mode: str = "fallback"
    error: Optional[str] = None

@app.post("/chat", response_model=ChatOut)
def chat(body: ChatIn):
    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")

    faiss_id = None
    if body.save:
        faiss_id, entry_id = vs.add(text)
        db.insert_entry(faiss_id, entry_id, text)

    ids, sims = vs.search(text, k=max(body.top_k + 1, 4))
    if faiss_id is not None and ids and ids[0] == faiss_id:
        ids, sims = ids[1:], sims[1:]

    rows = db.fetch_by_ids(ids)
    similars: List[Snippet] = []
    for i, (fid, eid, ts, t, sent, tags) in enumerate(rows[:body.top_k]):
        score = float(sims[i]) if i < len(sims) else 0.0
        similars.append(Snippet(id=eid, date=ts.split("T")[0], preview=_fmt_preview(t), score=score))

    lines = [f"- {s.date} · {s.preview}" for s in similars if s.score >= 0.45]
    context_block = "\n".join(lines) if lines else "(no strong matches)"

    used_mode = "fallback"
    err_text = None
    try:
        reply = _call_llm(text, context_block)
        used_mode = "llm"
    except Exception as e:
        logging.getLogger("uvicorn.error").warning(f"/chat LLM error: {e}")
        reply = _fallback_reply(text, similars)
        err_text = str(e)

    return ChatOut(reply=reply, similar=similars, sources=similars[:3], mode=used_mode, error=err_text)

# ---- reindex (preserve entries, rebuild faiss ids) ----
@app.post("/reindex")
def reindex():
    global vs
    try:
        try:
            if vs.index_path.exists():
                vs.index_path.unlink()
            if vs.id_path.exists():
                vs.id_path.unlink()
        except Exception as e:
            raise HTTPException(500, f"Failed to clear FAISS files: {e}")

        from .storage import VectorStore as _VS
        vs = _VS(settings.data_dir, settings.embeddings_model)

        rows = db.fetch_all()  # (faiss_id, entry_id, ts, text, sentiment, tags)
        if not rows:
            db.replace_all([])
            return {"status": "ok", "reindexed": 0}

        new_rows = []
        count = 0
        for _old_fid, entry_id, ts, text, sentiment, tags in rows:
            if not text or not str(text).strip():
                continue
            new_fid, _ = vs.add(str(text).strip())
            new_rows.append((new_fid, entry_id, ts, text, sentiment, tags))
            count += 1

        db.replace_all(new_rows)
        return {"status": "ok", "reindexed": count}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"/reindex failed: {e}")

@app.get("/llm-status")
def llm_status():
    ok, models = _ollama_status()
    return {
        "app": "MindEcho",
        "model": LLM_MODEL,
        "llm": "online" if ok else "offline",
        "verified_models": models[:8],
        "base_url": OLLAMA_BASE_URL,
    }

def _do_warmup():
    try:
        vs.embed(["__mindecho_warmup__"])
    except Exception:
        pass
    try:
        _ = _call_llm("Warm up.", "(no strong matches)")
    except Exception:
        pass

@app.on_event("startup")
def _warm_models_on_startup():
    Thread(target=_do_warmup, daemon=True).start()
