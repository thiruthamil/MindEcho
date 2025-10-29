# streamlit_app.py
import os
import re
import requests
import streamlit as st

st.set_page_config(page_title="MindEcho", layout="centered")

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# ---------- HTTP helpers ----------
def api_get(path, **params):
    try:
        r = requests.get(f"{API_URL}{path}", params=params, timeout=20)
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.RequestException as e:
        return None, e

def api_post(path, body):
    try:
        r = requests.post(f"{API_URL}{path}", json=body, timeout=60)
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.RequestException as e:
        return None, e

def backend_ok():
    try:
        requests.get(f"{API_URL}/openapi.json", timeout=3)
        return True
    except Exception:
        return False

# ------------ CSS ------------
st.markdown("""
<style>
  .center-title { text-align:center; font-size:3em; font-weight:700; margin-bottom:.3em; }
  .center-caption { text-align:center; color:#9ca3af; font-size:1.05em; margin-bottom:1.2em; }
  .chat-card { background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.1);
               border-radius:14px; padding:16px 20px; box-shadow:0 6px 18px rgba(0,0,0,0.2); }
</style>
""", unsafe_allow_html=True)

# ------------ Backend status ------------
st.caption(f"Backend: {'✅ connected' if backend_ok() else '❌ offline'} · API_URL={API_URL}")

# ------------ Session defaults ------------
defaults = {
    "__pending_reflection": None,
    "reflection_input": "",
    "similar_latest": [],
    "chat_mode": False,
    "chat_msgs": [],         # list of dicts: {"role","text", optional "sims":[...] }
    "chat_phase": "intro",
    "chat_prompt": "What’s on your mind right now?",
    "show_summary": False,
    "summary_data": [],
    "search_panel_open": False,
}
for k, v in defaults.items():
    st.session_state.setdefault(k, v)

# ------------ Header ------------
st.markdown("<div class='center-title'>MindEcho</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='center-caption'>Your AI-powered journaling companion that reflects your thoughts back to you 💭</div>",
    unsafe_allow_html=True,
)

# ------------ Chat Mode toggle ------------
chat_mode = st.toggle("Chat mode", value=st.session_state.chat_mode)
st.session_state.chat_mode = chat_mode

# =========================
# CHAT MODE
# =========================
if chat_mode:
    prompts = [
        "What’s on your mind right now?",
        "How are you really feeling today?",
        "What made you smile or laugh recently?",
        "Is there something that’s been on your mind for a while?",
        "Who or what are you grateful for today?",
        "What’s something you’d like to improve or change?",
        "What moment today felt meaningful or peaceful?",
        "What’s one small thing you want to focus on tomorrow?",
    ]

    # Initial message
    if not st.session_state.chat_msgs:
        st.session_state.chat_msgs.append({"role": "assistant", "text": "Hey, I’m here. Want to jot down today’s entry?"})
        st.session_state.chat_phase = "await_entry"

    # Restart chat
    _, restart_col = st.columns([0.7, 0.3])
    with restart_col:
        if st.button("↺ Restart chat", use_container_width=True):
            st.session_state.chat_msgs = []
            st.session_state.similar_latest = []
            st.session_state.chat_phase = "intro"
            st.session_state.chat_prompt = "What’s on your mind right now?"
            st.rerun()

    # Render chat history. If a message has "sims", render a dropdown just below that message.
    for m in st.session_state.chat_msgs:
        with st.chat_message("assistant" if m["role"] == "assistant" else "user"):
            st.markdown(m["text"])
            sims = m.get("sims") or []
            if sims:
                STRONG, WEAK = 0.70, 0.45
                best = sims[0]["score"] if sims else 0.0
                if best >= WEAK:
                    label = "Similar entries (likely related)" if best >= STRONG else "Similar entries (may not be perfect)"
                    with st.expander(label, expanded=False):
                        for s in sims[:5]:
                            st.markdown(f"- _{s['date']} · {s['score']:.2f}_ — {s['preview']}")

    phase = st.session_state.chat_phase

    # Thought-starter (only for new entry)
    if phase == "await_entry":
        st.selectbox(
            "Thought starter",
            prompts,
            index=prompts.index(st.session_state.chat_prompt) if st.session_state.chat_prompt in prompts else 0,
            key="chat_prompt",
        )

    # SEARCH PHASE (single input)
    if phase == "await_search_query":
        q = st.chat_input("Type what to search…")
        if q:
            data, err = api_get("/search", q=q, k=5)
            if err:
                st.session_state.chat_msgs.append({"role": "assistant", "text": "Search failed to reach the server."})
            else:
                res = data.get("results", [])
                if not res:
                    st.session_state.chat_msgs.append({"role": "assistant", "text": "No close matches yet — try a broader search."})
                else:
                    lines = "\n".join([f"- _{s['date']} · {s['score']:.2f}_ — {s['preview']}" for s in res])
                    st.session_state.chat_msgs.append({"role": "assistant", "text": "**Closest matches:**\n" + lines})
            st.session_state.chat_phase = "ask_add_more"
            st.rerun()
        st.stop()

    # Generic chat input
    placeholder = "Type your reply…" if phase != "await_entry" else st.session_state.chat_prompt
    user_text = st.chat_input(placeholder)

    if user_text:
        st.session_state.chat_msgs.append({"role": "user", "text": user_text})
        phase = st.session_state.chat_phase

        # AWAIT ENTRY
        if phase == "await_entry":
            data, err = api_post("/chat", {"text": user_text, "top_k": 5, "save": True})
            if err:
                st.session_state.chat_msgs.append({"role": "assistant", "text": "I couldn’t reach the server just now. Try again later."})
            else:
                reply = (data.get("reply") or "").strip()
                sims = data.get("similar", []) or []
                st.session_state.similar_latest = sims

                # Put the LLM reply and the Similar entries expander in the SAME assistant message
                st.session_state.chat_msgs.append({
                    "role": "assistant",
                    "text": f"**Noted — saved today’s entry.**\n\n{reply}",
                    "sims": sims,   # <- rendered inline right below this message
                })

                # Follow-up prompt as a separate assistant line (keeps flow the same)
                st.session_state.chat_msgs.append(
                    {"role": "assistant", "text": "_You can reply to continue the conversation, or use the options below._"}
                )
                st.session_state.chat_phase = "ask_add_more"
            st.rerun()

        # ASK ADD MORE
        elif phase == "ask_add_more":
            ans = user_text.strip().lower()
            if ans in ["yes", "y"]:
                st.session_state.chat_msgs.append({"role": "assistant", "text": "Great — what’s next? Type your next thought below."})
                st.session_state.chat_phase = "await_entry"
            elif ans in ["no", "n", "finish", "done"]:
                st.session_state.chat_msgs.append({"role": "assistant", "text": "Okay. Session saved — see you next time!"})
                st.session_state.chat_phase = "done"
            else:
                st.session_state.chat_msgs.append({"role": "assistant", "text": "Please reply Yes or No."})
            st.rerun()

    # Action buttons (ask_add_more)
    if st.session_state.chat_phase == "ask_add_more":
        col1, col2, col3 = st.columns(3)
        if col1.button("Add another entry", use_container_width=True):
            st.session_state.chat_msgs.append({"role": "assistant", "text": "Sure — type your next thought below."})
            st.session_state.chat_phase = "await_entry"
            st.rerun()

        if col2.button("Search by meaning", use_container_width=True):
            st.session_state.chat_msgs.append({"role": "assistant", "text": "What would you like me to search for?"})
            st.session_state.chat_phase = "await_search_query"
            st.rerun()

        if col3.button("Finish", use_container_width=True):
            st.session_state.chat_msgs.append({"role": "assistant", "text": "Okay. Session saved — see you next time!"})
            st.session_state.chat_phase = "done"
            st.rerun()

    st.stop()

# =========================
# CLASSIC MODE
# =========================

prompts = [
    "What’s on your mind right now?",
    "How are you really feeling today?",
    "What made you smile or laugh recently?",
    "Is there something that’s been on your mind for a while?",
    "Who or what are you grateful for today?",
    "What’s something you’d like to improve or change?",
    "What moment today felt meaningful or peaceful?",
    "What’s one small thing you want to focus on tomorrow?",
]

st.subheader("Let’s capture your day...")
st.caption("Write a few lines about what stood out — MindEcho will remember it for you.")
prompt_choice = st.selectbox("Thought starter", prompts, index=0, key="prompt_choice")

# Save entry
with st.form("reflect_form", clear_on_submit=True):
    text = st.text_area(
        "Go ahead, jot it down",
        height=120,
        placeholder=prompt_choice,
        key="reflection_input",
    )
    submit_reflect = st.form_submit_button("Save this moment", use_container_width=True)

if submit_reflect:
    if not text.strip():
        st.warning("Please write something first ✍️")
    else:
        data, err = api_post("/entry", {"text": text})
        if err:
            st.error("Couldn’t save right now. Is the server running?")
            st.caption(str(err))
            st.session_state.similar_latest = []
        else:
            st.success("Noted — saved today’s entry.")
            st.session_state.similar_latest = data.get("similar", []) or []

# Similar entries (dropdown, classic)
with st.expander("Similar entries", expanded=False):
    sims = st.session_state.get("similar_latest", [])
    if not sims:
        st.caption("No similar entries yet — save a note to see suggestions.")
    else:
        for snip in sims:
            st.markdown(f"- _{snip['date']} · {snip['score']:.2f}_ — {snip['preview']}")

# ---------- Weekly Summary ----------
def _parse_weekly(bullets: list[str]):
    text = " ".join(bullets)
    m = re.search(r"(\d+)\s+reflections", text, re.I)
    count = int(m.group(1)) if m else None
    t = re.search(r"Top themes:\s*([^\.]+)", text, re.I)
    themes = [s.strip() for s in (t.group(1).split(",") if t else []) if s.strip()]
    return count, themes

if not st.session_state.show_summary:
    if st.button("Weekly Summary", use_container_width=True):
        data, err = api_get("/weekly-summary")
        if err:
            st.error("Failed to fetch summary.")
            st.caption(str(err))
        else:
            st.session_state.summary_data = data.get("bullets", [])
            st.session_state.show_summary = True
else:
    bullets = st.session_state.get("summary_data", [])
    kpi, themes = _parse_weekly(bullets)

    st.markdown("<div class='ws-card'>", unsafe_allow_html=True)
    st.markdown("<div class='ws-title'>This Week</div>", unsafe_allow_html=True)

    c1, c2 = st.columns([0.35, 0.65])
    with c1:
        st.markdown(f"<div class='ws-kpi'>{kpi if kpi is not None else '—'}</div>", unsafe_allow_html=True)
        st.markdown("<div class='ws-muted'>entries</div>", unsafe_allow_html=True)
    with c2:
        if themes:
            st.markdown("<div class='ws-muted'>Top themes</div>", unsafe_allow_html=True)
            st.markdown(" ".join(f"<span class='ws-chip'>#{t}</span>" for t in themes), unsafe_allow_html=True)
        else:
            st.markdown("<div class='ws-muted'>No themes yet</div>", unsafe_allow_html=True)

    if themes:
        st.markdown("<div class='ws-muted' style='margin-top:8px;'>Try a tiny step</div>", unsafe_allow_html=True)
        cols = st.columns(min(3, len(themes)))
        actions = [f"Do one small thing for #{t} tomorrow." for t in themes[:3]]
        for i, a in enumerate(actions):
            with cols[i]:
                if st.button(a, key=f"ws_action_{i}", use_container_width=True):
                    st.session_state["__pending_reflection"] = a
                    st.session_state.show_summary = False
                    st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("Back to Journal", use_container_width=True):
        st.session_state.show_summary = False

# ---------- Search by Meaning (classic panel) ----------
if not st.session_state.search_panel_open:
    st.write("")
    if st.button("Search by Meaning", use_container_width=True):
        st.session_state.search_panel_open = True
else:
    st.markdown("<div class='search-panel'>", unsafe_allow_html=True)

    h1, h2 = st.columns([0.75, 0.25])
    with h1:
        st.markdown("<div class='search-header'>Search by Meaning</div>", unsafe_allow_html=True)
        st.caption("Find notes that match how you feel or what you’re thinking 💭")
    with h2:
        if st.button("Back to Journal", key="search_close_btn", use_container_width=True):
            st.session_state.search_panel_open = False
            st.rerun()

    with st.form("search_form", clear_on_submit=False):
        query = st.text_input(
            "Revisit your entries",
            placeholder="e.g., interview nervousness",
            key="search_input",
        )
        submit_search = st.form_submit_button("Search", use_container_width=True)

    if submit_search:
        if not query.strip():
            st.warning("Type something to search 🔎")
        else:
            data, err = api_get("/search", q=query, k=5)
            if err:
                st.error("Search failed.")
                st.caption(str(err))
            else:
                res = data.get("results", [])
                if not res:
                    st.info("No close matches yet — add a few more reflections and try again.")
                else:
                    st.markdown("**Closest notes:**")
                    for snip in res:
                        st.markdown(f"- _{snip['date']} · {snip['score']:.2f}_ — {snip['preview']}")

    st.markdown("</div>", unsafe_allow_html=True)
