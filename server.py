# server.py — FastAPI bot backend for Second Life (legacy OpenAI client)
# Start: uvicorn server:app --host 0.0.0.0 --port $PORT

import os, sqlite3, time
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel

# ─── OpenAI legacy client ─────────────────────────────────────────────────
import openai
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY
    print("[startup] OpenAI legacy client ready (0.28.x)")
else:
    print("[startup] No OPENAI_API_KEY set; using rule-based replies")

# ─── App & DB ────────────────────────────────────────────────────────────
DB_PATH = "memory.db"
app = FastAPI()

def db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""CREATE TABLE IF NOT EXISTS convos(
        ts INTEGER, agent_key TEXT, agent_name TEXT, object_key TEXT, region TEXT, message TEXT
    )""")
    con.execute("""CREATE TABLE IF NOT EXISTS memory(
        agent_key TEXT, fact TEXT, UNIQUE(agent_key, fact) ON CONFLICT IGNORE
    )""")
    con.execute("""CREATE TABLE IF NOT EXISTS consent(
        agent_key TEXT PRIMARY KEY, allowed INTEGER
    )""")
    return con

# ─── Models ──────────────────────────────────────────────────────────────
class Payload(BaseModel):
    agent_key: str
    agent_name: str
    message: str
    object_name: str
    object_key: str
    position: str
    region: str
    timestamp: int

# ─── Memory helpers ──────────────────────────────────────────────────────
def get_mem(con, agent_key):
    cur = con.execute("SELECT fact FROM memory WHERE agent_key=?", (agent_key,))
    return [r[0] for r in cur.fetchall()]

def set_consent(con, agent_key, allowed: bool):
    con.execute(
        "INSERT OR REPLACE INTO consent(agent_key, allowed) VALUES (?,?)",
        (agent_key, 1 if allowed else 0)
    )
    con.commit()

def get_consent(con, agent_key) -> bool:
    cur = con.execute("SELECT allowed FROM consent WHERE agent_key=?", (agent_key,))
    row = cur.fetchone()
    return (row and row[0] == 1)

def add_fact(con, agent_key, fact):
    con.execute(
        "INSERT OR IGNORE INTO memory(agent_key, fact) VALUES (?,?)",
        (agent_key, fact.strip())
    )
    con.commit()

def remove_fact(con, agent_key, fact):
    con.execute(
        "DELETE FROM memory WHERE agent_key=? AND fact=?",
        (agent_key, fact.strip())
    )
    con.commit()

# ─── Rules fallback ──────────────────────────────────────────────────────
def rule_based_reply(name: str, msg: str, mem: List[str]) -> str:
    low = msg.lower().strip()
    if low in ("hi", "hello", "hey", "yo", "sup"):
        return f"hey {name.split(' ')[0] if name else 'there'}! how’s it going?"
    if "how are you" in low:
        return "i'm doing pretty well—appreciate you asking!"
    if "who are you" in low:
        return "i'm your friendly in-world bot. try 'help' for tricks."
    if "help" in low:
        return "i can chat, and store/forget facts: 'remember: i like oolong tea' or 'forget: i like oolong tea'. toggle learning with 'consent: yes/no'."
    if "remember:" in low:
        return "got it—i saved that. (say 'forget: ...' to remove it)"
    if "forget:" in low:
        return "ok, scrubbed from memory."
    if "bye" in low or "goodnight" in low or "good night" in low:
        return "catch you later!"
    if mem:
        return f"noted. btw i remember: {', '.join(mem[:3])}"[:300]
    return "ok! tell me more?"

# ─── LLM helper (legacy client) ──────────────────────────────────────────
def llm_reply(agent_name: str, msg: str, mem: List[str], consent_ok: bool) -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("No OPENAI_API_KEY")

    system = (
        "You are a friendly in-world NPC living in Second Life.\n"
        "Keep replies short (<= 2 sentences) and ask occasional follow-ups.\n"
        "Respect consent: only use 'memory' if the user opted in.\n"
    )
    if consent_ok and mem:
        system += "Known preferences for this user: " + "; ".join(mem[:10]) + "\n"

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"{agent_name}: {msg}"},
    ]

    # legacy API call
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",     # stable on legacy client
        messages=messages,
        temperature=0.7,
        max_tokens=120,
    )
    return resp.choices[0].message["content"].strip()

# ─── Routes ──────────────────────────────────────────────────────────────
@app.get("/")
async def root_status():
    return {"ok": True, "hint": "POST /chat (or /) with the bot payload"}

@app.post("/")
async def chat_root_alias(payload: Payload):
    return await chat(payload)

@app.post("/chat")
async def chat(payload: Payload):
    con = db()
    con.execute(
        "INSERT INTO convos VALUES (?,?,?,?,?,?)",
        (int(time.time()), payload.agent_key, payload.agent_name,
         payload.object_key, payload.region, payload.message)
    )
    con.commit()

    msg = payload.message.strip()
    low = msg.lower()

    # Commands
    if low.startswith("consent:"):
        allow = "yes" in low
        set_consent(con, payload.agent_key, allow)
        return {"reply": "thanks! learning is now " + ("ON." if allow else "OFF.")}

    if low.startswith("remember:"):
        fact = msg.split(":", 1)[1].strip()
        add_fact(con, payload.agent_key, fact)
        return {"reply": f"saved: '{fact}'"}

    if low.startswith("forget:"):
        fact = msg.split(":", 1)[1].strip()
        remove_fact(con, payload.agent_key, fact)
        return {"reply": f"forgot: '{fact}'"}

    mem = get_mem(con, payload.agent_key)
    consent_ok = get_consent(con, payload.agent_key)

    # Try LLM; fall back to rules on any error
    if OPENAI_API_KEY:
        try:
            text = llm_reply(payload.agent_name, msg, mem, consent_ok)
            return {"reply": f"{text} [llm]"}
        except Exception as e:
            print(f"[llm error] {e}")
            return {"reply": f"(fallback) LLM error: {e}"}

    return {"reply": rule_based_reply(payload.agent_name, msg, mem)}

