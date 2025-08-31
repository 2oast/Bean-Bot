# server.py
# pip install fastapi uvicorn pydantic "openai>=1.0.0"
# Run: uvicorn server:app --host 0.0.0.0 --port 8000
import os, sqlite3, time
from typing import Optional
from fastapi import FastAPI, Request, Header, HTTPException
from pydantic import BaseModel

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # optional
SHARED_SECRET  = os.getenv("SHARED_SECRET", "change-me-please")

try:
    from openai import OpenAI
    oai = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
except Exception:
    oai = None

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

class Payload(BaseModel):
    agent_key: str
    agent_name: str
    message: str
    object_name: str
    object_key: str
    position: str
    region: str
    timestamp: int

def get_mem(con, agent_key):
    cur = con.execute("SELECT fact FROM memory WHERE agent_key=?", (agent_key,))
    return [r[0] for r in cur.fetchall()]

def set_consent(con, agent_key, allowed: bool):
    con.execute("INSERT OR REPLACE INTO consent(agent_key, allowed) VALUES (?,?)", (agent_key, 1 if allowed else 0))
    con.commit()

def get_consent(con, agent_key) -> bool:
    cur = con.execute("SELECT allowed FROM consent WHERE agent_key=?", (agent_key,))
    row = cur.fetchone()
    return (row and row[0] == 1)

def add_fact(con, agent_key, fact):
    con.execute("INSERT OR IGNORE INTO memory(agent_key, fact) VALUES (?,?)", (agent_key, fact.strip()))
    con.commit()

def remove_fact(con, agent_key, fact):
    con.execute("DELETE FROM memory WHERE agent_key=? AND fact=?", (agent_key, fact.strip()))
    con.commit()

def rule_based_reply(name: str, msg: str, mem: list[str]) -> str:
    low = msg.lower().strip()
    if low in ("hi", "hello", "hey", "yo"):
        return f"hey {name.split(' ')[0]}! what are you up to?"
    if "help" in low:
        return "i can chat, remember things if you ask, and forget them too. try: 'remember: i like oolong tea'"
    if "remember:" in low:
        return "got it—i saved that. (say 'forget: ...' to remove it)"
    if "forget:" in low:
        return "ok, scrubbed from memory."
    # light personalization with known facts
    if mem:
        return f"noted! btw i remember: {', '.join(mem[:3])}"[:300]
    return "ok! tell me more?"

@app.post("/chat")
async def chat(payload: Payload, x_auth: Optional[str] = Header(None)):
    if x_auth != SHARED_SECRET:
        raise HTTPException(401, "bad secret")

    con = db()
    con.execute("INSERT INTO convos VALUES (?,?,?,?,?,?)", (
        int(time.time()),
        payload.agent_key,
        payload.agent_name,
        payload.object_key,
        payload.region,
        payload.message
    ))
    con.commit()

    msg = payload.message.strip()
    # Commands
    low = msg.lower()
    if low.startswith("consent:"):
        allow = "yes" in low
        set_consent(con, payload.agent_key, allow)
        return {"reply": "thanks! learning is now " + ("ON." if allow else "OFF.")}

    if low.startswith("remember:"):
        fact = msg.split(":",1)[1].strip()
        add_fact(con, payload.agent_key, fact)
        return {"reply": f"saved: '{fact}'"}

    if low.startswith("forget:"):
        fact = msg.split(":",1)[1].strip()
        remove_fact(con, payload.agent_key, fact)
        return {"reply": f"forgot: '{fact}'"}

    mem = get_mem(con, payload.agent_key)

    # If you have an API key, use an LLM; otherwise simple mode
    if oai:
        # Build a small system prompt with local memory
        sys = (
            "You are a friendly in-world NPC living in Second Life.\n"
            "Keep replies short (<= 2 sentences) and ask occasional follow-ups.\n"
            "Respect consent: only use 'memory' if the user opted in.\n"
        )
        if get_consent(con, payload.agent_key) and mem:
            sys += "Known preferences for this user: " + "; ".join(mem[:10]) + "\n"

        messages = [
            {"role":"system", "content": sys},
            {"role":"user", "content": f"{payload.agent_name}: {msg}"}
        ]
        # Use GPT (or any compatible provider) if configured
        try:
            resp = oai.chat.completions.create(
                model="gpt-4o-mini",  # or any chat-capable model you prefer
                messages=messages,
                temperature=0.7,
                max_tokens=120,
            )
            text = resp.choices[0].message.content.strip()
            return {"reply": text}
        except Exception as e:
            # fall back gracefully
            return {"reply": rule_based_reply(payload.agent_name, msg, mem)}

    # No LLM configured: rule-based fallback
    return {"reply": rule_based_reply(payload.agent_name, msg, mem)}
