"""
Pathfinder Agent

Wires the model, the tools, and the candidate profile together.

The prompt is built in two parts on every turn: the complete structured profile
(always present, never retrieved) and a few retrieved passages from the resume
and strategy guide relevant to what was just asked. See pathfinder_rag.py for
why the split is drawn there.
"""

from __future__ import annotations

import os
from datetime import datetime

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage

from pathfinder_profile import get_profile
from pathfinder_rag import get_retriever, retrieve_context
from pathfinder_tools import ALL_TOOLS

load_dotenv()
try:
    import streamlit as st
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except Exception:
    pass

MODEL_NAME = os.getenv("PATHFINDER_MODEL", "openai:gpt-4o-mini")
MODEL = init_chat_model(MODEL_NAME, temperature=0.4)

# How many past turns to carry. Unbounded history quietly turns into an
# unbounded bill, and job-search chat rarely needs more context than this.
MAX_HISTORY_TURNS = 12


BASE_SYSTEM_PROMPT = """\
You are Pathfinder, a job-search assistant built for {name}.

Today is {today}.

WHAT YOU ARE FOR
Finding roles that genuinely fit, tracking applications, and giving honest,
specific advice about the search. You are a working tool, not a cheerleader.

HOW RANKING WORKS, AND WHY YOU SHOULD NOT SECOND-GUESS IT
The search_jobs tool does not return raw listings. It screens every posting
against hard requirements, then scores the survivors from 0 to 100 across seven
weighted axes: role fit, skills overlap, location, cost-of-living-adjusted pay,
industry, job-quality signals, and freshness. It returns them already ranked,
with the reasoning attached.

Present that ranking as it comes back. Do not reorder it, and do not invent a
fit judgement that contradicts the score. What you add is interpretation: which
one or two are worth applying to first and why, what to emphasize in the
application, and what the weak spots mean in practice. If a listing scored well
but has a flag like "Staffing agency" or "No salary listed", say so plainly.

YOUR TOOLS
- search_jobs: ranked live listings. Leave keywords blank for an open-ended
  request so it sweeps the top target role families. Pass min_score=70 when
  the user only wants strong matches.
- score_posting: run the same scoring on a listing the user pastes in.
- log_application / update_application_status: maintain the tracker. When
  logging a role that came from a search, pass its fit_score and url through.
- get_applications / get_pipeline_stats: read the tracker.

RULES
- Use a tool for anything involving real listings or the tracker. Never invent
  a job, a company, a salary, or a link.
- Be concise. Lead with the answer. Bullets and tables where they help.
- Be honest about weak matches. Telling {first_name} that a mediocre role is
  great makes you useless.
- Reference specifics from the profile and resume when explaining fit. Generic
  encouragement is worth nothing.
- Do not use em dashes. Use commas, periods, parentheses, or colons.

--- CANDIDATE PROFILE (complete, authoritative) ---
{profile_block}

--- RETRIEVED CONTEXT FOR THIS QUESTION ---
{rag_context}
"""


print("[pathfinder] building vector store...")
_RETRIEVER = get_retriever(k=4)
print("[pathfinder] creating agent...")
AGENT = create_agent(model=MODEL, tools=ALL_TOOLS)
print("[pathfinder] ready.")


def _build_system_prompt(user_input: str) -> str:
    profile = get_profile()
    return BASE_SYSTEM_PROMPT.format(
        name=profile.name,
        first_name=profile.first_name,
        today=datetime.now().strftime("%A, %B %d, %Y"),
        profile_block=profile.to_prompt_block(),
        rag_context=retrieve_context(user_input, _RETRIEVER) or "(nothing retrieved)",
    )


def initialize_messages() -> list[dict[str, str]]:
    """A fresh conversation. The system prompt is rebuilt per turn, so history
    holds only the user and assistant turns."""
    return []


def get_pathfinder_response(messages: list[dict[str, str]], user_input: str):
    """Run one turn. Returns (assistant_text, updated_messages)."""
    system_message = SystemMessage(content=_build_system_prompt(user_input))
    messages.append({"role": "user", "content": user_input})

    trimmed = messages[-(MAX_HISTORY_TURNS * 2):]
    result = AGENT.invoke({"messages": [system_message] + trimmed})

    reply = _strip_dashes(result["messages"][-1].content)
    messages.append({"role": "assistant", "content": reply})
    return reply, messages


def _strip_dashes(text: str) -> str:
    """Belt and braces: the prompt asks for no em dashes, this guarantees it."""
    if not text:
        return text
    return (text.replace(" — ", ", ")
                .replace("—", ", ")
                .replace(" – ", ", ")
                .replace("–", "-"))
