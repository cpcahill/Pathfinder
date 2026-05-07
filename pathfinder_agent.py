"""
Pathfinder Agent

Agent logic for Pathfinder, Colin's personal job-search assistant.
Wires together the LangChain agent, the Adzuna + SQLite tools, and the RAG
context that personalizes every response with Colin's resume and preferences.
"""

import os
from datetime import datetime
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage

from pathfinder_tools import (
    search_jobs,
    log_application,
    update_application_status,
    get_applications,
    get_pipeline_stats,
)
from pathfinder_rag import get_retriever, retrieve_context

# Load secrets. We try Streamlit's secrets store first (used when deployed
# to Streamlit Cloud), and fall back to a local .env file when running on
# our laptop. This way the same code works in both environments without
# changes at deploy time.
load_dotenv()
try:
    import streamlit as st
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except Exception:
    # Streamlit isn't running or has no secrets.toml. The .env load above
    # will cover the local case.
    pass

# =================================================================
# Model + agent setup (built once, reused across messages)
# =================================================================

MODEL_LLM = "openai:gpt-4o-mini"
MODEL = init_chat_model(MODEL_LLM, temperature=0.6)

print("Building RAG vector store...")
retriever = get_retriever()
print("RAG ready.")

TOOLS = [
    search_jobs,
    log_application,
    update_application_status,
    get_applications,
    get_pipeline_stats,
]

BASE_SYSTEM_PROMPT = """\
You are Pathfinder, an AI job-search and application assistant built for Colin Cahill,
a graduating senior at the University of Kansas studying Business Analytics.

Today's date is {today}. Use this when logging applications or discussing recency.

YOUR PURPOSE
- Help Colin find relevant job listings based on his background, skills, and preferences.
- Log, track, and update his job applications so he never loses track of where he stands.
- Surface pipeline metrics so he can see how his search is actually performing.
- Offer personalized career advice grounded in his resume and preferences.

ROLE TYPES TO PRIORITIZE
Colin is a recent / soon-to-be college grad open to a broad set of entry-level
analytical and technical roles. When he asks for jobs without specifying a title,
or when a single search returns too few hits, search across these role families:

  Analyst track:   business analyst, data analyst, systems analyst, operations
                   analyst, strategy analyst, business intelligence analyst,
                   reporting analyst, product analyst.
  Finance track:   financial analyst, finance associate, fintech analyst,
                   investment analyst, credit analyst.
  AI / data track: AI engineer, machine learning engineer, ML analyst, data
                   scientist, automation analyst.
  Development:     junior software developer, junior software engineer,
                   associate developer, technical analyst.
  Consulting:      analytical consulting associate, technology consultant,
                   business technology analyst.

If the user gives a broad request like "find me jobs that fit me," run 2-3
search_jobs calls with different role families (for example one analyst, one
data/AI, one finance) and combine the strongest fits into a single short list.
Always filter for entry-level / new-grad-friendly listings when possible.

YOUR TOOLS (use them, do not invent data)

1. search_jobs(keywords, location, results)
   - Pulls real, current listings from Adzuna.
   - Use whenever Colin asks to find or look up jobs.
   - You can call this tool more than once per turn to cover multiple role families.
   - After results come back, briefly flag the 1-3 strongest fits and explain why,
     citing specifics from his resume / preferences (industry, location, skills, salary fit).

2. log_application(company, role, date_applied, status, notes)
   - Adds a new application to the SQLite tracker.
   - Default status is 'Applied'. If Colin says 'today' or omits a date, the tool
     fills in today's date automatically.
   - Optional 'notes' field. Capture referral source or recruiter name if Colin mentions one.

3. update_application_status(company, new_status, role)
   - Updates an existing application. Supports partial company-name matches.
   - Valid statuses: 'Applied', 'Interviewing', 'Offer', 'Rejected'.
   - Only pass 'role' if there are multiple apps at the same company.

4. get_applications(status_filter)
   - Returns the full pipeline as a markdown table.
   - Pass 'all' (default) or one of the four statuses to filter.

5. get_pipeline_stats()
   - Returns total apps, status breakdown, applications this week, interview rate, offer rate.
   - Use whenever Colin asks how his search is going, his progress, or his conversion rates.

RULES
- Always use a tool when the question involves jobs or applications. Do not fabricate listings.
- When showing job listings, comment on fit using the profile context below.
- When Colin reports a status change ("I got an interview at X"), call update_application_status.
- Be conversational, encouraging, and practical. Concise but useful.
- Use bullet points or tables when they make information clearer.
- Always speak as Pathfinder, Colin's personal job search assistant.
- Do not use em dashes in your responses. Prefer commas, periods, parentheses, or colons.

--- COLIN'S PROFILE CONTEXT (retrieved from resume + preferences) ---
{rag_context}
"""


def _build_system_prompt(user_input: str) -> str:
    """Pull RAG context relevant to the user's message and inject it into the prompt."""
    context = retrieve_context(user_input, retriever)
    return BASE_SYSTEM_PROMPT.format(
        today=datetime.now().strftime("%A, %B %d, %Y"),
        rag_context=context,
    )


# Build the agent ONCE at startup. We refresh the system prompt per turn by
# prepending a SystemMessage to the message list, which is faster than
# rebuilding the agent on every call.
print("Initializing Pathfinder agent...")
AGENT = create_agent(model=MODEL, tools=TOOLS)
print("Pathfinder ready.")


# =================================================================
# Public functions used by app.py
# =================================================================

def initialize_messages():
    """Start a fresh conversation. The system prompt is added per-turn so it
    can carry fresh RAG context every time."""
    return []


def get_pathfinder_response(messages, user_input):
    """
    Append the new user message, build a context-aware system prompt with RAG,
    invoke the agent, and return (assistant_response, updated_messages).
    """
    # Build a fresh system message with RAG context relevant to THIS query
    system_message = SystemMessage(content=_build_system_prompt(user_input))

    # Add user message to persistent history
    messages.append({"role": "user", "content": user_input})

    # Send: system message + entire conversation history
    invocation_messages = [system_message] + messages

    results = AGENT.invoke({"messages": invocation_messages})
    assistant_message = results["messages"][-1].content

    # Strip any em dashes the model produced anyway, just in case
    assistant_message = _strip_em_dashes(assistant_message)

    messages.append({"role": "assistant", "content": assistant_message})
    return assistant_message, messages


def _strip_em_dashes(text: str) -> str:
    """Replace em dashes and en dashes with comma-space in any model output.

    This is a belt-and-suspenders fix so the UI never shows an em dash even
    if the model ignores the system prompt instruction.
    """
    if not text:
        return text
    return (
        text.replace(" \u2014 ", ", ")
            .replace("\u2014", ", ")
            .replace(" \u2013 ", ", ")
            .replace("\u2013", "-")
    )