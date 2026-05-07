"""
Pathfinder: Streamlit App
The browser interface for Pathfinder, Colin's AI job-search assistant.

This file owns the look and feel only.
All agent logic lives in pathfinder_agent.py, all tool logic lives in
pathfinder_tools.py, and all RAG content lives in pathfinder_rag.py.
"""

import os
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

from pathfinder_agent import initialize_messages, get_pathfinder_response
from pathfinder_tools import get_pipeline_dataframe


# =================================================================
# PAGE CONFIG
# =================================================================

st.set_page_config(
    page_title="Pathfinder Job Search Assistant",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =================================================================
# CUSTOM CSS: KU blue theme + readability fixes
#
# Important: this app is designed for a LIGHT theme. We force readable
# colors on chat bubbles and cards so the app stays legible regardless
# of the user's system or Streamlit theme setting.
# =================================================================

st.markdown("""
<style>
    /* Color palette: KU blue inspired */
    :root {
        --pf-deep-blue: #0033A0;
        --pf-mid-blue:  #1E5BC6;
        --pf-light-blue:#E8F0FE;
        --pf-accent:    #4A90E2;
        --pf-text:      #1a1a2e;
        --pf-muted:     #4b5563;
    }

    /* Header band */
    .pf-header {
        background: linear-gradient(135deg, #0033A0 0%, #1E5BC6 60%, #4A90E2 100%);
        padding: 1.6rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.2rem;
        color: white;
        box-shadow: 0 4px 14px rgba(0,51,160,0.18);
    }
    .pf-header h1 {
        margin: 0;
        font-size: 2.0rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        color: white;
    }
    .pf-header p {
        margin: 0.4rem 0 0 0;
        opacity: 0.95;
        font-size: 1.0rem;
        color: white;
    }

    /* Sidebar metric cards */
    .pf-metric-card {
        background: white;
        border: 1px solid #E1E8F5;
        border-left: 4px solid var(--pf-mid-blue);
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.6rem;
    }
    .pf-metric-label {
        font-size: 0.78rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 2px;
    }
    .pf-metric-value {
        font-size: 1.55rem;
        font-weight: 700;
        color: var(--pf-deep-blue);
        line-height: 1.1;
    }
    .pf-metric-sub {
        font-size: 0.78rem;
        color: #6b7280;
        margin-top: 2px;
    }

    /* Quick-action buttons */
    .stButton > button {
        background: white;
        color: var(--pf-deep-blue);
        border: 1px solid var(--pf-mid-blue);
        border-radius: 8px;
        font-weight: 500;
        padding: 0.4rem 0.9rem;
        transition: all 0.15s ease;
    }
    .stButton > button:hover {
        background: var(--pf-mid-blue);
        color: white;
        border-color: var(--pf-mid-blue);
    }

    /* ---------------------------------------------------------------
       CHAT MESSAGE READABILITY
       Force black text on the white chat bubble regardless of theme.
       Streamlit's dark mode otherwise inherits white text on the
       white bubble background, which makes responses invisible.
       --------------------------------------------------------------- */
    [data-testid="stChatMessage"] {
        background: #FFFFFF !important;
        border-radius: 10px;
        padding: 0.7rem 1rem;
        margin-bottom: 0.6rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        border: 1px solid #E1E8F5;
    }
    [data-testid="stChatMessage"],
    [data-testid="stChatMessage"] *,
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] span,
    [data-testid="stChatMessage"] strong,
    [data-testid="stChatMessage"] em,
    [data-testid="stChatMessage"] td,
    [data-testid="stChatMessage"] th {
        color: #1a1a2e !important;
    }
    [data-testid="stChatMessage"] a {
        color: #1E5BC6 !important;
        text-decoration: underline;
    }
    [data-testid="stChatMessage"] code {
        color: #0033A0 !important;
        background: #F4F7FC !important;
        padding: 0.1rem 0.35rem;
        border-radius: 4px;
    }

    /* ---------------------------------------------------------------
       CHAT INPUT BOX
       Force the input area to white-on-dark-text regardless of theme.
       Without this, dark-mode browsers render the box and its text
       both as dark, making typed input invisible.
       --------------------------------------------------------------- */
    [data-testid="stChatInput"] {
        background: #FFFFFF !important;
    }
    [data-testid="stChatInput"] > div,
    [data-testid="stChatInput"] > div > div {
        background: #FFFFFF !important;
        border-radius: 8px;
    }
    [data-testid="stChatInput"] textarea {
        background: #FFFFFF !important;
        color: #1a1a2e !important;
        caret-color: #1a1a2e !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: #6b7280 !important;
        opacity: 1;
    }
    /* The send-arrow button area inside the input */
    [data-testid="stChatInput"] button {
        background: #FFFFFF !important;
        color: #1E5BC6 !important;
    }

    /* Tighten default Streamlit padding */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)


# =================================================================
# HEADER
# =================================================================

st.markdown("""
<div class="pf-header">
    <h1>🧭 Pathfinder</h1>
    <p>Your personal AI job-search and application assistant, built for Colin Cahill</p>
</div>
""", unsafe_allow_html=True)


# =================================================================
# HELPERS: sidebar metrics
# =================================================================

def compute_metrics(apps):
    """Return dict of pipeline metrics from the raw applications list."""
    if not apps:
        return {
            "total": 0, "this_week": 0,
            "applied": 0, "interviewing": 0, "offer": 0, "rejected": 0,
            "interview_rate": 0.0, "offer_rate": 0.0,
        }

    total = len(apps)
    counts = {"Applied": 0, "Interviewing": 0, "Offer": 0, "Rejected": 0}
    for a in apps:
        if a["status"] in counts:
            counts[a["status"]] += 1

    cutoff = datetime.now().date() - timedelta(days=7)
    this_week = 0
    for a in apps:
        try:
            if datetime.strptime(a["date_applied"], "%Y-%m-%d").date() >= cutoff:
                this_week += 1
        except (ValueError, TypeError):
            continue

    return {
        "total": total,
        "this_week": this_week,
        "applied": counts["Applied"],
        "interviewing": counts["Interviewing"],
        "offer": counts["Offer"],
        "rejected": counts["Rejected"],
        "interview_rate": (counts["Interviewing"] + counts["Offer"]) / total * 100,
        "offer_rate": counts["Offer"] / total * 100,
    }


def metric_card(label, value, sub=""):
    """Render a single sidebar metric card.

    Build the HTML as a single line to avoid Streamlit's markdown renderer
    from leaving stray closing tags visible.
    """
    sub_html = f'<div class="pf-metric-sub">{sub}</div>' if sub else ""
    html = (
        f'<div class="pf-metric-card">'
        f'<div class="pf-metric-label">{label}</div>'
        f'<div class="pf-metric-value">{value}</div>'
        f'{sub_html}'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


# =================================================================
# SIDEBAR: live pipeline tracking
# =================================================================

with st.sidebar:
    st.markdown("### 📊 Pipeline Tracker")
    st.caption("Live from your application database")

    apps = get_pipeline_dataframe()
    m = compute_metrics(apps)

    metric_card("Total applications", m["total"],
                f"{m['this_week']} added this week")
    metric_card("Active interviews", m["interviewing"],
                f"{m['interview_rate']:.0f}% interview rate")
    metric_card("Offers", m["offer"],
                f"{m['offer_rate']:.0f}% offer rate" if m["total"] else "")

    st.markdown("---")
    st.markdown("### 💡 Try asking...")
    st.caption(
        "• Find me business analyst jobs in Chicago  \n"
        "• Show me entry-level finance roles  \n"
        "• Find AI or machine learning jobs for new grads  \n"
        "• Log my Capital One BA application  \n"
        "• I got an interview at Google  \n"
        "• How is my pipeline performing?"
    )

    st.markdown("---")
    if st.button("🔄 Reset conversation", use_container_width=True):
        st.session_state.messages = initialize_messages()
        st.rerun()


# =================================================================
# DASHBOARD EXPANDER: status breakdown + recent apps
# =================================================================

with st.expander("📈 Pipeline Dashboard", expanded=False):
    if not apps:
        st.info("No applications yet. Log one in chat to see your pipeline visualized here.")
    else:
        col1, col2 = st.columns([1, 1.3])

        with col1:
            st.markdown("**Status breakdown**")
            status_data = pd.DataFrame({
                "Status":  ["Applied", "Interviewing", "Offer", "Rejected"],
                "Count":   [m["applied"], m["interviewing"], m["offer"], m["rejected"]],
            })
            st.bar_chart(status_data.set_index("Status"), color="#1E5BC6")

        with col2:
            st.markdown("**Recent applications**")
            df = pd.DataFrame(apps)[["company", "role", "date_applied", "status"]]
            df.columns = ["Company", "Role", "Date Applied", "Status"]
            st.dataframe(df.head(10), use_container_width=True, hide_index=True)


# =================================================================
# QUICK ACTIONS: one-tap suggested prompts
# Broadened to cover the full set of roles Colin is open to:
# analyst, finance, development, AI, systems, BI, and similar
# entry-level friendly roles.
# =================================================================

st.markdown("#### Quick actions")
qa_col1, qa_col2, qa_col3, qa_col4 = st.columns(4)

quick_prompt = None
with qa_col1:
    if st.button("🔎 Business & systems analyst", use_container_width=True):
        quick_prompt = (
            "Find me entry-level business analyst, systems analyst, and business "
            "intelligence roles in Chicago, Kansas City, or remote. Include any "
            "operations or strategy analyst roles that fit a recent grad."
        )
with qa_col2:
    if st.button("💻 Data, AI & development", use_container_width=True):
        quick_prompt = (
            "Find me entry-level data analyst, data science, AI or machine "
            "learning, and junior software development roles. Include remote "
            "and hybrid options open to recent college graduates."
        )
with qa_col3:
    if st.button("💼 Finance & consulting", use_container_width=True):
        quick_prompt = (
            "Find me entry-level finance, financial analyst, fintech, and "
            "analytical consulting roles in Chicago, Kansas City, or remote "
            "that are open to new graduates."
        )
with qa_col4:
    if st.button("📋 My pipeline summary", use_container_width=True):
        quick_prompt = "Show me all of my applications and a summary of how my job search is going."


# =================================================================
# CHAT: main interaction area
# =================================================================

st.markdown("#### Chat with Pathfinder")

# Initialize conversation memory once per session
if "messages" not in st.session_state:
    st.session_state.messages = initialize_messages()

# Show a welcome message if the chat is empty
if not st.session_state.messages:
    with st.chat_message("assistant", avatar="🧭"):
        st.markdown(
            "Hey, I'm Pathfinder. I can search live job listings across analyst, "
            "finance, development, AI, systems, and BI roles, log applications, "
            "track your pipeline, and give you fit analysis based on your resume "
            "and preferences. **What do you want to work on first?**"
        )

# Render chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user", avatar="👤").markdown(msg["content"])
    elif msg["role"] == "assistant":
        st.chat_message("assistant", avatar="🧭").markdown(msg["content"])

# Input: either typed or from a quick-action button
user_input = st.chat_input("Ask Pathfinder anything about your job search...")
if quick_prompt and not user_input:
    user_input = quick_prompt

if user_input:
    st.chat_message("user", avatar="👤").markdown(user_input)

    with st.spinner("Pathfinder is thinking..."):
        try:
            response, updated_messages = get_pathfinder_response(
                st.session_state.messages,
                user_input,
            )
        except Exception as e:
            response = (
                "Sorry, I hit an error while processing that request. "
                f"Details: {str(e)}"
            )
            updated_messages = st.session_state.messages

    st.session_state.messages = updated_messages
    st.chat_message("assistant", avatar="🧭").markdown(response)

    # Refresh sidebar metrics after any tool-using turn
    st.rerun()