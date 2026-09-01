"""
Pathfinder: Streamlit app

Layout and interaction only. Ranking lives in pathfinder_scoring.py, the
candidate's data lives in profile.yaml, agent behaviour lives in
pathfinder_agent.py, and every pixel of styling lives in pathfinder_ui.py.

The structural change from the first version: search results are no longer
plain markdown inside a chat bubble. The tools publish structured, scored
records to the session, and this file renders them as cards with an
expandable score breakdown. The chat is for interpretation; the cards are
for the data.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import streamlit as st

import pathfinder_ui as ui

st.set_page_config(
    page_title="Pathfinder",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)
ui.inject_css()


# =============================================================================
# Heavy imports, deferred and cached
# =============================================================================

@st.cache_resource(show_spinner="Starting Pathfinder...")
def boot():
    """Build the agent once per server process, not once per rerun."""
    from pathfinder_agent import get_pathfinder_response, initialize_messages
    from pathfinder_profile import get_profile
    return get_pathfinder_response, initialize_messages, get_profile()


def _startup_failure(exc: Exception) -> None:
    """Explain a failed start in plain language instead of a stack trace.

    The most common cause by far is a missing, revoked, or expired OpenAI key.
    A raw traceback tells the user nothing actionable, and on a deployed app it
    leaks file paths, so this catches the boot and says what to actually do.
    """
    text = f"{type(exc).__name__}: {exc}"
    lowered = text.lower()

    if "authenticationerror" in lowered or "incorrect api key" in lowered or "401" in text:
        headline = "OpenAI rejected the API key"
        body = (
            "The key Pathfinder loaded reached OpenAI, but OpenAI refused it. "
            "That means the key is revoked, expired, deleted, or belongs to a "
            "project that no longer exists. It is not a problem with this code."
        )
        steps = [
            "Create a fresh key at platform.openai.com/api-keys",
            "Replace the OPENAI_API_KEY line in your .env file with it",
            "Confirm the account has billing credit, since a key with no credit still authenticates but fails on use",
            "Restart the app",
        ]
    elif "openai_api_key" in lowered or "api_key" in lowered:
        headline = "No OpenAI API key was found"
        body = (
            "Pathfinder could not find OPENAI_API_KEY. Locally it is read from "
            "a .env file in the project root. On Streamlit Cloud it is read "
            "from the app's Secrets tab."
        )
        steps = [
            "Create a file named .env beside app.py",
            'Add: OPENAI_API_KEY="sk-..."',
            "Add your ADZUNA_APP_ID and ADZUNA_API_KEY the same way",
            "Restart the app",
        ]
    else:
        headline = "Pathfinder could not start"
        body = "The error below happened while building the agent and vector store."
        steps = []

    step_html = ""
    if steps:
        items = "".join(f"<li>{s}</li>" for s in steps)
        step_html = (
            '<ol style="margin:0.9rem 0 0;padding-left:1.15rem;font-size:0.85rem;'
            f'color:var(--text-dim);line-height:1.75">{items}</ol>'
        )

    st.markdown(
        '<div class="pf-panel" style="border-color:#4A2530;background:#160E12">'
        '<div style="font-size:0.66rem;font-weight:600;letter-spacing:0.1em;'
        'text-transform:uppercase;color:#FB7185;margin-bottom:0.5rem">Startup failed</div>'
        f'<div style="font-size:1.05rem;font-weight:600;color:var(--text);'
        f'margin-bottom:0.5rem">{headline}</div>'
        f'<div style="font-size:0.87rem;color:var(--text-dim);line-height:1.65">{body}</div>'
        f'{step_html}</div>',
        unsafe_allow_html=True,
    )
    with st.expander("Technical detail"):
        st.code(text, language="text")
    st.stop()


try:
    get_pathfinder_response, initialize_messages, PROFILE = boot()
except Exception as exc:  # noqa: BLE001 - any failure here is fatal to the app
    _startup_failure(exc)

from pathfinder_tools import get_last_results, get_pipeline_dataframe  # noqa: E402


# =============================================================================
# Metrics
# =============================================================================

def compute_metrics(apps: list[dict]) -> dict:
    counts = {"Applied": 0, "Interviewing": 0, "Offer": 0, "Rejected": 0}
    for app in apps:
        if app["status"] in counts:
            counts[app["status"]] += 1

    total = len(apps)
    cutoff = datetime.now().date() - timedelta(days=7)
    this_week = 0
    for app in apps:
        try:
            if datetime.strptime(app["date_applied"], "%Y-%m-%d").date() >= cutoff:
                this_week += 1
        except (ValueError, TypeError):
            continue

    advanced = counts["Interviewing"] + counts["Offer"]
    scores = [a["fit_score"] for a in apps if a.get("fit_score")]

    return {
        "counts": counts,
        "total": total,
        "this_week": this_week,
        "advanced": advanced,
        "interview_rate": (advanced / total * 100) if total else 0.0,
        "avg_score": (sum(scores) / len(scores)) if scores else None,
    }


def md_safe(text: str) -> str:
    """Escape dollar signs before handing text to Streamlit's markdown.

    Streamlit treats $...$ as LaTeX. A reply containing two salaries, which is
    most of them, gets everything between the first and second dollar sign
    silently re-rendered as maths. Escaping is the only reliable fix.
    """
    return (text or "").replace("$", "\\$")


apps = get_pipeline_dataframe()
m = compute_metrics(apps)


# =============================================================================
# Top bar
# =============================================================================

results = get_last_results()
n_scored = len(results.get("fits", []))

ui.nav([
    (PROFILE.identity.get("name", "Candidate"), False),
    (f"{len(PROFILE.role_families)} target roles", False),
    ("Live listings", True),
])


# =============================================================================
# Stat row
# =============================================================================

avg = f"{m['avg_score']:.0f}" if m["avg_score"] else "--"
ui.stat_row([
    ("Applications", m["total"], f"<b>{m['this_week']}</b> in the last 7 days"),
    ("In play", m["advanced"], f"<b>{m['interview_rate']:.0f}%</b> reached interview"),
    ("Offers", m["counts"]["Offer"],
     "still hunting" if not m["counts"]["Offer"] else "nice work"),
    ("Avg fit applied", avg,
     "of 100, across scored applications" if m["avg_score"] else "log a scored role to track"),
])


# =============================================================================
# Sidebar
# =============================================================================

with st.sidebar:
    ui.eyebrow("Pipeline")
    if apps:
        ui.funnel_bar(m["counts"])
    else:
        ui.empty_state("No applications logged yet.")

    st.markdown("---")
    ui.eyebrow("Run a search")

    if st.button("Sweep all target roles", use_container_width=True):
        st.session_state.pending = (
            "Search across all of my top target role families nationwide and "
            "show me the strongest matches. Explain which two you would apply "
            "to first and why."
        )
    if st.button("Strong matches only", use_container_width=True):
        st.session_state.pending = (
            "Search my target roles and only show me listings scoring 75 or "
            "above. If nothing clears that bar, tell me what came closest and "
            "what held it back."
        )
    if st.button("AI and ML roles", use_container_width=True):
        st.session_state.pending = (
            "Find AI engineer, AI analyst, applied AI, and model evaluation "
            "roles that fit my background building agents and evaluating LLM "
            "output. Remote is welcome."
        )
    if st.button("Kansas City and Chicago", use_container_width=True):
        st.session_state.pending = (
            "Search my target roles in Kansas City and Chicago specifically, "
            "then tell me how the local options compare to what is available "
            "remotely."
        )
    if st.button("How is my search going", use_container_width=True):
        st.session_state.pending = (
            "Show my pipeline stats and give me an honest read on how the "
            "search is performing and what I should change."
        )

    st.markdown("---")
    ui.eyebrow("Session")
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = initialize_messages()
        st.session_state.pop("last_results", None)
        st.rerun()

    st.markdown(
        f'<div style="font-size:0.7rem;color:var(--text-mute);line-height:1.6;'
        f'margin-top:0.8rem">Ranking weights and preferences live in '
        f'<code style="font-size:0.9em">profile.yaml</code>. '
        f'Edit that file to change how jobs are scored.</div>',
        unsafe_allow_html=True,
    )


# =============================================================================
# Main: results and chat, side by side
# =============================================================================

left, right = st.columns([1.15, 1], gap="large")

# --------------------------------------------------------------- left column
with left:
    if results.get("fits"):
        ui.eyebrow(f"Ranked matches  ·  {n_scored} listings")
        st.markdown(
            f'<div style="font-size:0.75rem;color:var(--text-mute);'
            f'margin:-0.2rem 0 0.9rem">{results.get("note", "")}</div>',
            unsafe_allow_html=True,
        )
        for fit in results["fits"]:
            ui.job_card(fit)

        rejected = results.get("rejected", [])
        if rejected:
            with st.expander(f"Screened out ({len(rejected)})"):
                st.markdown(
                    '<div style="font-size:0.76rem;color:var(--text-mute);'
                    'margin-bottom:0.6rem">These never reached scoring. Hard '
                    'filters come from the seniority and hard_filters sections '
                    'of profile.yaml.</div>',
                    unsafe_allow_html=True,
                )
                rows = ""
                for fit in rejected[:40]:
                    rows += (
                        f'<tr><td class="name">{fit.job.title}</td>'
                        f'<td>{fit.job.company}</td>'
                        f'<td style="color:var(--text-mute)">{fit.reject_reason}</td></tr>'
                    )
                st.markdown(
                    '<table class="pf-table"><thead><tr><th>Role</th>'
                    '<th>Company</th><th>Why it was dropped</th></tr></thead>'
                    f'<tbody>{rows}</tbody></table>',
                    unsafe_allow_html=True,
                )
    else:
        ui.eyebrow("Ranked matches")
        ui.empty_state(
            "Run a search and every listing gets screened, scored across seven "
            "weighted axes, and ranked here with its reasoning attached."
        )

    if apps:
        st.markdown("<div style='height:1.6rem'></div>", unsafe_allow_html=True)
        ui.eyebrow("Application tracker")
        ui.pipeline_table(apps, limit=8)
        if len(apps) > 8:
            st.markdown(
                f'<div style="font-size:0.73rem;color:var(--text-mute);'
                f'margin-top:0.6rem">Showing 8 of {len(apps)}. Ask in chat to '
                f'see the full pipeline.</div>',
                unsafe_allow_html=True,
            )

# -------------------------------------------------------------- right column
with right:
    ui.eyebrow("Assistant")

    if "messages" not in st.session_state:
        st.session_state.messages = initialize_messages()

    if not st.session_state.messages:
        st.markdown(
            '<div class="pf-panel">'
            '<div style="font-size:0.9rem;color:var(--text);font-weight:500;'
            'margin-bottom:0.45rem">Ready when you are.</div>'
            '<div style="font-size:0.83rem;color:var(--text-dim);line-height:1.6">'
            'I search live listings, score each one against your profile, and '
            'track what you apply to. Ask me to find roles, paste a posting you '
            'found elsewhere for a second opinion, or log an application.'
            '</div></div>',
            unsafe_allow_html=True,
        )

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).markdown(md_safe(msg["content"]))

    typed = st.chat_input("Ask Pathfinder...")
    pending = st.session_state.pop("pending", None)
    user_input = typed or pending

    if user_input:
        st.chat_message("user").markdown(md_safe(user_input))
        with st.spinner("Searching and scoring..."):
            try:
                reply, updated = get_pathfinder_response(
                    st.session_state.messages, user_input
                )
            except Exception as exc:
                reply = (f"That request hit an error and did not complete.\n\n"
                         f"`{type(exc).__name__}: {exc}`")
                updated = st.session_state.messages
        st.session_state.messages = updated
        st.rerun()