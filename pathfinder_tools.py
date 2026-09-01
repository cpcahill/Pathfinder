"""
Pathfinder Tools

The functions the agent can call: job search, and the SQLite application tracker.

What changed from the first version
-----------------------------------
search_jobs used to make one API call and hand ten raw listings to the model.
It now fans out across several role families in one turn, deduplicates the
overlap, runs every posting through the scoring engine, drops the ones that
fail hard screening, and returns a ranked shortlist with the reasoning already
attached. The model receives a decision, not a pile of text.

The structured results are also stashed on the Streamlit session so the UI can
render them as cards with score breakdowns rather than as chat markdown.
"""

from __future__ import annotations

import os
import sqlite3
import time
from datetime import datetime, timedelta
from typing import Any

import requests
from dotenv import load_dotenv
from langchain.tools import tool

from pathfinder_profile import get_profile
from pathfinder_scoring import Fit, Job, score_and_rank

# Secrets: Streamlit Cloud's store first, local .env as the fallback.
load_dotenv()
try:
    import streamlit as st
    for _key in ("OPENAI_API_KEY", "ADZUNA_APP_ID", "ADZUNA_API_KEY"):
        if _key in st.secrets:
            os.environ[_key] = st.secrets[_key]
except Exception:
    pass

# Anchor the database to this file rather than the working directory, so the
# app finds the same database no matter where it was launched from.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "tools_files")
DB_PATH = os.path.join(DB_DIR, "pathfinder.db")

VALID_STATUSES = ["Applied", "Interviewing", "Offer", "Rejected"]
ADZUNA_URL = "https://api.adzuna.com/v1/api/jobs/us/search/1"


# =============================================================================
# Result hand-off to the UI
# =============================================================================

def publish_results(fits: list[Fit], rejected: list[Fit], query_note: str) -> None:
    """Make the last search's structured results available to the UI layer.

    Streamlit reruns the whole script on every interaction, so session_state is
    the correct place for this. We fall back to a module global when Streamlit
    is not running, which keeps the tools testable from a plain script.
    """
    payload = {
        "fits": fits,
        "rejected": rejected,
        "note": query_note,
        "at": datetime.now().isoformat(timespec="seconds"),
    }
    global _LAST_RESULTS
    _LAST_RESULTS = payload
    try:
        import streamlit as st
        st.session_state["last_results"] = payload
    except Exception:
        pass


_LAST_RESULTS: dict[str, Any] = {}


def get_last_results() -> dict[str, Any]:
    try:
        import streamlit as st
        return st.session_state.get("last_results", _LAST_RESULTS)
    except Exception:
        return _LAST_RESULTS


# =============================================================================
# Adzuna access, with a small cache
# =============================================================================

_CACHE: dict[tuple, tuple[float, list[Job]]] = {}
_CACHE_TTL = 600          # seconds; job boards do not change minute to minute


def _adzuna_search(query: str, location: str = "", results: int = 20,
                   max_days_old: int = 30) -> list[Job]:
    """One call to Adzuna, normalized into Job objects. Never raises."""
    app_id = os.getenv("ADZUNA_APP_ID")
    api_key = os.getenv("ADZUNA_API_KEY")
    if not app_id or not api_key:
        return []

    key = (query.lower().strip(), location.lower().strip(), results, max_days_old)
    hit = _CACHE.get(key)
    if hit and time.time() - hit[0] < _CACHE_TTL:
        return hit[1]

    params = {
        "app_id": app_id,
        "app_key": api_key,
        "results_per_page": max(1, min(results, 50)),
        "what": query.strip(),
        "sort_by": "relevance",
        "max_days_old": max_days_old,
        "content-type": "application/json",
    }
    if location.strip():
        params["where"] = location.strip()

    try:
        resp = requests.get(ADZUNA_URL, params=params, timeout=20)
        if resp.status_code != 200:
            print(f"[adzuna] {resp.status_code} for '{query}': {resp.text[:200]}")
            return []
        raw = resp.json().get("results", [])
    except (requests.exceptions.RequestException, ValueError) as exc:
        print(f"[adzuna] request failed for '{query}': {exc}")
        return []

    jobs = [Job.from_adzuna(r) for r in raw]
    _CACHE[key] = (time.time(), jobs)
    return jobs


# =============================================================================
# TOOL 1: Job search
# =============================================================================

@tool
def search_jobs(keywords: str = "", location: str = "", limit: int = 8,
                min_score: int = 0) -> str:
    """
    Search live job listings and return them ranked by fit score.

    This tool does the ranking itself. Every listing is screened against the
    candidate's hard requirements (entry level, no clearance, no internships)
    and then scored 0-100 across role fit, skills overlap, location,
    cost-of-living-adjusted pay, industry, job-quality signals, and freshness.
    Trust the ordering it returns. Do not re-rank the results yourself.

    Parameters:
    - keywords: what to search for, for example 'business analyst' or
      'ai engineer'. Leave blank to sweep the candidate's top target role
      families automatically, which is the right choice for open-ended
      requests like "find me jobs that fit me".
    - location: a US city such as 'Chicago' or 'Kansas City'. Leave blank to
      search nationwide, which is usually better since remote roles qualify.
    - limit: how many ranked listings to return. Default 8.
    - min_score: drop anything scoring below this. Use 70 when the user asks
      for only strong matches.

    Returns a ranked shortlist with each listing's score, the reasons behind
    it, any warning flags, and the application link.
    """
    profile = get_profile()

    if keywords and keywords.strip():
        queries = [q.strip() for q in keywords.split(",") if q.strip()]
        sweep_note = f"'{', '.join(queries)}'"
    else:
        queries = profile.search_queries(n_families=5, per_family=1)
        sweep_note = "the top target role families"

    print(f"[search_jobs] queries={queries} location='{location}'")

    collected: list[Job] = []
    for q in queries[:6]:
        collected.extend(_adzuna_search(q, location, results=20))

    if not collected:
        if not os.getenv("ADZUNA_APP_ID"):
            return ("Job search is not configured: ADZUNA_APP_ID and "
                    "ADZUNA_API_KEY are missing from the environment.")
        return (f"No listings came back for {sweep_note}"
                + (f" in {location}" if location else "")
                + ". Try a broader keyword, or drop the location filter so "
                  "remote roles can qualify.")

    kept, rejected = score_and_rank(collected, profile)
    if min_score:
        kept = [f for f in kept if f.score >= min_score]
    shortlist = kept[:max(1, min(limit, 15))]

    where = f" in {location}" if location else " nationwide"
    note = (f"Swept {sweep_note}{where}: {len(collected)} raw listings, "
            f"{len(collected) - len(kept) - len(rejected)} duplicates removed, "
            f"{len(rejected)} screened out, {len(kept)} scored.")
    publish_results(shortlist, rejected, note)

    if not shortlist:
        reasons = _summarize_rejections(rejected)
        return (note + "\n\nNothing cleared the bar. "
                + (f"Most common screen-out reason: {reasons}." if reasons else ""))

    lines = [note, ""]
    for i, fit in enumerate(shortlist, start=1):
        job = fit.job
        flags = f"  [{', '.join(fit.flags)}]" if fit.flags else ""
        lines.append(f"{i}. **{job.title}** at {job.company} ({job.location})")
        lines.append(f"   Fit score {fit.score}/100, rated {fit.band}.{flags}")
        for reason in fit.top_reasons(3):
            lines.append(f"   + {reason}")
        for weak in fit.weak_spots(1):
            lines.append(f"   - {weak}")
        lines.append(f"   Apply: {job.url}")
        lines.append("")

    if rejected:
        lines.append(f"Screened out {len(rejected)} listings. "
                     f"Most common reason: {_summarize_rejections(rejected)}.")
    return "\n".join(lines)


def _summarize_rejections(rejected: list[Fit]) -> str:
    if not rejected:
        return ""
    counts: dict[str, int] = {}
    for fit in rejected:
        head = fit.reject_reason.split(":")[0]
        counts[head] = counts.get(head, 0) + 1
    top = max(counts.items(), key=lambda kv: kv[1])
    return f"{top[0]} ({top[1]})"


# =============================================================================
# TOOL 2: Score a specific posting the user pastes in
# =============================================================================

@tool
def score_posting(title: str, company: str = "", location: str = "",
                  description: str = "", salary: str = "") -> str:
    """
    Score a single job posting the user pastes in or describes, using the same
    engine that ranks search results.

    Use this when the user shares a listing they found elsewhere and asks
    whether it is worth applying to, or how it compares to what they have seen.

    Parameters:
    - title: the job title exactly as posted
    - company: hiring company
    - location: city and state, or 'Remote'
    - description: as much of the posting text as the user provided
    - salary: any salary figure mentioned, for example '80000' or '75k-90k'
    """
    salary_min = None
    digits = "".join(c for c in salary if c.isdigit())
    if digits:
        val = float(digits[:6]) if len(digits) >= 5 else float(digits) * 1000
        salary_min = val

    job = Job(
        title=title, company=company, location=location,
        description=description, salary_min=salary_min,
        created=datetime.now().strftime("%Y-%m-%d"),
    )
    from pathfinder_scoring import score_job
    fit = score_job(job)

    if fit.rejected:
        return (f"**{title}** at {company or 'this company'} would be screened out: "
                f"{fit.reject_reason}. That is a hard filter, so it would not "
                f"appear in search results at all.")

    lines = [f"**{title}** at {company or 'unknown company'}",
             f"Fit score: {fit.score}/100 ({fit.band})", ""]
    for axis in fit.axes:
        lines.append(f"- {axis.label}: {axis.points:.0f} of {axis.weight:.0f} points. {axis.detail}")
    for label, delta in fit.adjustments:
        lines.append(f"- Adjustment: {label} ({delta:+.0f})")
    if fit.missing_skills:
        lines.append("")
        lines.append("Core skills the posting does not mention: "
                     + ", ".join(fit.missing_skills))
    return "\n".join(lines)


# =============================================================================
# Application tracker: shared plumbing
# =============================================================================

def _get_connection() -> sqlite3.Connection:
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS applications (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            company      TEXT NOT NULL,
            role         TEXT NOT NULL,
            date_applied TEXT NOT NULL,
            status       TEXT NOT NULL DEFAULT 'Applied',
            notes        TEXT
        )
    """)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(applications)").fetchall()]
    for col, ddl in (("notes", "TEXT"), ("fit_score", "INTEGER"), ("url", "TEXT")):
        if col not in cols:
            conn.execute(f"ALTER TABLE applications ADD COLUMN {col} {ddl}")
    conn.commit()
    return conn


def _normalize_status(s: str) -> str:
    if not s:
        return ""
    s = s.strip().lower()
    for v in VALID_STATUSES:
        if v.lower() == s:
            return v
    return s.title()


# =============================================================================
# TOOL 3: Log an application
# =============================================================================

@tool
def log_application(company: str, role: str, date_applied: str = "",
                    status: str = "Applied", notes: str = "",
                    fit_score: int = 0, url: str = "") -> str:
    """
    Log a new job application to the tracker.

    - company: company name, for example 'Capital One'
    - role: job title, for example 'Business Analyst'
    - date_applied: YYYY-MM-DD. Blank or 'today' uses today's date.
    - status: 'Applied', 'Interviewing', 'Offer', or 'Rejected'. Defaults to Applied.
    - notes: optional free text, such as a referral source or recruiter name.
    - fit_score: if this role came from a Pathfinder search, pass the fit score
      it was given so the tracker can later compare score against outcome.
    - url: the application link, if known.
    """
    if not date_applied or date_applied.strip().lower() in ("today", "now", ""):
        date_applied = datetime.now().strftime("%Y-%m-%d")
    try:
        datetime.strptime(date_applied, "%Y-%m-%d")
    except ValueError:
        return f"'{date_applied}' is not a valid date. Use YYYY-MM-DD."

    status = _normalize_status(status)
    if status not in VALID_STATUSES:
        return f"Invalid status '{status}'. Use one of: {', '.join(VALID_STATUSES)}"

    conn = _get_connection()
    try:
        existing = conn.execute(
            "SELECT id, date_applied, status FROM applications "
            "WHERE LOWER(company) = LOWER(?) AND LOWER(role) = LOWER(?)",
            (company.strip(), role.strip())
        ).fetchone()
        if existing:
            return (f"{company}, {role} is already tracked (logged {existing[1]}, "
                    f"status '{existing[2]}'). Say the word if you want it updated.")

        conn.execute(
            "INSERT INTO applications (company, role, date_applied, status, notes, fit_score, url) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (company.strip(), role.strip(), date_applied, status,
             notes.strip() or None, int(fit_score) or None, url.strip() or None)
        )
        conn.commit()
        return f"Logged. {company}, {role} is tracked as {status}, applied {date_applied}."
    except Exception as exc:
        return f"Error logging application: {exc}"
    finally:
        conn.close()


# =============================================================================
# TOOL 4: Update status
# =============================================================================

@tool
def update_application_status(company: str, new_status: str, role: str = "") -> str:
    """
    Update the status of an application already in the tracker.

    - company: company name. Partial matches work.
    - new_status: 'Applied', 'Interviewing', 'Offer', or 'Rejected'.
    - role: only needed when several applications share a company.
    """
    new_status = _normalize_status(new_status)
    if new_status not in VALID_STATUSES:
        return f"Invalid status '{new_status}'. Use one of: {', '.join(VALID_STATUSES)}"

    conn = _get_connection()
    try:
        if role.strip():
            rows = conn.execute(
                "SELECT id, company, role, status FROM applications "
                "WHERE LOWER(company) LIKE LOWER(?) AND LOWER(role) LIKE LOWER(?) "
                "ORDER BY id DESC",
                (f"%{company.strip()}%", f"%{role.strip()}%")).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, company, role, status FROM applications "
                "WHERE LOWER(company) LIKE LOWER(?) ORDER BY id DESC",
                (f"%{company.strip()}%",)).fetchall()

        if not rows:
            return (f"No application matching '{company}'"
                    + (f" / '{role}'" if role else "")
                    + " is in the tracker yet.")
        if len(rows) > 1 and not role.strip():
            options = "; ".join(f"{r[1]}, {r[2]}" for r in rows)
            return f"Several applications match '{company}': {options}. Which one?"

        target = rows[0]
        conn.execute("UPDATE applications SET status = ? WHERE id = ?",
                     (new_status, target[0]))
        conn.commit()
        return f"Updated. {target[1]}, {target[2]} is now '{new_status}'."
    except Exception as exc:
        return f"Error updating: {exc}"
    finally:
        conn.close()


# =============================================================================
# TOOL 5: List applications
# =============================================================================

@tool
def get_applications(status_filter: str = "all") -> str:
    """
    Show the application pipeline as a table.

    - status_filter: 'all', or one of 'Applied', 'Interviewing', 'Offer', 'Rejected'.
    """
    conn = _get_connection()
    try:
        if status_filter.lower() == "all":
            rows = conn.execute(
                "SELECT company, role, date_applied, status, notes, fit_score "
                "FROM applications ORDER BY date_applied DESC").fetchall()
        else:
            rows = conn.execute(
                "SELECT company, role, date_applied, status, notes, fit_score "
                "FROM applications WHERE LOWER(status) = LOWER(?) "
                "ORDER BY date_applied DESC",
                (_normalize_status(status_filter),)).fetchall()

        if not rows:
            return ("Nothing logged yet." if status_filter.lower() == "all"
                    else f"No applications with status '{status_filter}'.")

        lines = ["| Company | Role | Applied | Status | Fit | Notes |",
                 "|---|---|---|---|---|---|"]
        for company, role, applied, status, notes, score in rows:
            note_cell = (notes or "").replace("|", "\\|") or "-"
            lines.append(f"| {company} | {role} | {applied} | {status} | "
                         f"{score or '-'} | {note_cell} |")
        return "Application pipeline:\n\n" + "\n".join(lines)
    except Exception as exc:
        return f"Error retrieving applications: {exc}"
    finally:
        conn.close()


# =============================================================================
# TOOL 6: Pipeline stats
# =============================================================================

@tool
def get_pipeline_stats() -> str:
    """
    Summarize how the job search is performing: total applications, status
    breakdown, activity this week, interview rate, offer rate, and whether
    higher-scoring applications are converting better than lower-scoring ones.
    """
    conn = _get_connection()
    try:
        rows = conn.execute(
            "SELECT date_applied, status, fit_score FROM applications").fetchall()
        if not rows:
            return "Nothing logged yet, so there is nothing to summarize."

        total = len(rows)
        counts = {s: 0 for s in VALID_STATUSES}
        for _, status, _ in rows:
            if status in counts:
                counts[status] += 1

        cutoff = datetime.now().date() - timedelta(days=7)
        recent = 0
        for applied, _, _ in rows:
            try:
                if datetime.strptime(applied, "%Y-%m-%d").date() >= cutoff:
                    recent += 1
            except (ValueError, TypeError):
                continue

        advanced = counts["Interviewing"] + counts["Offer"]
        out = [
            "**Pipeline summary**",
            f"- Total applications: {total}",
            f"- Applied in the last 7 days: {recent}",
            f"- Applied {counts['Applied']}, Interviewing {counts['Interviewing']}, "
            f"Offer {counts['Offer']}, Rejected {counts['Rejected']}",
            f"- Interview rate: {advanced / total * 100:.0f}%",
            f"- Offer rate: {counts['Offer'] / total * 100:.0f}%",
        ]

        # Does fit score actually predict outcomes? Only meaningful with data.
        scored = [(score, status) for _, status, score in rows if score]
        if len(scored) >= 5:
            hi = [st for s, st in scored if s >= 75]
            lo = [st for s, st in scored if s < 75]
            if hi and lo:
                hi_rate = sum(1 for s in hi if s in ("Interviewing", "Offer")) / len(hi) * 100
                lo_rate = sum(1 for s in lo if s in ("Interviewing", "Offer")) / len(lo) * 100
                out.append(f"- Roles scored 75+ convert at {hi_rate:.0f}%, "
                           f"roles below 75 at {lo_rate:.0f}%")
        return "\n".join(out)
    except Exception as exc:
        return f"Error computing stats: {exc}"
    finally:
        conn.close()


# =============================================================================
# Helper for the UI (not exposed to the agent)
# =============================================================================

def get_pipeline_dataframe() -> list[dict[str, Any]]:
    """All applications as a list of dicts, for the dashboard."""
    conn = _get_connection()
    try:
        rows = conn.execute(
            "SELECT id, company, role, date_applied, status, notes, fit_score, url "
            "FROM applications ORDER BY date_applied DESC").fetchall()
        return [{"id": r[0], "company": r[1], "role": r[2], "date_applied": r[3],
                 "status": r[4], "notes": r[5], "fit_score": r[6], "url": r[7]}
                for r in rows]
    finally:
        conn.close()


ALL_TOOLS = [
    search_jobs,
    score_posting,
    log_application,
    update_application_status,
    get_applications,
    get_pipeline_stats,
]
