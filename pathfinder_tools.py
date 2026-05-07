"""
Pathfinder Tools

Tools the Pathfinder agent uses to search for jobs and manage the
application pipeline. Includes a SQLite-backed tracker and the Adzuna
job-search API integration.
"""

import sqlite3
import os
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
from langchain.tools import tool

# Load secrets. We try Streamlit's secrets store first (used when deployed
# to Streamlit Cloud), and fall back to a local .env file when running on
# our laptop. Same approach as pathfinder_agent.py.
load_dotenv()
try:
    import streamlit as st
    for key in ("OPENAI_API_KEY", "ADZUNA_APP_ID", "ADZUNA_API_KEY"):
        if key in st.secrets:
            os.environ[key] = st.secrets[key]
except Exception:
    pass

DB_PATH = "tools_files/pathfinder.db"

VALID_STATUSES = ["Applied", "Interviewing", "Offer", "Rejected"]


# =============================================
# HELPER: shared DB setup
# =============================================

def _get_connection():
    """Open a connection to the SQLite database, creating the table if needed."""
    os.makedirs("tools_files", exist_ok=True)
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
    # Make sure 'notes' column exists for users who created the DB before this version.
    cols = [r[1] for r in conn.execute("PRAGMA table_info(applications)").fetchall()]
    if "notes" not in cols:
        conn.execute("ALTER TABLE applications ADD COLUMN notes TEXT")
    conn.commit()
    return conn


def _normalize_status(s: str) -> str:
    """Match user-supplied status to canonical capitalization (e.g. 'applied' -> 'Applied')."""
    if not s:
        return ""
    s = s.strip().lower()
    for v in VALID_STATUSES:
        if v.lower() == s:
            return v
    return s.title()


# =============================================
# TOOL 1: Job Search (Adzuna API)
# =============================================

@tool
def search_jobs(keywords: str, location: str = "", results: int = 10):
    """
    Use this tool when the user wants to search for job listings.
    Calls the Adzuna Jobs API and returns current openings, including a short
    description for each so you can evaluate fit against the user's profile.

    Pass meaningful keywords. Good examples for a recent grad include:
    'business analyst', 'data analyst', 'systems analyst',
    'business intelligence analyst', 'financial analyst', 'fintech analyst',
    'junior software developer', 'AI engineer', 'machine learning analyst',
    'operations analyst', 'strategy analyst'. Never call this tool with empty keywords.

    You may call this tool more than once per turn with different keywords to
    cover multiple role families when the user's request is broad.

    Parameters:
    - keywords: job title or skills (e.g. 'business analyst', 'AI engineer')
    - location: city to search in (e.g. 'Chicago'). Leave blank for nationwide.
    - results: number of listings to return (default 10, max 10).
    """
    print(f"[search_jobs] keywords='{keywords}' location='{location}' results={results}")

    app_id = os.getenv("ADZUNA_APP_ID")
    api_key = os.getenv("ADZUNA_API_KEY")

    if not app_id or not api_key:
        return ("Adzuna credentials not found. Add ADZUNA_APP_ID and "
                "ADZUNA_API_KEY to your .env file.")

    if not keywords or not keywords.strip():
        keywords = "business analyst"

    url = "https://api.adzuna.com/v1/api/jobs/us/search/1"
    params = {
        "app_id": app_id,
        "app_key": api_key,
        "results_per_page": min(max(results, 1), 10),
        "what": keywords.strip(),
        "sort_by": "relevance",
        "max_days_old": 30,           # filter out stale listings
        "content-type": "application/json",
    }

    if location.strip():
        params["where"] = location.strip()

    try:
        response = requests.get(url, params=params, timeout=15)
        if response.status_code != 200:
            print(f"[search_jobs] {response.status_code}: {response.text[:300]}")
            return f"Adzuna API returned status {response.status_code}. Check your API credentials."
        data = response.json()
    except requests.exceptions.RequestException as e:
        return f"Error calling Adzuna API: {str(e)}"

    jobs = data.get("results", [])
    if not jobs:
        return (
            f"No listings found for '{keywords}'"
            + (f" in '{location}'" if location else "")
            + ". Try a broader keyword like 'analyst', 'data', 'finance', or "
              "'developer', or remove the location filter to widen the search."
        )

    lines = []
    for i, job in enumerate(jobs, start=1):
        title   = job.get("title", "N/A")
        company = (job.get("company") or {}).get("display_name", "N/A")
        loc     = (job.get("location") or {}).get("display_name", "N/A")
        link    = job.get("redirect_url", "N/A")
        created = job.get("created", "")[:10]  # YYYY-MM-DD

        # Trim and clean the description so the LLM can evaluate fit
        desc = (job.get("description") or "").strip().replace("\n", " ")
        if len(desc) > 350:
            desc = desc[:350].rsplit(" ", 1)[0] + "..."

        sal_min = job.get("salary_min")
        sal_max = job.get("salary_max")
        if sal_min and sal_max:
            salary = f"${sal_min:,.0f} to ${sal_max:,.0f}"
        elif sal_min:
            salary = f"From ${sal_min:,.0f}"
        else:
            salary = "Not listed"

        lines.append(
            f"{i}. **{title}**, {company}\n"
            f"   Location: {loc}  |  Salary: {salary}  |  Posted: {created or 'N/A'}\n"
            f"   Description: {desc or 'Not provided'}\n"
            f"   Apply: {link}\n"
        )

    header = (
        f"Found {len(jobs)} listings for '{keywords}'"
        + (f" in '{location}'" if location else " nationwide")
        + ":\n\n"
    )
    return header + "\n".join(lines)


# =============================================
# TOOL 2: Log Application
# =============================================

@tool
def log_application(company: str, role: str, date_applied: str = "",
                    status: str = "Applied", notes: str = ""):
    """
    Use this tool when the user wants to log a new job application.
    - company: company name (e.g. 'Capital One')
    - role: job title (e.g. 'Business Analyst')
    - date_applied: date in YYYY-MM-DD format. Leave blank or pass 'today' to use today's date.
    - status: one of 'Applied', 'Interviewing', 'Offer', 'Rejected'. Defaults to 'Applied'.
    - notes: optional free-text notes (e.g. referral source, recruiter name).
    """
    # Default to today if missing or "today"
    if not date_applied or date_applied.strip().lower() in ("today", "now", ""):
        date_applied = datetime.now().strftime("%Y-%m-%d")

    # Validate date format
    try:
        datetime.strptime(date_applied, "%Y-%m-%d")
    except ValueError:
        return f"Invalid date '{date_applied}'. Use YYYY-MM-DD format."

    status = _normalize_status(status)
    if status not in VALID_STATUSES:
        return f"Invalid status '{status}'. Use one of: {', '.join(VALID_STATUSES)}"

    conn = _get_connection()
    try:
        # Duplicate guard: same company + role already logged?
        existing = conn.execute(
            "SELECT id, date_applied, status FROM applications "
            "WHERE LOWER(company) = LOWER(?) AND LOWER(role) = LOWER(?)",
            (company.strip(), role.strip())
        ).fetchone()

        if existing:
            return (f"Heads up: {company}, {role} is already in your tracker "
                    f"(logged {existing[1]}, status '{existing[2]}'). "
                    f"If you want to update its status, just say so.")

        conn.execute(
            "INSERT INTO applications (company, role, date_applied, status, notes) "
            "VALUES (?, ?, ?, ?, ?)",
            (company.strip(), role.strip(), date_applied, status, notes.strip() or None)
        )
        conn.commit()
        return (f"Logged it. {company}, {role} is now tracked "
                f"({status}, applied {date_applied}). Good luck!")
    except Exception as e:
        return f"Error logging application: {str(e)}"
    finally:
        conn.close()


# =============================================
# TOOL 3: Update Application Status
# =============================================

@tool
def update_application_status(company: str, new_status: str, role: str = ""):
    """
    Use this tool when the user wants to update the status of an existing application.
    - company: company name (e.g. 'Google'). Partial matches work, so 'Capital One' will
               match 'Capital One Financial'.
    - new_status: one of 'Applied', 'Interviewing', 'Offer', 'Rejected'.
    - role: optional. Use only if the user has multiple applications at the same company.
    """
    new_status = _normalize_status(new_status)
    if new_status not in VALID_STATUSES:
        return f"Invalid status '{new_status}'. Use one of: {', '.join(VALID_STATUSES)}"

    conn = _get_connection()
    try:
        # Find candidate matches with fuzzy LIKE
        if role.strip():
            rows = conn.execute(
                "SELECT id, company, role, status FROM applications "
                "WHERE LOWER(company) LIKE LOWER(?) AND LOWER(role) LIKE LOWER(?) "
                "ORDER BY id DESC",
                (f"%{company.strip()}%", f"%{role.strip()}%")
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, company, role, status FROM applications "
                "WHERE LOWER(company) LIKE LOWER(?) "
                "ORDER BY id DESC",
                (f"%{company.strip()}%",)
            ).fetchall()

        if not rows:
            return (f"No application found matching '{company}'"
                    + (f" / '{role}'" if role else "")
                    + ". Check the company name or log it first.")

        if len(rows) > 1 and not role.strip():
            options = "; ".join(f"{r[1]}, {r[2]}" for r in rows)
            return (f"Found multiple applications for '{company}': {options}. "
                    f"Tell me which role to update.")

        target = rows[0]
        conn.execute(
            "UPDATE applications SET status = ? WHERE id = ?",
            (new_status, target[0])
        )
        conn.commit()
        return f"Updated. {target[1]}, {target[2]} is now marked as '{new_status}'."
    except Exception as e:
        return f"Error updating: {str(e)}"
    finally:
        conn.close()


# =============================================
# TOOL 4: Get Applications
# =============================================

@tool
def get_applications(status_filter: str = "all"):
    """
    Use this tool when the user asks to see their applications or pipeline.
    - status_filter: 'all' to see everything, or filter by 'Applied', 'Interviewing',
                     'Offer', or 'Rejected'.
    """
    conn = _get_connection()
    try:
        if status_filter.lower() == "all":
            rows = conn.execute(
                "SELECT company, role, date_applied, status, notes "
                "FROM applications ORDER BY date_applied DESC"
            ).fetchall()
        else:
            normalized = _normalize_status(status_filter)
            rows = conn.execute(
                "SELECT company, role, date_applied, status, notes "
                "FROM applications WHERE LOWER(status) = LOWER(?) "
                "ORDER BY date_applied DESC",
                (normalized,)
            ).fetchall()

        if not rows:
            if status_filter.lower() == "all":
                return "No applications logged yet. Try asking me to search for jobs first!"
            return f"No applications with status '{status_filter}' found."

        # Render as a markdown table. Streamlit will display it cleanly.
        lines = ["| Company | Role | Date Applied | Status | Notes |",
                 "|---------|------|--------------|--------|-------|"]
        for company, role, date_applied, status, notes in rows:
            notes_cell = (notes or "").replace("|", "\\|") or "-"
            lines.append(f"| {company} | {role} | {date_applied} | {status} | {notes_cell} |")

        return "Here is your application pipeline:\n\n" + "\n".join(lines)
    except Exception as e:
        return f"Error retrieving applications: {str(e)}"
    finally:
        conn.close()


# =============================================
# TOOL 5: Pipeline Stats
# =============================================

@tool
def get_pipeline_stats():
    """
    Use this tool when the user asks about their job-search progress, conversion
    rates, weekly activity, or how their pipeline is performing overall. Returns
    summary metrics: total applications, status breakdown, applications this week,
    interview rate, and offer rate.
    """
    conn = _get_connection()
    try:
        rows = conn.execute(
            "SELECT date_applied, status FROM applications"
        ).fetchall()

        if not rows:
            return "No applications logged yet, so there is nothing to summarize."

        total = len(rows)
        counts = {s: 0 for s in VALID_STATUSES}
        for _, status in rows:
            if status in counts:
                counts[status] += 1

        # Applications in the last 7 days
        cutoff = datetime.now().date() - timedelta(days=7)
        recent = 0
        for date_applied, _ in rows:
            try:
                if datetime.strptime(date_applied, "%Y-%m-%d").date() >= cutoff:
                    recent += 1
            except ValueError:
                continue

        # Interview rate = (Interviewing + Offer) / total
        interview_rate = (counts["Interviewing"] + counts["Offer"]) / total * 100
        offer_rate     = counts["Offer"] / total * 100

        return (
            f"**Pipeline summary**\n"
            f"- Total applications: {total}\n"
            f"- Applied this week: {recent}\n"
            f"- Applied: {counts['Applied']} | Interviewing: {counts['Interviewing']} | "
            f"Offer: {counts['Offer']} | Rejected: {counts['Rejected']}\n"
            f"- Interview rate: {interview_rate:.0f}%\n"
            f"- Offer rate: {offer_rate:.0f}%"
        )
    except Exception as e:
        return f"Error computing stats: {str(e)}"
    finally:
        conn.close()


# =============================================
# Helper used by the Streamlit dashboard (NOT an agent tool)
# =============================================

def get_pipeline_dataframe():
    """
    Returns a list of dicts with all applications. Used by app.py to render
    the sidebar metrics and dashboard chart. Not exposed to the agent.
    """
    conn = _get_connection()
    try:
        rows = conn.execute(
            "SELECT id, company, role, date_applied, status, notes "
            "FROM applications ORDER BY date_applied DESC"
        ).fetchall()
        return [
            {"id": r[0], "company": r[1], "role": r[2],
             "date_applied": r[3], "status": r[4], "notes": r[5]}
            for r in rows
        ]
    finally:
        conn.close()