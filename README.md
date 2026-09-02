# Pathfinder

Pathfinder is an AI job-search assistant that **screens and scores** live job
listings against a structured candidate profile, then explains its reasoning.

It searches real postings, filters out the ones that are simply wrong (senior
titles, internships, clearance requirements), scores the survivors from 0 to
100 across seven weighted axes, ranks them, and tracks what you apply to.

Built for BSAN 400 (Detelina Stoyanova) with LangChain, OpenAI, FAISS, and
Streamlit. Live at https://projectpathfinder.streamlit.app/

---

## What makes this different from a chatbot with a job API bolted on

The obvious way to build this is to hand ten raw listings to a language model
and ask which ones look good. That produces sentences, but it is not precise
and it is not reproducible. The same listing can be called a strong fit on one
turn and skipped on the next, and nothing can be audited afterwards.

Pathfinder splits the work along the line where each component is actually
good:

| Job | Owner | Why |
|---|---|---|
| Deciding what fits | `pathfinder_scoring.py` | Deterministic, weighted, auditable |
| Explaining what fits | The language model | Nuance and phrasing |
| Storing what the candidate wants | `profile.yaml` | Structured, complete, editable |
| Retrieving background context | FAISS | Genuinely needs semantic lookup |

The model never reorders the results. It interprets them.

---

## How scoring works

Every posting goes through two stages.

**Stage 1: hard filters.** A posting is dropped outright, with the reason
recorded, if it fails any non-negotiable requirement:

- a senior-signalling title (`senior`, `lead`, `manager`, `principal`, `III`...)
- more years of experience than the profile's cap
- an internship, co-op, or academic post
- a security clearance or polygraph requirement

Screened-out listings are shown in a collapsible panel with the reason for
each, so a filter that is too aggressive is visible rather than silent.

**Stage 2: weighted scoring.** Survivors are scored on seven axes, with the
weights defined in `profile.yaml`:

| Axis | Weight | What it measures |
|---|---|---|
| Role fit | 28 | How closely the title matches a target role family |
| Skills overlap | 24 | Which of the candidate's skills the posting asks for, weighted by how central each skill is |
| Location | 15 | Tiered city preference, with remote treated as near-top |
| Compensation | 12 | Pay against target, **normalized by local cost of living** |
| Industry | 8 | Whether the employer is in a preferred sector |
| Job quality signals | 8 | Green and red flag phrases drawn from the strategy guide |
| Freshness | 5 | How long the posting has been up |

Point adjustments then apply for things that are not axes but still matter:
staffing-agency listings, contract rather than permanent, part time.

Cost-of-living normalization is the axis worth calling out. A $70,000 offer in
Kansas City scores higher than an $85,000 offer in San Francisco, because the
preference sheet explicitly says to weigh pay against what it costs to live
there. Every city in the profile carries a cost index.

Each score is fully decomposed in the UI: expand **Score breakdown** on any
card to see the points each axis contributed and why.

---

## Why the profile is structured data, not a document

The first version of this app stored the candidate's preferences as a written
document and retrieved four 500-character chunks from a vector store on every turn. That
has a subtle failure mode: facts needed on *every* turn were only present when
the user's phrasing happened to retrieve the chunk containing them. Ask about
pay and the model saw the salary range; ask about a role in Denver and it might
not.

Retrieval is the right tool when there is more material than fits in a prompt
and different questions need different parts of it. A one-page preference sheet is
neither. So the split is now deliberate:

- **`profile.yaml`** is structured, complete, injected into every prompt in
  full, and drives the scoring engine.
- **FAISS** holds the resume and the job-search strategy guide, where semantic
  retrieval genuinely helps.

This is also what makes the app portable. Nothing about any particular
candidate is hardcoded in the Python. Swap `profile.yaml` and the entire app
retargets: different skills, different cities, different weights, different
filters.

---

## Project structure

```
ProjectPathfinder/
├── app.py                   # Streamlit layout and interaction
├── pathfinder_ui.py         # Design system: stylesheet and HTML components
├── pathfinder_agent.py      # Model, prompt assembly, conversation
├── pathfinder_tools.py      # Six agent tools: search, score, log, update, list, stats
├── pathfinder_scoring.py    # Hard filters and the seven-axis scoring engine
├── pathfinder_profile.py    # Loads profile.yaml into structured objects
├── pathfinder_rag.py        # FAISS vector store, cached to disk
├── profile.yaml             # The candidate: skills, roles, cities, weights, filters
├── .streamlit/config.toml   # Dark theme base
├── rag_docs/                # Documents embedded into the vector store
│   ├── resume.txt
│   └── job_search_guide.txt
└── tools_files/             # SQLite tracker, created at runtime, gitignored
```

## The tools the agent can call

1. `search_jobs` — fans out across role families, deduplicates, screens, scores, ranks
2. `score_posting` — runs the same engine on a listing pasted in from elsewhere
3. `log_application` — adds to the tracker, including the fit score it was given
4. `update_application_status` — Applied, Interviewing, Offer, Rejected
5. `get_applications` — the pipeline as a table
6. `get_pipeline_stats` — conversion rates, and whether higher-scored applications actually convert better

That last point is the interesting one over time. Because the fit score is
stored alongside the outcome, the tracker can eventually answer whether the
scoring model is any good.

---

## Running it locally

1. Clone the repository.
2. Create and activate a virtual environment.
3. `pip install -r requirements.txt`
4. Create `.env` in the project root:
   ```
   OPENAI_API_KEY=...
   ADZUNA_APP_ID=...
   ADZUNA_API_KEY=...
   ```
5. `streamlit run app.py`

Opens at `http://localhost:8501`.

## Deploying to Streamlit Cloud

Push to GitHub, create an app pointed at `app.py`, and paste the same three
keys into the **Secrets** tab as TOML:

```toml
OPENAI_API_KEY = "..."
ADZUNA_APP_ID  = "..."
ADZUNA_API_KEY = "..."
```

Note that the SQLite tracker lives on the container's local disk, so on
Streamlit Cloud it resets whenever the app redeploys or sleeps. That is fine
for a demo; a hosted Postgres would be the fix for real persistence.

## Tuning the scoring

Everything that determines ranking is in `profile.yaml`, and no code change is
needed to adjust it:

- `weights` — the seven axis weights, which must sum to 100
- `seniority` — title tokens to exclude, and the years-of-experience cap
- `role_families` — what to search for and how much each family is wanted
- `skills` — with `core`, `working`, and `exposure` tiers that scale their value
- `location.tiers` — city preference tiers and cost-of-living indices
- `compensation` — floor, target, and how to treat an unpublished salary
- `signals` — green and red flag phrases, with individual weights

## Built with

Streamlit · LangChain · OpenAI (gpt-4o-mini, text-embedding) · FAISS ·
Adzuna Jobs API · SQLite
