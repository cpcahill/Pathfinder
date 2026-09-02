# Pathfinder

An AI job-search assistant that finds live job postings, scores how well each
one fits me, and keeps track of everywhere I have applied.

**Live app:** https://projectpathfinder.streamlit.app/

Built for BSAN 400 at the University of Kansas.

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-app-red)

## What it does

- Searches live job listings through the Adzuna API
- Screens out postings that are clearly wrong for me: senior titles,
  internships, anything needing a security clearance
- Scores everything else from 0 to 100 across seven weighted factors
- Shows the breakdown behind every score, so I can see why a job ranked
  where it did
- Logs applications to a SQLite database and reports my interview and offer
  rates

## How the scoring works

Every posting goes through two steps.

**First, hard filters.** If a posting fails a non-negotiable requirement it is
dropped and the reason is recorded, so I can see what got screened out instead
of wondering where it went.

**Then, weighted scoring.** Whatever survives is scored on seven factors:

| Factor | Weight | What it looks at |
|---|---|---|
| Role fit | 28 | How close the title is to the roles I am targeting |
| Skills overlap | 24 | Which of my skills the posting actually asks for |
| Location | 15 | My city preferences, with remote rated near the top |
| Compensation | 12 | Pay, adjusted for local cost of living |
| Industry | 8 | Whether the company is in a field I want |
| Job quality | 8 | Green and red flags in the posting text |
| Freshness | 5 | How long the posting has been up |

Pay is compared after adjusting for cost of living, so $70,000 in Kansas City
scores better than $85,000 in San Francisco.

Everything about me lives in `profile.yaml`: my skills, target roles, cities,
salary range, and the weights above. Editing that file changes how jobs get
scored without touching any code.

## Tech stack

Python · Streamlit · LangChain · OpenAI · FAISS · SQLite · Adzuna Jobs API

## Running it locally

```bash
git clone https://github.com/cpcahill/Pathfinder.git
cd Pathfinder
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root with your own API keys:

```
OPENAI_API_KEY=your_key_here
ADZUNA_APP_ID=your_id_here
ADZUNA_API_KEY=your_key_here
```

Then run it:

```bash
streamlit run app.py
```

It opens at `http://localhost:8501`.

## Project structure

```
app.py                  Streamlit interface
pathfinder_ui.py        Styling and page components
pathfinder_agent.py     The LangChain agent and its prompt
pathfinder_tools.py     The six tools the agent can call
pathfinder_scoring.py   Hard filters and the seven-factor scoring engine
pathfinder_profile.py   Loads profile.yaml
pathfinder_rag.py       FAISS vector store for my resume and notes
profile.yaml            My skills, roles, cities, salary range, and weights
rag_docs/               Documents the vector store reads
```

## Notes

The application tracker is a SQLite file stored on disk. On Streamlit Cloud
that resets whenever the app redeploys, which is fine for a demo but would
need a hosted database to be permanent.

The resume in `rag_docs/` has my contact details removed, since this
repository is public.
