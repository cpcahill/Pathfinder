# Pathfinder

Pathfinder is a personal AI job-search and application assistant built for Colin Cahill, a graduating senior at the University of Kansas studying Business Analytics. It searches live job listings, logs and tracks applications, surfaces pipeline metrics, and gives personalized fit analysis grounded in Colin's resume and preferences.

Built for BSAN 400 (Detelina Stoyanova) using LangChain, OpenAI, FAISS, and Streamlit.

## What Pathfinder can do

- Search live job listings via the Adzuna Jobs API
- Cover a broad set of entry-level roles: business analyst, data analyst, systems analyst, business intelligence, financial analyst, fintech, AI engineer, ML analyst, junior software developer, operations and strategy analyst, analytical consulting
- Log new applications to a SQLite tracker
- Update the status of existing applications (Applied, Interviewing, Offer, Rejected)
- Show the full application pipeline as a table
- Compute pipeline statistics (interview rate, offer rate, applications this week)
- Personalize every response using a FAISS vector store built from Colin's resume, job preferences, and a job-search strategy guide

## Project structure

```
Pathfinder/
├── app.py                    # Streamlit UI
├── pathfinder_agent.py       # LangChain agent logic
├── pathfinder_tools.py       # Five @tool functions: search, log, update, list, stats
├── pathfinder_rag.py         # RAG vector store builder
├── requirements.txt          # Python package list for Streamlit Cloud
├── .gitignore                # Files Git should ignore
├── README.md                 # This file
├── rag_docs/                 # Documents fed to the RAG vector store
│   ├── Resume.pdf
│   ├── job_preferences.txt
│   └── job_search_guide.txt
└── tools_files/              # SQLite database lives here at runtime
```

## Running it locally

1. Clone this repository.
2. Create and activate a virtual environment.
3. Install dependencies: `pip install -r requirements.txt`
4. Create a file called `.env` in the project root with the following keys:
   ```
   OPENAI_API_KEY=your_openai_key_here
   ADZUNA_APP_ID=your_adzuna_app_id_here
   ADZUNA_API_KEY=your_adzuna_api_key_here
   ```
5. Run the app: `streamlit run app.py`

The app will open in your browser at `http://localhost:8501`.

## Deploying to Streamlit Cloud

1. Push this project to a GitHub repository.
2. Go to [streamlit.io](https://streamlit.io), sign in with GitHub, and create a new app pointed at your repo and `app.py`.
3. In the Streamlit app settings, open the **Secrets** tab and paste:
   ```
   OPENAI_API_KEY = "your_openai_key_here"
   ADZUNA_APP_ID = "your_adzuna_app_id_here"
   ADZUNA_API_KEY = "your_adzuna_api_key_here"
   ```
4. Save. The app will redeploy and read its keys from Streamlit's secret store.

## How it works

- The Streamlit app (`app.py`) handles UI, session memory, the sidebar pipeline tracker, and the dashboard expander.
- The agent (`pathfinder_agent.py`) is built once at startup using `create_agent` from LangChain. On every turn, a fresh system message is built with relevant RAG context and prepended to the conversation history.
- The tools (`pathfinder_tools.py`) are five `@tool`-decorated functions that the agent calls when it needs to search jobs or interact with the application tracker.
- The RAG layer (`pathfinder_rag.py`) loads three documents at startup, splits them into chunks, embeds them with OpenAI embeddings, and stores them in a FAISS vector store. The retriever pulls the four most relevant chunks per query.
- A small SQLite database in `tools_files/pathfinder.db` stores the application pipeline. It is created automatically the first time you log an application.

## Built with

- Streamlit
- LangChain
- OpenAI (gpt-4o-mini for the agent, text-embedding for RAG)
- FAISS (vector store)
- Adzuna Jobs API (live job listings)
- SQLite (application tracker)
# Pathfinder
# Pathfinder
# Pathfinder
# Pathfinder
