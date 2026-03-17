# RouteIQ (MVP) — AI Study Planner Chatbot

RouteIQ is a lightweight Streamlit MVP for a college AI mini-project that **visibly demonstrates**:
- **Knowledge Representation**: structured `Unit -> topics` + stored topic importance
- **Inference / Reasoning**: rule-based prioritization and explanations
- **Planning**: day-wise study plan generation
- **Expert System**: explicit rule engine that recommends a revision strategy

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## How to use

- **Syllabus** tab:
  - Paste syllabus text or upload a syllabus PDF
  - Optionally select only specific units
  - Optionally upload past papers PDFs
  - Set days (+ optional hours/day), then click **Run Analysis**
- **Analysis** tab:
  - View inferred top topics + structured unit/topic table
  - See **Expert recommendation** (rule-based)
- **Study Plan** tab:
  - Get a day-wise plan prioritizing important topics
- **Materials & Notes** tab:
  - Upload material PDFs (optional)
  - Offline retrieval shows excerpts
  - OpenAI notes generation runs **only** when you click **Generate Notes**
  - If `OPENAI_API_KEY` is missing, the app still works (notes section shows a gentle message)
- **Chatbot** tab:
  - Ask grounded questions like “show units”, “show top topics”, “make a 7 day plan”, “did you use past papers?”, “why is <topic> important?”

## Optional: enable OpenAI notes

Set an environment variable before launching Streamlit:

```bash
export OPENAI_API_KEY="your_key_here"
```

