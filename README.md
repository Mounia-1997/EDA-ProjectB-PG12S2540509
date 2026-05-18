# Mini Project B — Time-Series Forecasting Starter

Student: Mounia  
Student ID: PG12S2540509

This repository contains a starter Streamlit app for UTAS Energy Data Analytics Mini Project B. The app is intentionally limited to data loading, auditing, timestamp/target selection, optional resampling, baseline feature-table creation, export files, and the fixed AI grader. Students must add their own modeling, metrics, dashboard improvements, and insights.

## Files

- `app.py` — one-file Streamlit app
- `requirements.txt` — Python dependencies
- `data/dataset_sample.csv` — cleaned and time-sorted dataset sample

## How to run locally

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud deployment

1. Create a public GitHub repository named `EDA-ProjectB-PG12S2540509`.
2. Upload:
   - `app.py`
   - `requirements.txt`
   - `README.md`
   - the full `data` folder containing `dataset_sample.csv`
3. In Streamlit Community Cloud, create a new app.
4. Connect the GitHub repository.
5. Select branch `main`.
6. Set the main file path to `app.py`.
7. Deploy.

## OpenRouter key for AI grading

The app does not hardcode an API key. It checks:

1. Streamlit Secrets: `OPENROUTER_API_KEY`
2. Environment variable: `OPENROUTER_API_KEY`
3. Password input field in the Streamlit app

## What to submit

Submit the following to your instructor:

- Streamlit deployed app URL
- GitHub repository URL
- Exported `submission.json`
- Exported `project_card.md`
- Required screenshots:
  - first 10 rows preview
  - metrics table after you add your model results
  - at least one dashboard plot after you add your dashboard enhancements

## Student next steps

Use the marked `STUDENT ADDITIONS` sections in `app.py` to add:

- your forecasting model or models
- time-based train/test split evidence
- metrics table
- additional dashboard plots or KPIs
- written insights and limitations
