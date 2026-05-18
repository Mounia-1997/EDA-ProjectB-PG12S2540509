import json
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st


OPENROUTER_MODEL = "openai/gpt-oss-20b:free"
DEFAULT_DATA_PATH = "data/dataset_sample.csv"
DEFAULT_STUDENT_NAME = "Mounia"
DEFAULT_STUDENT_ID = "PG12S2540509"
DEFAULT_TIMESTAMP_COLUMN = "Timestamp "
DEFAULT_TARGET_COLUMN = "MIS Demand (MWh)"

AI_GRADER_PROMPT_TEMPLATE = """# Exact AI Grading Prompt (Hardcode inside app.py)

SYSTEM:
You are a strict academic grader. Return ONLY valid JSON.

USER:
Grade this time-series forecasting Streamlit project OUT OF 80 points using the fixed rubric below.
Be strict: do not award points unless evidence is present in the submitted JSON.
Return ONLY JSON exactly matching the schema.

RUBRIC MAX:
Data & integrity: 20
Feature engineering: 15
Modeling & evaluation: 25
Dashboard quality: 10
Presentation & rigor: 10

STRICT CAPS:
- If the project only uses baseline features/models with no meaningful additions, cap total_80 <= 45.
- If time-based split is missing/unclear, cap Modeling & evaluation <= 12.
- If missing timestamps/outliers/resampling are not discussed or evidenced, cap Data & integrity <= 10.
- If no metrics table is present, cap Modeling & evaluation <= 10.
- If no insights are provided, cap Presentation & rigor <= 5.

Return JSON:
{
  "scores": {
    "Data & integrity": int,
    "Feature engineering": int,
    "Modeling & evaluation": int,
    "Dashboard quality": int,
    "Presentation & rigor": int
  },
  "total_80": int,
  "strengths": [string, ...],
  "weaknesses": [string, ...],
  "actionable_improvements": [string, ...]
}

EVIDENCE JSON:
<insert submission.json contents here>
"""


st.set_page_config(
    page_title="Mini Project B — Time-Series Forecasting Starter",
    page_icon="📈",
    layout="wide",
)


def load_dataset(path_text):
    """Load the local CSV dataset from the repository."""
    path = Path(path_text)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path_text}")
    return pd.read_csv(path)


def missing_percent_table(df):
    """Return missing percentage by column."""
    if df.empty:
        return pd.DataFrame(columns=["column", "missing_percent"])
    return (
        df.isna()
        .mean()
        .mul(100)
        .round(3)
        .rename("missing_percent")
        .reset_index()
        .rename(columns={"index": "column"})
        .sort_values("missing_percent", ascending=False)
    )


def dtype_table(df):
    """Return a compact column dtype table."""
    return pd.DataFrame(
        {
            "column": df.columns,
            "inferred_dtype": [str(dtype) for dtype in df.dtypes],
        }
    )


def numeric_like_columns(df, exclude_column=None, threshold=0.70):
    """Find columns that can mostly be converted to numeric."""
    candidates = []
    for column in df.columns:
        if column == exclude_column:
            continue
        converted = pd.to_numeric(df[column], errors="coerce")
        valid_ratio = float(converted.notna().mean()) if len(converted) else 0.0
        if valid_ratio >= threshold:
            candidates.append(column)
    return candidates


def clean_time_series(df, timestamp_column, target_column):
    """Parse timestamp, convert target to numeric, drop invalid rows, and sort by time."""
    working = df.copy()
    initial_rows = len(working)

    parsed_timestamp = pd.to_datetime(working[timestamp_column], errors="coerce")
    invalid_timestamp_rows = int(parsed_timestamp.isna().sum())
    working[timestamp_column] = parsed_timestamp

    converted_target = pd.to_numeric(working[target_column], errors="coerce")
    invalid_target_rows = int(converted_target.isna().sum())
    working[target_column] = converted_target

    working = (
        working.dropna(subset=[timestamp_column, target_column])
        .sort_values(timestamp_column)
        .reset_index(drop=True)
    )

    duplicate_timestamps = int(working[timestamp_column].duplicated().sum())

    summary = {
        "initial_rows": int(initial_rows),
        "invalid_timestamp_rows": invalid_timestamp_rows,
        "invalid_target_rows": invalid_target_rows,
        "clean_rows": int(len(working)),
        "duplicate_timestamps": duplicate_timestamps,
    }
    return working, summary


def infer_time_coverage(df, timestamp_column):
    """Return min/max timestamp and inferred frequency where possible."""
    if df.empty or timestamp_column not in df.columns:
        return {
            "min_timestamp": None,
            "max_timestamp": None,
            "inferred_frequency": None,
        }

    series = pd.to_datetime(df[timestamp_column], errors="coerce").dropna()
    if series.empty:
        return {
            "min_timestamp": None,
            "max_timestamp": None,
            "inferred_frequency": None,
        }

    ordered = series.sort_values()
    sample_for_frequency = ordered.drop_duplicates().head(10000)
    try:
        inferred_frequency = pd.infer_freq(sample_for_frequency)
    except Exception:
        inferred_frequency = None

    return {
        "min_timestamp": str(ordered.min()),
        "max_timestamp": str(ordered.max()),
        "inferred_frequency": inferred_frequency,
    }


def apply_optional_resampling(df, timestamp_column, target_column, frequency):
    """Optionally resample numeric columns by mean."""
    if frequency is None:
        return df.copy(), False

    working = df.copy()
    numeric_columns = []
    for column in working.columns:
        if column == timestamp_column:
            continue
        converted = pd.to_numeric(working[column], errors="coerce")
        if converted.notna().mean() >= 0.70:
            working[column] = converted
            numeric_columns.append(column)

    if target_column not in numeric_columns:
        numeric_columns.append(target_column)
        working[target_column] = pd.to_numeric(working[target_column], errors="coerce")

    resampled = (
        working.set_index(timestamp_column)[numeric_columns]
        .resample(frequency)
        .mean()
        .dropna(subset=[target_column])
        .reset_index()
    )
    return resampled, True


def create_baseline_features(df, timestamp_column, target_column, horizon):
    """Create baseline forecasting features only. No model training is included."""
    working = df[[timestamp_column, target_column]].copy()
    working = working.sort_values(timestamp_column).reset_index(drop=True)

    working["lag_1"] = working[target_column].shift(1)
    working["lag_24"] = working[target_column].shift(24)
    working["rolling_mean_24"] = working[target_column].shift(1).rolling(window=24).mean()
    working["hour"] = working[timestamp_column].dt.hour
    working["weekend"] = working[timestamp_column].dt.dayofweek.isin([5, 6]).astype(int)
    working["month"] = working[timestamp_column].dt.month
    working["y_target"] = working[target_column].shift(-int(horizon))

    feature_columns = [
        "lag_1",
        "lag_24",
        "rolling_mean_24",
        "hour",
        "weekend",
        "month",
    ]

    feature_table = working.dropna(subset=feature_columns + ["y_target"]).reset_index(drop=True)
    X = feature_table[feature_columns].copy()
    y = feature_table["y_target"].copy()
    return feature_table, X, y, feature_columns


def to_jsonable(value):
    """Convert common scientific Python values to JSON-safe objects."""
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, pd.DataFrame):
        return to_jsonable(value.to_dict(orient="records"))
    if isinstance(value, pd.Series):
        return to_jsonable(value.tolist())
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def parse_grader_response(text):
    """Parse a JSON grader response, with regex fallback for extra text."""
    try:
        return json.loads(text), None
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None, "No JSON object was found in the AI response."
        try:
            return json.loads(match.group(0)), None
        except json.JSONDecodeError as error:
            return None, f"Could not parse JSON object: {error}"


def get_openrouter_api_key():
    """Read OpenRouter key from Streamlit Secrets, environment, or password input."""
    secret_key = None
    try:
        secret_key = st.secrets["OPENROUTER_API_KEY"]
    except Exception:
        secret_key = None

    if secret_key:
        return str(secret_key)

    environment_key = os.getenv("OPENROUTER_API_KEY")
    if environment_key:
        return environment_key

    return st.text_input(
        "OpenRouter API Key",
        type="password",
        help="Used only for this session if no Streamlit Secret or environment variable is available.",
    )


def call_openrouter(api_key, prompt_text, deployed_url, project_title):
    """Call OpenRouter chat completions endpoint."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if deployed_url:
        headers["HTTP-Referer"] = deployed_url
    if project_title:
        headers["X-Title"] = project_title

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt_text,
            }
        ],
        "temperature": 0,
    }

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


def build_submission_json(
    student_name,
    student_id,
    deployed_url,
    project_title,
    project_goal,
    data_path,
    raw_df,
    clean_df,
    feature_table,
    X,
    y,
    timestamp_column,
    target_column,
    cleaning_summary,
    coverage_summary,
    resampling_label,
    horizon,
    feature_columns,
    results_df,
    student_insights,
):
    """Build the project evidence JSON for export and AI grading."""
    has_metrics_table = isinstance(results_df, pd.DataFrame)
    results_table = [] if results_df is None else results_df.to_dict(orient="records")

    evidence = {
        "student": {
            "name": student_name,
            "id": student_id,
        },
        "project": {
            "title": project_title,
            "goal": project_goal,
            "deployed_url": deployed_url,
        },
        "dataset": {
            "path": data_path,
            "raw_shape": [int(raw_df.shape[0]), int(raw_df.shape[1])],
            "clean_shape": [int(clean_df.shape[0]), int(clean_df.shape[1])],
            "columns": list(raw_df.columns),
            "timestamp_column": timestamp_column,
            "target_column": target_column,
            "coverage": coverage_summary,
            "cleaning_summary": cleaning_summary,
            "resampling": resampling_label,
            "forecast_horizon_rows": int(horizon),
        },
        "baseline_feature_table": {
            "baseline_features": feature_columns,
            "feature_table_shape": [int(feature_table.shape[0]), int(feature_table.shape[1])],
            "X_shape": [int(X.shape[0]), int(X.shape[1])],
            "y_length": int(len(y)),
        },
        "student_additions_evidence": {
            "has_metrics_table": has_metrics_table,
            "results_table": results_table,
            "has_extra_dashboard": False,
            "insights": student_insights,
        },
    }
    return to_jsonable(evidence)


def build_project_card(evidence):
    """Create a Markdown project card for submission."""
    return f"""# Mini Project B Project Card

## Student

- Name: {evidence["student"]["name"]}
- Student ID: {evidence["student"]["id"]}

## Project

- Title: {evidence["project"]["title"]}
- Goal: {evidence["project"]["goal"]}
- Streamlit URL: {evidence["project"]["deployed_url"]}

## Dataset

- Dataset path: {evidence["dataset"]["path"]}
- Raw shape: {evidence["dataset"]["raw_shape"]}
- Clean shape: {evidence["dataset"]["clean_shape"]}
- Timestamp column: {evidence["dataset"]["timestamp_column"]}
- Target column: {evidence["dataset"]["target_column"]}
- Time coverage: {evidence["dataset"]["coverage"]}
- Resampling: {evidence["dataset"]["resampling"]}
- Forecast horizon rows: {evidence["dataset"]["forecast_horizon_rows"]}

## Baseline feature table

- Baseline features: {", ".join(evidence["baseline_feature_table"]["baseline_features"])}
- Feature table shape: {evidence["baseline_feature_table"]["feature_table_shape"]}
- X shape: {evidence["baseline_feature_table"]["X_shape"]}
- y length: {evidence["baseline_feature_table"]["y_length"]}

## Student additions evidence

- Has metrics table: {evidence["student_additions_evidence"]["has_metrics_table"]}
- Insights: {evidence["student_additions_evidence"]["insights"]}

## Next improvement checklist

- Add a time-based train/test split.
- Add at least one forecasting model.
- Add a metrics table.
- Add at least one extra dashboard plot or KPI.
- Discuss missing timestamps, outliers, resampling choices, and limitations.
"""


st.title("Mini Project B — Time-Series Forecasting Starter")
st.caption(
    "This starter stops at dataset audit, time-series preparation, baseline features, exports, and AI grading. "
    "Students add modeling, metrics, and dashboard enhancements under the marked placeholders."
)

with st.sidebar:
    st.header("Student info")
    student_name = st.text_input("Student name", value=DEFAULT_STUDENT_NAME)
    student_id = st.text_input("Student ID", value=DEFAULT_STUDENT_ID)
    deployed_url = st.text_input("Deployed Streamlit app URL", value="")
    project_title = st.text_input(
        "Project title",
        value="Energy demand forecasting with weather context",
    )
    project_goal = st.text_area(
        "Project goal",
        value="Prepare a time-series forecasting workflow for MIS demand using timestamp-based features.",
        height=90,
    )
    data_path = st.text_input("Local dataset path", value=DEFAULT_DATA_PATH)

try:
    df_raw = load_dataset(data_path)
except Exception as error:
    st.error(f"Could not load dataset: {error}")
    st.stop()

st.subheader("1. Dataset preview and audit")
st.write("First 10 rows")
st.dataframe(df_raw.head(10), use_container_width=True)

left_col, right_col = st.columns(2)
with left_col:
    st.write("Columns and inferred dtypes")
    st.dataframe(dtype_table(df_raw), use_container_width=True)
with right_col:
    st.write("Missing percentage by column")
    st.dataframe(missing_percent_table(df_raw).head(10), use_container_width=True)

st.subheader("2. Choose timestamp and target columns")
columns = list(df_raw.columns)
default_timestamp_index = columns.index(DEFAULT_TIMESTAMP_COLUMN) if DEFAULT_TIMESTAMP_COLUMN in columns else 0

timestamp_column = st.selectbox(
    "Timestamp column",
    options=columns,
    index=default_timestamp_index,
)

target_options = numeric_like_columns(df_raw, exclude_column=timestamp_column)
if not target_options:
    target_options = [column for column in columns if column != timestamp_column]

default_target_index = (
    target_options.index(DEFAULT_TARGET_COLUMN)
    if DEFAULT_TARGET_COLUMN in target_options
    else 0
)

target_column = st.selectbox(
    "Target column",
    options=target_options,
    index=default_target_index,
)

clean_df, cleaning_summary = clean_time_series(df_raw, timestamp_column, target_column)
coverage_summary = infer_time_coverage(clean_df, timestamp_column)

if clean_df.empty:
    st.error("No valid rows remain after timestamp parsing and target conversion.")
    st.stop()

audit_col_1, audit_col_2, audit_col_3 = st.columns(3)
audit_col_1.metric("Clean rows", f"{len(clean_df):,}")
audit_col_2.metric("Invalid timestamps dropped", cleaning_summary["invalid_timestamp_rows"])
audit_col_3.metric("Invalid target rows dropped", cleaning_summary["invalid_target_rows"])

st.write("Time coverage")
st.json(coverage_summary)

st.subheader("3. Optional resampling and forecast horizon")
resampling_options = {
    "No resampling": None,
    "Hourly mean": "h",
    "Daily mean": "D",
    "Weekly mean": "W",
}
resampling_label = st.selectbox("Resampling option", options=list(resampling_options.keys()), index=0)
resampled_df, used_resampling = apply_optional_resampling(
    clean_df,
    timestamp_column,
    target_column,
    resampling_options[resampling_label],
)

max_horizon = max(1, min(1000, len(resampled_df) - 1))
default_horizon = 24 if max_horizon >= 24 else max_horizon
forecast_horizon = st.number_input(
    "Forecast horizon in rows after optional resampling",
    min_value=1,
    max_value=max_horizon,
    value=default_horizon,
    step=1,
)

st.write("Prepared time-series preview")
st.dataframe(resampled_df.head(10), use_container_width=True)

st.subheader("4. Baseline feature table")
feature_table, X, y, feature_columns = create_baseline_features(
    resampled_df,
    timestamp_column,
    target_column,
    forecast_horizon,
)

if feature_table.empty:
    st.warning(
        "The feature table is empty. Try using more data, a smaller forecast horizon, or no resampling."
    )
else:
    st.write("Baseline features used for X")
    st.write(feature_columns)
    st.write("Feature table preview")
    st.dataframe(feature_table.head(20), use_container_width=True)
    st.write(f"X shape: {X.shape} | y length: {len(y)}")

st.subheader("5. STUDENT ADDITIONS")

st.markdown("### STUDENT ADDITIONS — MODELING")
st.info(
    "Edit app.py and paste your modeling work under this marker. "
    "Create a metrics table named results_df so the export can detect it."
)

# ==============================
# STUDENT ADDITIONS: MODELING
# Paste your forecasting code below this marker.
# Keep results_df as a pandas DataFrame with one row per model and columns such as model, MAE, RMSE, and MAPE.
# Example structure only: create results_df after you build your own model and metrics.
results_df = None
# ==============================

st.code(
    """
# Paste your modeling code under the STUDENT ADDITIONS: MODELING marker.
# Required output for grading evidence:
# results_df = pd.DataFrame([
#     {"model": "Your model name", "MAE": value, "RMSE": value, "MAPE": value}
# ])
""",
    language="python",
)

if isinstance(results_df, pd.DataFrame):
    st.write("Metrics table")
    st.dataframe(results_df, use_container_width=True)
else:
    st.warning("No metrics table yet. Add your modeling code and set results_df to a DataFrame.")

st.markdown("### STUDENT ADDITIONS — DASHBOARD")
st.info(
    "Edit app.py and paste extra plots, KPIs, and written insights under this marker."
)

# ==============================
# STUDENT ADDITIONS: DASHBOARD
# Paste additional dashboard visuals and KPIs below this marker.
# ==============================

st.code(
    """
# Paste extra dashboard code under the STUDENT ADDITIONS: DASHBOARD marker.
# Add at least one plot or KPI and explain what it shows.
""",
    language="python",
)

student_insights = st.text_area(
    "Student insights and limitations",
    value="",
    height=120,
    help="Add data integrity notes, forecasting interpretation, limitations, and next steps.",
)

st.subheader("6. Export submission files")
submission_evidence = build_submission_json(
    student_name=student_name,
    student_id=student_id,
    deployed_url=deployed_url,
    project_title=project_title,
    project_goal=project_goal,
    data_path=data_path,
    raw_df=df_raw,
    clean_df=clean_df,
    feature_table=feature_table,
    X=X,
    y=y,
    timestamp_column=timestamp_column,
    target_column=target_column,
    cleaning_summary=cleaning_summary,
    coverage_summary=coverage_summary,
    resampling_label=resampling_label,
    horizon=forecast_horizon,
    feature_columns=feature_columns,
    results_df=results_df,
    student_insights=student_insights,
)
submission_json_text = json.dumps(submission_evidence, indent=2, ensure_ascii=False)
project_card_text = build_project_card(submission_evidence)

export_col_1, export_col_2 = st.columns(2)
with export_col_1:
    st.download_button(
        "Download submission.json",
        data=submission_json_text,
        file_name="submission.json",
        mime="application/json",
    )
with export_col_2:
    st.download_button(
        "Download project_card.md",
        data=project_card_text,
        file_name="project_card.md",
        mime="text/markdown",
    )

with st.expander("Preview submission.json"):
    st.code(submission_json_text, language="json")

st.subheader("7. AI grader (/80)")
st.write(f"OpenRouter model: `{OPENROUTER_MODEL}`")
api_key = get_openrouter_api_key()
grader_prompt = AI_GRADER_PROMPT_TEMPLATE.replace(
    "<insert submission.json contents here>",
    submission_json_text,
)

with st.expander("Preview AI grader prompt"):
    st.text_area("Prompt sent to AI grader", value=grader_prompt, height=260)

if st.button("Run AI grader"):
    if not api_key:
        st.error("Enter an OpenRouter API key or configure OPENROUTER_API_KEY first.")
    else:
        with st.spinner("Calling AI grader..."):
            try:
                raw_output = call_openrouter(
                    api_key=api_key,
                    prompt_text=grader_prompt,
                    deployed_url=deployed_url,
                    project_title=project_title,
                )
                parsed_output, parse_error = parse_grader_response(raw_output)

                if parsed_output is not None:
                    st.success("AI grader returned valid JSON.")
                    st.json(parsed_output)
                else:
                    st.error(parse_error)
                    st.text_area("Raw AI output", value=raw_output, height=260)
            except Exception as error:
                st.error(f"AI grader request failed: {error}")
