import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import io, base64, re
from scipy import stats


def encode_fig():
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=100)
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def detect_dataset_type(df: pd.DataFrame) -> str:
    cols = " ".join(df.columns.str.lower())
    if any(k in cols for k in ["parking", "entry", "exit", "lot", "space"]):
        return "parking"
    if any(k in cols for k in ["sales", "revenue", "product", "order", "price", "quantity"]):
        return "sales"
    if any(k in cols for k in ["student", "score", "marks", "grade", "math", "reading", "writing", "exam"]):
        return "student"
    return "generic"


def detect_column_types(df: pd.DataFrame):
    numeric, categorical, datetime_cols = [], [], []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric.append(col)
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            datetime_cols.append(col)
        else:
            # Try parsing as datetime
            try:
                pd.to_datetime(df[col], infer_datetime_format=True)
                datetime_cols.append(col)
            except Exception:
                categorical.append(col)
    return numeric, categorical, datetime_cols


def analyze_numeric(df, col):
    series = df[col].dropna()
    z = np.abs(stats.zscore(series))
    outliers = int((z > 3).sum())

    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    axes[0].hist(series, bins=20, color="#4f86f7", edgecolor="white")
    axes[0].set_title(f"Distribution: {col}")
    axes[1].boxplot(series, vert=False, patch_artist=True,
                    boxprops=dict(facecolor="#4f86f7", color="#333"))
    axes[1].set_title(f"Boxplot: {col}")
    chart = encode_fig()

    return {
        "column": col,
        "mean": round(float(series.mean()), 3),
        "median": round(float(series.median()), 3),
        "std": round(float(series.std()), 3),
        "min": round(float(series.min()), 3),
        "max": round(float(series.max()), 3),
        "outliers": outliers,
        "chart": chart
    }


def analyze_categorical(df, col):
    vc = df[col].value_counts().head(10)

    fig, ax = plt.subplots(figsize=(8, 3))
    vc.plot(kind="bar", ax=ax, color="#f4845f", edgecolor="white")
    ax.set_title(f"Top Categories: {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Count")
    chart = encode_fig()

    return {
        "column": col,
        "top_category": str(vc.index[0]),
        "top_count": int(vc.iloc[0]),
        "unique_count": int(df[col].nunique()),
        "frequencies": vc.to_dict(),
        "chart": chart
    }


def generate_insights(df, dataset_type, numeric_cols, categorical_cols, datetime_cols):
    insights = []

    for col in numeric_cols:
        max_val = df[col].max()
        insights.append(f"Highest value in '{col}' is {round(max_val, 2)}")

    for col in categorical_cols:
        top = df[col].value_counts().index[0]
        insights.append(f"Most frequent category in '{col}' is '{top}'")

    for col in datetime_cols:
        try:
            parsed = pd.to_datetime(df[col], infer_datetime_format=True)
            peak = parsed.dt.hour.value_counts().idxmax()
            insights.append(f"Peak activity hour detected in '{col}' at {peak}:00")
        except Exception:
            pass

    # Dataset-specific insights
    if dataset_type == "sales":
        if "QUANTITYORDERED" in df.columns and "SALES" in df.columns:
            top_row = df.loc[df["SALES"].idxmax()]
            insights.append(f"Top revenue row has SALES = {top_row['SALES']}")
    elif dataset_type == "parking":
        insights.append("Parking dataset detected – analyze entry/exit patterns for peak hours.")
    elif dataset_type == "student":
        for col in ["math score", "reading score", "writing score"]:
            if col in df.columns:
                avg = round(df[col].mean(), 2)
                insights.append(f"Average {col}: {avg}")

    return insights


def full_analysis(df: pd.DataFrame):
    dataset_type = detect_dataset_type(df)
    numeric_cols, categorical_cols, datetime_cols = detect_column_types(df)

    numeric_results = [analyze_numeric(df, c) for c in numeric_cols[:5]]
    categorical_results = [analyze_categorical(df, c) for c in categorical_cols[:5]]
    insights = generate_insights(df, dataset_type, numeric_cols, categorical_cols, datetime_cols)

    return {
        "dataset_type": dataset_type,
        "shape": {"rows": df.shape[0], "cols": df.shape[1]},
        "columns": list(df.columns),
        "column_types": {
            "numeric": numeric_cols,
            "categorical": categorical_cols,
            "datetime": datetime_cols
        },
        "numeric_analysis": numeric_results,
        "categorical_analysis": categorical_results,
        "insights": insights
    }