import pandas as pd
import numpy as np
import pytest
from analyzer import (
    detect_dataset_type,
    detect_column_types,
    analyze_numeric,
    analyze_categorical,
    full_analysis
)


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def sales_df():
    return pd.DataFrame({
        "ORDERNUMBER": [1001, 1002, 1003],
        "QUANTITYORDERED": [30, 10, 25],
        "SALES": [5000.0, 2100.0, 4300.0],
        "PRODUCTLINE": ["Motorcycles", "Cars", "Motorcycles"]
    })

@pytest.fixture
def parking_df():
    return pd.DataFrame({
        "lot_id": [1, 2, 3],
        "entry_time": ["2023-01-01 08:00", "2023-01-01 09:30", "2023-01-01 11:00"],
        "exit_time":  ["2023-01-01 10:00", "2023-01-01 11:00", "2023-01-01 13:00"],
        "spaces_available": [10, 5, 0]
    })

@pytest.fixture
def student_df():
    return pd.DataFrame({
        "gender": ["female", "male", "female"],
        "race/ethnicity": ["group B", "group C", "group A"],
        "math score": [72, 69, 90],
        "reading score": [80, 75, 95],
        "writing score": [78, 70, 93]
    })


# ─── Dataset Type Detection ───────────────────────────────────────────────────

def test_detect_sales(sales_df):
    assert detect_dataset_type(sales_df) == "sales"

def test_detect_parking(parking_df):
    assert detect_dataset_type(parking_df) == "parking"

def test_detect_student(student_df):
    assert detect_dataset_type(student_df) == "student"


# ─── Column Type Detection ────────────────────────────────────────────────────

def test_numeric_columns(sales_df):
    numeric, _, _ = detect_column_types(sales_df)
    assert "SALES" in numeric

def test_categorical_columns(sales_df):
    _, categorical, _ = detect_column_types(sales_df)
    assert "PRODUCTLINE" in categorical


# ─── Numeric Analysis ─────────────────────────────────────────────────────────

def test_numeric_analysis_keys(sales_df):
    result = analyze_numeric(sales_df, "SALES")
    for key in ["mean", "median", "std", "min", "max", "outliers", "chart"]:
        assert key in result

def test_numeric_mean(sales_df):
    result = analyze_numeric(sales_df, "SALES")
    expected = round(sales_df["SALES"].mean(), 3)
    assert result["mean"] == expected


# ─── Categorical Analysis ─────────────────────────────────────────────────────

def test_categorical_analysis_keys(sales_df):
    result = analyze_categorical(sales_df, "PRODUCTLINE")
    for key in ["top_category", "top_count", "unique_count", "chart"]:
        assert key in result

def test_categorical_top(sales_df):
    result = analyze_categorical(sales_df, "PRODUCTLINE")
    assert result["top_category"] == "Motorcycles"


# ─── Full Analysis ────────────────────────────────────────────────────────────

def test_full_analysis_sales(sales_df):
    result = full_analysis(sales_df)
    assert result["dataset_type"] == "sales"
    assert "insights" in result
    assert len(result["insights"]) > 0

def test_full_analysis_student(student_df):
    result = full_analysis(student_df)
    assert result["dataset_type"] == "student"
    assert result["shape"]["rows"] == 3

def test_full_analysis_parking(parking_df):
    result = full_analysis(parking_df)
    assert result["dataset_type"] == "parking"