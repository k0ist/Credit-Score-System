import re
from typing import Optional

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="Credit Score PD API",
    description="Оценка вероятности дефолта (PD) заёмщика на основе CatBoost-модели.",
    version="1.0.0",
)

MODEL_PATH = "model/catboost_model.cbm"

FEATURE_ORDER = [
    "Month",
    "Age",
    "Occupation",
    "Annual_Income",
    "Monthly_Inhand_Salary",
    "Num_Bank_Accounts",
    "Num_Credit_Card",
    "Interest_Rate",
    "Num_of_Loan",
    "Type_of_Loan",
    "Delay_from_due_date",
    "Num_of_Delayed_Payment",
    "Changed_Credit_Limit",
    "Num_Credit_Inquiries",
    "Credit_Mix",
    "Outstanding_Debt",
    "Credit_Utilization_Ratio",
    "Credit_History_Age",
    "Payment_of_Min_Amount",
    "Total_EMI_per_month",
    "Amount_invested_monthly",
    "Payment_Behaviour",
    "Monthly_Balance",
]

HIGH_RISK_THRESHOLD = 0.70

_model: Optional[CatBoostClassifier] = None

MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}


class ClientData(BaseModel):
    # Числовые признаки
    Month: str = Field(..., example="March", description="Название месяца, напр. 'March'")
    Age: float = Field(..., example=32)
    Occupation: str = Field(..., example="Scientist", description="Профессия (категориальный)")
    Annual_Income: float = Field(..., example=45000)
    Monthly_Inhand_Salary: float = Field(..., example=3200)
    Num_Bank_Accounts: int = Field(..., example=3)
    Num_Credit_Card: int = Field(..., example=4)
    Interest_Rate: float = Field(..., example=14)
    Num_of_Loan: int = Field(..., example=2)
    Type_of_Loan: str = Field(..., example="Auto Loan, Personal Loan", description="Типы кредитов (категориальный)")
    Delay_from_due_date: float = Field(..., example=8)
    Num_of_Delayed_Payment: float = Field(..., example=3)
    Changed_Credit_Limit: float = Field(..., example=5.5)
    Num_Credit_Inquiries: float = Field(..., example=2)
    Credit_Mix: str = Field(..., example="Good", description="Микс кредитов (категориальный)")
    Outstanding_Debt: float = Field(..., example=1200.0)
    Credit_Utilization_Ratio: float = Field(..., example=32.1)
    Credit_History_Age: str = Field(
        ..., example="5 Years and 3 Months",
        description="Строка вида '5 Years and 3 Months', конвертируется в месяцы",
    )
    Payment_of_Min_Amount: str = Field(..., example="No", description="Минимальный платёж (категориальный)")
    Total_EMI_per_month: float = Field(..., example=150.0)
    Amount_invested_monthly: float = Field(..., example=100.0)
    Payment_Behaviour: str = Field(..., example="High_spent_Small_value_payments", description="Поведение (категориальный)")
    Monthly_Balance: float = Field(..., example=250.0)


class PredictionResponse(BaseModel):
    pd_score: float
    predicted_class: str
    high_risk_decile: bool


def parse_credit_history_age(value: str) -> float:
    """'5 Years and 3 Months' -> 63.0 (месяцы)."""
    years_match = re.search(r"(\d+)\s*Year", str(value), re.IGNORECASE)
    months_match = re.search(r"(\d+)\s*Month", str(value), re.IGNORECASE)
    years = int(years_match.group(1)) if years_match else 0
    months = int(months_match.group(1)) if months_match else 0
    return float(years * 12 + months)


def prepare_dataframe(data: ClientData) -> pd.DataFrame:
    payload = data.model_dump()

    month_str = str(payload["Month"]).strip().lower()
    if month_str in MONTHS:
        payload["Month"] = MONTHS[month_str]
    elif month_str.isdigit() and 1 <= int(month_str) <= 12:
        payload["Month"] = int(month_str)
    else:
        raise HTTPException(status_code=422, detail=f"Не удалось распознать месяц: '{payload['Month']}'")

    payload["Credit_History_Age"] = parse_credit_history_age(payload["Credit_History_Age"])

    df = pd.DataFrame([payload])
    
    cat_cols = ["Occupation", "Type_of_Loan", "Credit_Mix", "Payment_of_Min_Amount", "Payment_Behaviour"]
    for c in cat_cols:
        df[c] = df[c].astype(str)

    return df[FEATURE_ORDER]


@app.on_event("startup")
def load_model():
    global _model
    _model = CatBoostClassifier()
    _model.load_model(MODEL_PATH)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(data: ClientData):
    if _model is None:
        raise HTTPException(status_code=503, detail="Модель ещё не загружена")

    features_df = prepare_dataframe(data)

    proba = _model.predict_proba(features_df)[0]
    class_names = list(_model.classes_)

    if "Poor" in class_names:
        pd_score = float(proba[class_names.index("Poor")])
    else:
        pd_score = float(np.max(proba))

    predicted_class = str(_model.predict(features_df)[0])

    return PredictionResponse(
        pd_score=round(pd_score, 4),
        predicted_class=predicted_class,
        high_risk_decile=pd_score >= HIGH_RISK_THRESHOLD,
    )