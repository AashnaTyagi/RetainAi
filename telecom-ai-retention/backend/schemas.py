"""Pydantic schemas for the Telecom Retention Intelligence API."""

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CustomerInput(BaseModel):
    """Raw customer attributes, matching the Telco Customer Churn dataset
    schema. Validated at the API boundary so a malformed request fails
    fast with a clear error instead of surfacing as an obscure
    exception deep in feature engineering."""

    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0, le=100, description="Months with the company")
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0, le=500)
    TotalCharges: float = Field(ge=0, le=50000)

    @field_validator("TotalCharges")
    @classmethod
    def total_at_least_monthly(cls, v, info):
        monthly = info.data.get("MonthlyCharges")
        if monthly is not None and v < monthly - 1:
            raise ValueError("TotalCharges should generally be >= MonthlyCharges for tenure >= 1")
        return v

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
            "Dependents": "No", "tenure": 5, "PhoneService": "Yes",
            "MultipleLines": "No", "InternetService": "Fiber optic",
            "OnlineSecurity": "No", "OnlineBackup": "No",
            "DeviceProtection": "No", "TechSupport": "No",
            "StreamingTV": "Yes", "StreamingMovies": "No",
            "Contract": "Month-to-month", "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 85.5, "TotalCharges": 427.5,
        }
    })


class PredictionResponse(BaseModel):
    churn_probability: float
    prediction: Literal["Churn", "No Churn"]
    priority: str


class BatchPredictionRequest(BaseModel):
    customers: list[CustomerInput] = Field(min_length=1, max_length=500)


class BatchPredictionItem(PredictionResponse):
    index: int


class BatchPredictionResponse(BaseModel):
    predictions: list[BatchPredictionItem]
    count: int


class FeatureContribution(BaseModel):
    feature: str
    shap_value: float
    direction: Literal["toward_churn", "toward_staying"]


class ExplainResponse(BaseModel):
    churn_probability: float
    base_value: float
    top_contributions: list[FeatureContribution]
    top_reasons: list[str]
    recommended_actions: list[str]
    priority: str


class MetricsResponse(BaseModel):
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    n_features: int
    n_train_samples: int
    n_test_samples: int


class ModelInfoResponse(BaseModel):
    model_name: str
    model_type: str
    n_features: int
    feature_names: list[str]
    training_strategy: str


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_loaded: bool
