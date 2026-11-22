"""Pydantic request and response schemas for the fraud scoring API."""
from typing import List
from pydantic import BaseModel, ConfigDict, Field

FEATURES = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']

class TransactionFeatures(BaseModel):
    """One transaction represented by the model's 30 numeric input features."""
    model_config = ConfigDict(extra="forbid")

    Time: float = Field(..., description="Seconds elapsed since the first transaction")
    Amount: float = Field(..., ge=0, description="Transaction amount")
    V1: float = Field(..., description="Anonymized transaction feature V1")
    V2: float = Field(..., description="Anonymized transaction feature V2")
    V3: float = Field(..., description="Anonymized transaction feature V3")
    V4: float = Field(..., description="Anonymized transaction feature V4")
    V5: float = Field(..., description="Anonymized transaction feature V5")
    V6: float = Field(..., description="Anonymized transaction feature V6")
    V7: float = Field(..., description="Anonymized transaction feature V7")
    V8: float = Field(..., description="Anonymized transaction feature V8")
    V9: float = Field(..., description="Anonymized transaction feature V9")
    V10: float = Field(..., description="Anonymized transaction feature V10")
    V11: float = Field(..., description="Anonymized transaction feature V11")
    V12: float = Field(..., description="Anonymized transaction feature V12")
    V13: float = Field(..., description="Anonymized transaction feature V13")
    V14: float = Field(..., description="Anonymized transaction feature V14")
    V15: float = Field(..., description="Anonymized transaction feature V15")
    V16: float = Field(..., description="Anonymized transaction feature V16")
    V17: float = Field(..., description="Anonymized transaction feature V17")
    V18: float = Field(..., description="Anonymized transaction feature V18")
    V19: float = Field(..., description="Anonymized transaction feature V19")
    V20: float = Field(..., description="Anonymized transaction feature V20")
    V21: float = Field(..., description="Anonymized transaction feature V21")
    V22: float = Field(..., description="Anonymized transaction feature V22")
    V23: float = Field(..., description="Anonymized transaction feature V23")
    V24: float = Field(..., description="Anonymized transaction feature V24")
    V25: float = Field(..., description="Anonymized transaction feature V25")
    V26: float = Field(..., description="Anonymized transaction feature V26")
    V27: float = Field(..., description="Anonymized transaction feature V27")
    V28: float = Field(..., description="Anonymized transaction feature V28")

    def to_feature_list(self) -> List[float]:
        return [float(getattr(self, name)) for name in FEATURES]

class PredictionResponse(BaseModel):
    fraud_probability: float
    predicted_class: int
    decision: str
    threshold: float
    model_name: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
