from pydantic import BaseModel, Field
from typing import Optional, Any, List

class MLModelPackage(BaseModel):
    package_id: str = Field(..., description="MongoDB document ID", alias="_id")
    package_label: str = Field(..., description="ML model package label")
    model: str = Field(..., description="Trained ML model (stored as base64 encoded string)")
    model_features: List[str] = Field(..., description="List of features the model expects")
    model_scores: dict[str, Any] = Field(..., description="Dict of model scores")
    dataset: List[dict] = Field(..., description="Dataset the model was trained on")
    model_target: str = Field(..., description="Target column of the model")
    date_trained: str = Field(..., description="Date the model was trained")

class CreateMLModelPackageRequest(BaseModel):
    package_label: str = Field(..., description="ML model package label")
    model: str = Field(..., description="Trained ML model (stored as base64 encoded string)")
    model_features: List[str] = Field(..., description="List of features the model expects")
    model_scores: dict[str, Any] = Field(..., description="Dict of model scores")
    dataset: List[dict] = Field(..., description="Dataset the model was trained on")
    model_target: str = Field(..., description="Target column of the model")
    date_trained: str = Field(..., description="Date the model was trained")