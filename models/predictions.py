from pydantic import BaseModel, Field, model_validator
from typing import Optional

class Prediction(BaseModel):
    pred_id: str = Field(..., description="MongoDB document ID", alias='_id')
    season: int = Field(
        ...,
        ge=1920,
        le=2050,
        description="NFL season year (1920-2050)",
        examples=[2024]
    )
    week: int = Field(
        ...,
        ge=1,
        le=22,
        description="Week number (1-18 regular season, 19-22 playoffs)",
        examples=[10]
    )
    home_team: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Home team name",
        examples=["BAL"]
    )
    away_team: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Away team name",
        examples=["DEN"]
    )
    home_win: bool = Field(
        ...,
        description="Predicted winner (True if home team wins)",
        examples=[True]
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence level (0.0 to 1.0)",
        examples=[0.85]
    )
    model_used: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Name of the prediction model",
        examples=["RandomForest-v1"]
    )
    is_correct: Optional[bool] = Field(
        default=None,
        description="Whether prediction was correct (None if game not concluded)",
        examples=[True, None]
    )

    @model_validator(mode='after')
    def validate_teams_different(self):
        """Ensure home and away teams are different"""
        if self.home_team == self.away_team:
            raise ValueError("Home team and away team must be different")
        return self

class CreatePredictionRequest(BaseModel):
    season: int = Field(
        ...,
        ge=1920,
        le=2050,
        description="NFL season year (1920-2050)",
        examples=[2024]
    )
    week: int = Field(
        ...,
        ge=1,
        le=22,
        description="Week number (1-18 regular season, 19-22 playoffs)",
        examples=[10]
    )
    home_team: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Home team name",
        examples=["Kansas City Chiefs"]
    )
    away_team: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Away team name",
        examples=["Denver Broncos"]
    )
    home_win: bool = Field(
        ...,
        description="Predicted winner (True if home team wins)",
        examples=[True]
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence level (0.0 to 1.0)",
        examples=[0.85]
    )
    model_used: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Name of the prediction model",
        examples=["RandomForest-v1"]
    )
    is_correct: Optional[bool] = Field(
        default=None,
        description="Whether prediction was correct (None if game not concluded)",
        examples=[True, None]
    )

    @model_validator(mode='after')
    def validate_teams_different(self):
        """Ensure home and away teams are different"""
        if self.home_team == self.away_team:
            raise ValueError("Home team and away team must be different")
        return self

