"""
Main application script for NFL game prediction.

This module provides functions for:
- Training ML models and making predictions
- Retrieving models from the database
- Saving predictions to the database
- Validating prediction accuracy after games conclude
"""

from nfl_predictor import NFL_Predictor
from nfl_predictions_client import NFLPredictionsClient
import os
from dotenv import load_dotenv
from models.ml_model_packages import CreateMLModelPackageRequest
from models.predictions import CreatePredictionRequest
import pandas as pd



def main():
    """Main entry point for the application."""
    # Retrieve model from database and generate predictions for week 13
    #predictions = get_model_and_predict_weeks([13], package_id="692a535527a3e8ec90fa1c7f")

    model_package, predictions = train_model_and_predict_weeks([14])

    pred_response = save_predictions_to_database(predictions)
    model_response = save_model_to_database(model_package)

    print(pred_response)
    print(model_response)


def train_model_and_predict_weeks(weeks: list[int], model='random_forest'):
    """
    Train a new ML model and generate predictions for specified weeks.

    Args:
        weeks: List of week numbers to generate predictions for
        model: Type of model to train (default: 'random_forest')

    Returns:
        Tuple of (model_package, predictions)
    """
    print(f"Training {model} model and predicting NFL games for weeks {weeks}...")

    # Initialize predictor and load historical data
    nfl_predictor = NFL_Predictor()
    weekly_data, team_dict, schedule_data = nfl_predictor.import_data(2015, 2025)

    # Create training dataset from historical data
    dataset = nfl_predictor.create_dataset(weekly_data, schedule_data)

    # Train the model on the dataset
    model_package = nfl_predictor.train_model(dataset=dataset,
                                              model_name=model,
                                              model_target='home_win')

    # Generate predictions for the specified weeks
    predictions = predict_weeks(weeks, model_package)

    return model_package, predictions

def get_model_and_predict_weeks(weeks, package_label: str = None, package_id: str = None, date_trained: str = None):
    """
    Retrieve a trained model from the database and generate predictions.

    Args:
        weeks: List of week numbers to generate predictions for
        package_label: Model package label to search for
        package_id: Model package ID to retrieve
        date_trained: Date the model was trained (for filtering)

    Returns:
        List of prediction dictionaries
    """
    print(f"Fetching model from database and predicting NFL games for weeks {weeks}...")

    # Validate that at least one search parameter is provided
    if package_label is None and package_id is None and date_trained is None:
        print("You must enter a model label, training date, or id.")
        return

    # Load environment variables and initialize API client
    load_dotenv()
    base_url = os.getenv("API_URL")
    api_client = NFLPredictionsClient(base_url=base_url)

    # Retrieve model package from database using the provided search criteria
    if package_id is not None:
        try:
            model_package = api_client.get_model_package_by_id(package_id=package_id)
        except Exception as e:
            print(f"Error retrieving model package: {e}")
            return
    elif package_label is not None:
        try:
            model_package = api_client.get_all_model_packages(package_label=package_label)
        except Exception as e:
            print(f"Error retrieving model package: {e}")
            return
    elif date_trained is not None:
        try:
            model_package = api_client.get_all_model_packages(date_trained=date_trained)
        except Exception as e:
            print(f"Error retrieving model package: {e}")
            return
    else:
        print("Error retrieving model package with the given parameters")
        return

    # If multiple packages returned, use the first one
    if isinstance(model_package, list):
        model_package = model_package[0]

    # Generate predictions using the retrieved model
    predictions = predict_weeks(weeks, model_package)

    return predictions

def predict_weeks(weeks: list[int], model_package: dict) -> list[dict]:
    """
    Generate predictions for multiple weeks using a trained model.

    Args:
        weeks: List of week numbers to generate predictions for
        model_package: Dictionary containing the trained model and metadata

    Returns:
        List of prediction dictionaries for all specified weeks
    """
    # Initialize predictor and load current season data
    nfl_predictor = NFL_Predictor()
    weekly_data, _, schedule_data = nfl_predictor.import_data(2015, 2025)

    predictions = []

    # Generate predictions for each week
    for week in weeks:
        week_predictions = nfl_predictor.predict_games(weekly_data=weekly_data,
                                              schedule_data=schedule_data,
                                              season=2025, week=week,
                                              model_package=model_package,
                                              save_to_csv=False)

        predictions.extend(week_predictions)

    return predictions

def save_predictions_to_database(predictions: list[dict]):
    """
    Save a list of predictions to the database via the API.

    Args:
        predictions: List of prediction dictionaries to save

    Returns:
        List of API responses for each prediction
    """
    print(f"Saving {len(predictions)} predictions to the database...")

    # Convert raw dictionaries to Pydantic models for validation
    pydantic_predictions = [CreatePredictionRequest(**prediction) for prediction in predictions]

    # Load environment variables and initialize API client
    load_dotenv()
    base_url = os.getenv("API_URL")
    api_client = NFLPredictionsClient(base_url=base_url)

    responses = []

    # Save each prediction to the database
    for prediction in pydantic_predictions:
        try:
            response = api_client.create_prediction(prediction)
            responses.append(response)
        except Exception as e:
            print(f"Error trying to save prediction to database: {e}")

    return responses

def save_model_to_database(model_package: dict):
    """
    Save a trained model package to the database via the API.

    Args:
        model_package: Dictionary containing the model and metadata

    Returns:
        API response from the create operation
    """
    print(f"Saving model package labeled: {model_package["package_label"]} to database...")

    # Load environment variables and initialize API client
    load_dotenv()
    base_url = os.getenv("API_URL")
    api_client = NFLPredictionsClient(base_url=base_url)

    # Save model package to database
    try:
        response = api_client.create_model_package(CreateMLModelPackageRequest(**model_package))
    except Exception as e:
        print(f"Error trying to save model package to database: {e}")

    return response

def validate_predictions(week: int, season=2025):
    """
    Validate predictions against actual game results and update the database.

    This function retrieves predictions for a specific week, compares them to
    actual game results, and updates the 'is_correct' field for each prediction.

    Args:
        week: Week number to validate predictions for
        season: NFL season year (default: 2025)
    """
    print(f"Checking correctness of predictions from week {week} of the {season} NFL season...")

    # Load environment variables and initialize API client
    load_dotenv()
    base_url = os.getenv("API_URL")
    api_client = NFLPredictionsClient(base_url=base_url)

    # Load NFL predictor and schedule data to get actual game results
    nfl_predictor = NFL_Predictor()
    _, _, schedule = nfl_predictor.import_data(2015, 2025)

    # Retrieve all predictions for the specified week and season
    predictions = api_client.get_all_predictions(season=season, week=week)

    # Iterate through each prediction and check correctness
    updates = 0
    for prediction in predictions:
        pred_id = prediction["pred_id"]

        # Extract prediction details
        pred_season = prediction["season"]
        pred_week = prediction["week"]
        home_team = prediction["home_team"]
        away_team = prediction["away_team"]

        # Find the corresponding game in the schedule
        pred_game = schedule[(schedule['season'] == pred_season)
                             & (schedule['week'] == pred_week)
                             & (schedule['home_team'] == home_team)
                             & (schedule['away_team'] == away_team)]

        # Skip if game not found or multiple games found (shouldn't happen)
        if pred_game.empty or pred_game.shape[0] > 1:
            continue

        # Extract actual game scores
        home_score = pred_game['home_score'].values[0]
        away_score = pred_game['away_score'].values[0]

        # Skip if game hasn't been played yet (scores are NaN)
        if pd.isna(home_score) or pd.isna(away_score):
            continue

        # Determine actual outcome and compare with prediction
        outcome = True if home_score > away_score else False
        predicted_outcome = prediction["home_win"]

        # Update the prediction's correctness
        prediction["is_correct"] = predicted_outcome == outcome

        # Create Pydantic model and update in database
        new_prediction = CreatePredictionRequest(**prediction)
        response = api_client.update_prediction(pred_id, new_prediction)

        print(f"API response for {home_team} vs. {away_team} prediction: {response}")

        updates += 1

    print(f"{updates} predictions updated for week {week} of the {season} NFL season")
        
    
if __name__ == "__main__":
    main()
