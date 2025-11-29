from nfl_predictor import NFL_Predictor
from nfl_predictions_client import NFLPredictionsClient
import os
from dotenv import load_dotenv
from models.ml_model_packages import CreateMLModelPackageRequest
from models.predictions import CreatePredictionRequest
import pandas as pd

def main():

    predictions = get_model_and_predict_weeks([13], package_id="692a535527a3e8ec90fa1c7f")
    responses = save_predictions_to_database(predictions)
    validate_predictions(13)

def train_model_and_predict_weeks(weeks: list[int], model='random_forest'):
    print(f"Training {model} model and predicting NFL games for weeks {weeks}...")

    nfl_predictor = NFL_Predictor()
    weekly_data, team_dict, schedule_data = nfl_predictor.import_data(2015, 2025)
    dataset = nfl_predictor.create_dataset(weekly_data, schedule_data)

    model_package = nfl_predictor.train_model(dataset=dataset,
                                              model_name=model,
                                              model_target='home_win')
                                              
    
    predictions = predict_weeks(weeks, model_package)

    return model_package, predictions

def get_model_and_predict_weeks(weeks, package_label: str = None, package_id: str = None, date_trained: str = None):
    print(f"Fetching model from database and predicting NFL games for weeks {weeks}...")
    
    if package_label is None and package_id is None and date_trained is None:
        print("You must enter a model label, training date, or id.")
        return
    
    load_dotenv()
    base_url = os.getenv("API_URL")
    api_client = NFLPredictionsClient(base_url=base_url)
    
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
    
    if isinstance(model_package, list):
        model_package = model_package[0]

    predictions = predict_weeks(weeks, model_package)

    return predictions

def predict_weeks(weeks: list[int], model_package: dict) -> list[dict]:
    nfl_predictor = NFL_Predictor()
    weekly_data, _, schedule_data = nfl_predictor.import_data(2015, 2025)

    predictions = []

    for week in weeks:
        week_predictions = nfl_predictor.predict_games(weekly_data=weekly_data,
                                              schedule_data=schedule_data,
                                              season=2025, week=week,
                                              model_package=model_package,
                                              save_to_csv=False)
        
        predictions.extend(week_predictions)

    return predictions

def save_predictions_to_database(predictions: list[dict]):
    print(f"Saving {len(predictions)} predictions to the database...")

    pydantic_predictions = [CreatePredictionRequest(**prediction) for prediction in predictions]

    load_dotenv()
    base_url = os.getenv("API_URL")
    api_client = NFLPredictionsClient(base_url=base_url)

    responses = []

    for prediction in pydantic_predictions:
        try:
            response = api_client.create_prediction(prediction)
            responses.append(response)
        except Exception as e:
            print(f"Error trying to save prediction to database: {e}")

    return responses

def save_model_to_database(model_package: dict):
    print(f"Saving model package labeled: {model_package["package_label"]} to database...")

    load_dotenv()
    base_url = os.getenv("API_URL")
    api_client = NFLPredictionsClient(base_url=base_url)

    try:
        response = api_client.create_model_package(CreateMLModelPackageRequest(**model_package))
    except Exception as e:
        print(f"Error trying to save model package to database: {e}")

    return response

def validate_predictions(week: int, season=2025):
    print(f"Checking correctness of predictions from week {week} of the {season} NFL season...")

    # Load API client
    load_dotenv()
    base_url = os.getenv("API_URL")
    api_client = NFLPredictionsClient(base_url=base_url)

    # Load NFL predictor and schedule data
    nfl_predictor = NFL_Predictor()
    _, _, schedule = nfl_predictor.import_data(2015, 2025)

    # Get predictions for the given week & season
    predictions = api_client.get_all_predictions(season=season, week=week)

    # Check correctness of each prediction and update
    updates = 0
    for prediction in predictions:
        pred_id = prediction["pred_id"]

        pred_season = prediction["season"]
        pred_week = prediction["week"]
        home_team = prediction["home_team"]
        away_team = prediction["away_team"]

        pred_game = schedule[(schedule['season'] == pred_season)
                             & (schedule['week'] == pred_week)
                             & (schedule['home_team'] == home_team)
                             & (schedule['away_team'] == away_team)]

        if pred_game.empty or pred_game.shape[0] > 1:
            continue

        home_score = pred_game['home_score'].values[0]
        away_score = pred_game['away_score'].values[0]

        if pd.isna(home_score) or pd.isna(away_score):
            continue

        outcome = True if home_score > away_score else False
        predicted_outcome = prediction["home_win"]

        prediction["is_correct"] = predicted_outcome == outcome

        new_prediction = CreatePredictionRequest(**prediction)

        response = api_client.update_prediction(pred_id, new_prediction)

        print(f"API response for {home_team} vs. {away_team} prediction: {response}")

        updates += 1

    print(f"{updates} predictions updated for week {week} of the {season} NFL season")
        
    
if __name__ == "__main__":
    main()
