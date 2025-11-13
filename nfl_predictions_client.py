"""
NFL Predictions API Client

A Python client for interacting with the NFL Predictions API.
Supports creating predictions, reading predictions, and updating prediction results.
"""

import requests
from typing import Optional, List, Dict, Any
from datetime import datetime


class NFLPredictionsClient:
    """Client for interacting with the NFL Predictions API."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the NFL Predictions API client.

        Args:
            base_url: Base URL of the API (default: http://localhost:8000)
        """
        self.base_url = base_url.rstrip('/')
        self.predictions_endpoint = f"{self.base_url}/predictions"

    def health_check(self) -> Dict[str, Any]:
        """
        Check if the API is healthy and responsive.

        Returns:
            dict: Health check response
        """
        try:
            response = requests.get(self.base_url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e), "status": "unhealthy"}

    def create_prediction(
        self,
        season: int,
        week: int,
        home_team: str,
        away_team: str,
        home_win: bool,
        confidence: float,
        model_used: str,
        is_correct: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Create a new prediction.

        Args:
            season: NFL season year (1920-2050)
            week: Week number (1-22, where 1-18 are regular season, 19-22 are playoffs)
            home_team: Home team name (1-100 characters)
            away_team: Away team name (1-100 characters, must differ from home_team)
            home_win: True if predicting home team wins, False if away team wins
            confidence: Confidence level (0.0-1.0)
            model_used: Name/identifier of the ML model used (1-100 characters)
            is_correct: Whether prediction was correct (None before game, True/False after)

        Returns:
            dict: Created prediction with pred_id
        """
        payload = {
            "season": season,
            "week": week,
            "home_team": home_team,
            "away_team": away_team,
            "home_win": home_win,
            "confidence": confidence,
            "model_used": model_used,
            "is_correct": is_correct
        }

        try:
            response = requests.post(
                f"{self.predictions_endpoint}/",
                json=payload
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            return {"error": str(e), "status_code": e.response.status_code, "details": e.response.json()}
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}

    def get_all_predictions(
        self,
        season: Optional[int] = None,
        week: Optional[int] = None,
        team: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get all predictions with optional filters.

        Args:
            season: Filter by season year
            week: Filter by week number (requires season to be set)
            team: Filter by team name (matches home or away team)

        Returns:
            list: List of predictions matching the filters
        """
        params = {}
        if season is not None:
            params["season"] = season
        if week is not None:
            params["week"] = week
        if team is not None:
            params["team"] = team

        try:
            response = requests.get(
                f"{self.predictions_endpoint}/",
                params=params
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            return [{"error": str(e), "status_code": e.response.status_code, "details": e.response.json()}]
        except requests.exceptions.RequestException as e:
            return [{"error": str(e)}]

    def get_prediction_by_id(self, prediction_id: str) -> Dict[str, Any]:
        """
        Get a specific prediction by its ID.

        Args:
            prediction_id: The prediction ID (MongoDB ObjectId)

        Returns:
            dict: Prediction details
        """
        try:
            response = requests.get(
                f"{self.predictions_endpoint}/{prediction_id}"
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            return {"error": str(e), "status_code": e.response.status_code, "details": e.response.json()}
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}

    def update_prediction(
        self,
        prediction_id: str,
        season: int,
        week: int,
        home_team: str,
        away_team: str,
        home_win: bool,
        confidence: float,
        model_used: str,
        is_correct: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Update an existing prediction.

        Args:
            prediction_id: The prediction ID to update
            season: NFL season year (1920-2050)
            week: Week number (1-22)
            home_team: Home team name (1-100 characters)
            away_team: Away team name (1-100 characters, must differ from home_team)
            home_win: True if predicting home team wins, False if away team wins
            confidence: Confidence level (0.0-1.0)
            model_used: Name/identifier of the ML model used (1-100 characters)
            is_correct: Whether prediction was correct (None before game, True/False after)

        Returns:
            dict: Updated prediction
        """
        payload = {
            "season": season,
            "week": week,
            "home_team": home_team,
            "away_team": away_team,
            "home_win": home_win,
            "confidence": confidence,
            "model_used": model_used,
            "is_correct": is_correct
        }

        try:
            response = requests.put(
                f"{self.predictions_endpoint}/{prediction_id}",
                json=payload
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            return {"error": str(e), "status_code": e.response.status_code, "details": e.response.json()}
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}

    def update_prediction_result(
        self,
        prediction_id: str,
        is_correct: bool
    ) -> Dict[str, Any]:
        """
        Update only the is_correct field of a prediction (after game concludes).

        Args:
            prediction_id: The prediction ID to update
            is_correct: Whether the prediction was correct

        Returns:
            dict: Updated prediction
        """
        # First, fetch the existing prediction
        prediction = self.get_prediction_by_id(prediction_id)

        if "error" in prediction:
            return prediction

        # Update the is_correct field
        prediction["is_correct"] = is_correct

        # Send the full update
        return self.update_prediction(
            prediction_id=prediction_id,
            season=prediction["season"],
            week=prediction["week"],
            home_team=prediction["home_team"],
            away_team=prediction["away_team"],
            home_win=prediction["home_win"],
            confidence=prediction["confidence"],
            model_used=prediction["model_used"],
            is_correct=is_correct
        )

    def delete_prediction(self, prediction_id: str) -> Dict[str, Any]:
        """
        Delete a specific prediction.

        Args:
            prediction_id: The prediction ID to delete

        Returns:
            dict: Deletion confirmation
        """
        try:
            response = requests.delete(
                f"{self.predictions_endpoint}/{prediction_id}"
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            return {"error": str(e), "status_code": e.response.status_code, "details": e.response.json()}
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}

    def delete_all_predictions(self) -> Dict[str, Any]:
        """
        Delete all predictions (use with caution, intended for testing).

        Returns:
            dict: Deletion confirmation with count
        """
        try:
            response = requests.delete(
                f"{self.predictions_endpoint}/deleteall"
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            return {"error": str(e), "status_code": e.response.status_code, "details": e.response.json()}
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}

    def get_predictions_by_team(self, team_name: str) -> List[Dict[str, Any]]:
        """
        Get all predictions involving a specific team.

        Args:
            team_name: The team name to filter by

        Returns:
            list: List of predictions involving the team
        """
        return self.get_all_predictions(team=team_name)

    def get_predictions_by_week(self, season: int, week: int) -> List[Dict[str, Any]]:
        """
        Get all predictions for a specific week.

        Args:
            season: Season year
            week: Week number

        Returns:
            list: List of predictions for the specified week
        """
        return self.get_all_predictions(season=season, week=week)

    def get_pending_predictions(self) -> List[Dict[str, Any]]:
        """
        Get all predictions that haven't been marked as correct/incorrect yet.

        Returns:
            list: List of predictions with is_correct = None
        """
        all_predictions = self.get_all_predictions()
        return [pred for pred in all_predictions if pred.get("is_correct") is None]

    def get_accuracy_stats(self, season: Optional[int] = None, week: Optional[int] = None) -> Dict[str, Any]:
        """
        Calculate accuracy statistics for predictions.

        Args:
            season: Optional season filter
            week: Optional week filter

        Returns:
            dict: Statistics including accuracy, total predictions, correct, incorrect, pending
        """
        predictions = self.get_all_predictions(season=season, week=week)

        total = len(predictions)
        correct = sum(1 for p in predictions if p.get("is_correct") is True)
        incorrect = sum(1 for p in predictions if p.get("is_correct") is False)
        pending = sum(1 for p in predictions if p.get("is_correct") is None)

        accuracy = (correct / (correct + incorrect) * 100) if (correct + incorrect) > 0 else 0.0

        return {
            "total_predictions": total,
            "correct": correct,
            "incorrect": incorrect,
            "pending": pending,
            "accuracy_percentage": round(accuracy, 2),
            "filters": {
                "season": season,
                "week": week
            }
        }


def main():
    """Example usage of the NFL Predictions Client."""

    # Initialize client (change base_url for production)
    client = NFLPredictionsClient(base_url="http://localhost:8000")

    print("=" * 60)
    print("NFL Predictions API Client - Example Usage")
    print("=" * 60)

    # 1. Health check
    print("\n1. Checking API health...")
    health = client.health_check()
    print(f"Health Status: {health}")

    # 2. Create a new prediction
    print("\n2. Creating a new prediction...")
    new_prediction = client.create_prediction(
        season=2024,
        week=11,
        home_team="Kansas City Chiefs",
        away_team="Buffalo Bills",
        home_win=True,
        confidence=0.72,
        model_used="RandomForest-v2.1",
        is_correct=None  # Game hasn't been played yet
    )
    print(f"Created: {new_prediction}")

    if "pred_id" in new_prediction:
        pred_id = new_prediction["pred_id"]

        # 3. Get prediction by ID
        print(f"\n3. Retrieving prediction by ID: {pred_id}")
        retrieved = client.get_prediction_by_id(pred_id)
        print(f"Retrieved: {retrieved}")

        # 4. Update prediction result after game
        print(f"\n4. Updating prediction result (marking as correct)...")
        updated = client.update_prediction_result(pred_id, is_correct=True)
        print(f"Updated: {updated}")

    # 5. Get all predictions for a specific week
    print("\n5. Getting all predictions for Week 11, 2024...")
    week_predictions = client.get_predictions_by_week(season=2024, week=11)
    print(f"Found {len(week_predictions)} prediction(s)")
    for pred in week_predictions:
        print(f"  - {pred.get('away_team')} @ {pred.get('home_team')}: "
              f"Predicted {'Home' if pred.get('home_win') else 'Away'} win "
              f"(Confidence: {pred.get('confidence'):.2f})")

    # 6. Get all predictions for a specific team
    print("\n6. Getting all predictions involving 'Kansas City Chiefs'...")
    team_predictions = client.get_predictions_by_team("Kansas City Chiefs")
    print(f"Found {len(team_predictions)} prediction(s)")

    # 7. Get pending predictions
    print("\n7. Getting pending predictions (not yet marked)...")
    pending = client.get_pending_predictions()
    print(f"Pending predictions: {len(pending)}")

    # 8. Get accuracy statistics
    print("\n8. Calculating accuracy statistics...")
    stats = client.get_accuracy_stats(season=2024)
    print(f"Accuracy Stats for 2024 Season:")
    print(f"  Total: {stats['total_predictions']}")
    print(f"  Correct: {stats['correct']}")
    print(f"  Incorrect: {stats['incorrect']}")
    print(f"  Pending: {stats['pending']}")
    print(f"  Accuracy: {stats['accuracy_percentage']}%")

    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
