"""
NFL Game Predictor using Machine Learning.

This module provides the NFL_Predictor class which handles:
- Importing NFL historical data
- Creating feature sets from team statistics
- Training various ML models (Random Forest, Gradient Boosting, XGBoost, etc.)
- Making predictions for upcoming NFL games
- Encoding/decoding models for storage
"""

import nfl_data_py as nfl
import nflreadpy as nflr
import pandas as pd
import numpy as np
import datetime

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Sklearn imports
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

# XGBoost import
import xgboost as xgb

class NFL_Predictor:
    """
    Main class for NFL game prediction using machine learning.

    This class provides methods for importing NFL data, creating features,
    training models, and generating predictions for NFL games.
    """
    def __init__(self, current_season=2025):
        """
        Initialize the NFL Predictor.

        Args:
            current_season: The current NFL season year (default: 2025)
        """
        self.available_models = ['random_forest', 'gradient_boosting', 'decision_tree', 'logistic_regression', 'xgboost']
        self.current_season = current_season

    def import_data(self, start_year=2014, end_year=2025):
        """
        Import NFL data including team stats, descriptions, and schedules.

        Args:
            start_year: First year to import data from (default: 2014)
            end_year: Last year to import data from (default: 2025)

        Returns:
            Tuple of (weekly_data, team_dict, schedule_data)
        """
        if end_year > self.current_season:
            end_year = self.current_season

        # list of years to import data from
        years = range(start_year, end_year + 1)

        # import weekly team stats
        try:
            weekly_data = nflr.load_team_stats(years).to_pandas()
            print(f"Team stats for {start_year}-{end_year} loaded successfully.")
        except Exception as e:
            print(f"Error loading team stats: {e}")
            # if error, try reduced year range
            years = range(start_year, end_year)
            weekly_data = nflr.load_team_stats(years).to_pandas()
            print(f"Team stats for {start_year}-{end_year-1} loaded successfully.")

        # import team data
        try:
            team_desc = nfl.import_team_desc()
            team_dict = {row['team_abbr']: row['team_name'] for index, row in team_desc.iterrows()}
            print("Team descriptions loaded successfully.")
        except Exception as e:
            print(f"Error loading team descriptions: {e}") 

        # import schedules data
        try:
            schedule_data = nfl.import_schedules(years)
            print(f"Schedules for {start_year}-{end_year} loaded successfully.")
        except Exception as e:
            print(f"Error loading schedules: {e}")
            # if error, try reduced year range
            years = range(start_year, end_year)
            schedule_data = nfl.import_schedules(years)
            print(f"Schedules for {start_year}-{end_year-1} loaded successfully.")

        return weekly_data, team_dict, schedule_data

    def create_team_features(self,
                             weekly_data,
                             season,
                             week):
        """
        Create aggregated feature statistics for each team up to a given week.

        This calculates per-game averages for offensive, defensive, and special
        teams statistics for each team based on games played before the specified week.

        Args:
            weekly_data: DataFrame containing weekly team statistics
            season: NFL season year
            week: Week number (features created from weeks before this)

        Returns:
            Dictionary mapping team names to their feature dictionaries
        """
        
        season_data = weekly_data[(weekly_data['season'] == season) & (weekly_data['week'] < week)]

        team_features = {}

        for team in season_data['team'].unique():
            team_stats = season_data[season_data['team'] == team]
            opponent_stats = season_data[season_data['opponent_team'] == team]

            features = {
                # Passing/QB Stats
                "completion_pct": team_stats['completions'].mean() / team_stats['attempts'].mean(),
                "passing_yds_pg": team_stats['passing_yards'].mean(),
                "passing_tds_pg": team_stats['passing_tds'].mean(),
                "passing_cpoe": team_stats['passing_cpoe'].mean(),
                "sacks_pg": team_stats['sacks_suffered'].mean(),
                # Rushing Stats
                "carries_pg": team_stats['carries'].mean(),
                "rushing_yds_pg": team_stats['rushing_yards'].mean(),
                "rushing_tds_pg": team_stats['rushing_tds'].mean(),
                "rushing_epa_pg": team_stats['rushing_epa'].mean(),
                # Receiving Stats
                "targets_pg": team_stats['targets'].mean(),
                "receiving_yds_pg": team_stats['receiving_yards'].mean(),
                "receiving_tds_pg": team_stats['receiving_tds'].mean(),
                "receiving_epa_pg": team_stats['receiving_epa'].mean(),
                # Turnovers
                "turnovers_pg": team_stats['passing_interceptions'].mean() + team_stats['sack_fumbles_lost'].mean() + team_stats['rushing_fumbles_lost'].mean() + team_stats['receiving_fumbles_lost'].mean(),
                # Misc Stats
                "penalty_yds_pg": team_stats['penalty_yards'].mean(),
                "dst_tds_pg": team_stats['def_tds'].mean() + team_stats['special_teams_tds'].mean(),
                # Defensive Stats
                "passing_yds_allowed_pg": opponent_stats['passing_yards'].mean(),
                "passing_tds_allowed_pg": opponent_stats['passing_tds'].mean(),
                "rushing_yds_allowed_pg": opponent_stats['rushing_yards'].mean(),
                "rushing_tds_allowed_pg": opponent_stats['rushing_tds'].mean(),
                "receiving_yds_allowed_pg": opponent_stats['receiving_yards'].mean(),
                "receiving_tds_allowed_pg": opponent_stats['receiving_tds'].mean(),
            }

            team_features[team] = features

        return team_features

    def create_game_features(self,
                             team_features,
                             home_team, away_team,
                             season, week,
                             is_playoff=False, is_neutral=False):
        """
        Create feature vector for a specific game matchup.

        Combines team-level features for both home and away teams along with
        game context information (playoff status, neutral site, etc.).

        Args:
            team_features: Dictionary of team features from create_team_features()
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            season: NFL season year
            week: Week number
            is_playoff: Whether this is a playoff game (default: False)
            is_neutral: Whether played at neutral site (default: False)

        Returns:
            Dictionary of features for this specific game matchup
        """
        
        if home_team not in team_features or away_team not in team_features:
            return None

        home_stats = team_features[home_team]
        away_stats = team_features[away_team]

        game_features = {
            # Game Info
            "season": season,
            "week": week,
            "is_playoff": 1 if is_playoff else 0,
            "is_neutral": 1 if is_neutral else 0,
            # Home Team Stats
            "home_completion_pct": home_stats["completion_pct"],
            "home_passing_yds_pg": home_stats["passing_yds_pg"],
            "home_passing_tds_pg": home_stats["passing_tds_pg"],
            "home_passing_cpoe": home_stats["passing_cpoe"],
            "home_sacks_pg": home_stats["sacks_pg"],
            "home_carries_pg": home_stats["carries_pg"],
            "home_rushing_yds_pg": home_stats["rushing_yds_pg"],
            "home_rushing_tds_pg": home_stats["rushing_tds_pg"],
            "home_rushing_epa_pg": home_stats["rushing_epa_pg"],
            "home_targets_pg": home_stats["targets_pg"],
            "home_receiving_yds_pg": home_stats["receiving_yds_pg"],
            "home_receiving_tds_pg": home_stats["receiving_tds_pg"],
            "home_receiving_epa_pg": home_stats["receiving_epa_pg"],
            "home_turnovers_pg": home_stats["turnovers_pg"],
            "home_penalty_yds_pg": home_stats["penalty_yds_pg"],
            "home_dst_tds_pg": home_stats["dst_tds_pg"],
            "home_passing_yds_allowed_pg": home_stats["passing_yds_allowed_pg"],
            "home_passing_tds_allowed_pg": home_stats["passing_tds_allowed_pg"],
            "home_rushing_yds_allowed_pg": home_stats["rushing_yds_allowed_pg"],
            "home_rushing_tds_allowed_pg": home_stats["rushing_tds_allowed_pg"],
            "home_receiving_yds_allowed_pg": home_stats["receiving_yds_allowed_pg"],
            "home_receiving_tds_allowed_pg": home_stats["receiving_tds_allowed_pg"],
            # Away Team Stats
            "away_completion_pct": away_stats["completion_pct"],
            "away_passing_yds_pg": away_stats["passing_yds_pg"],
            "away_passing_tds_pg": away_stats["passing_tds_pg"],
            "away_passing_cpoe": away_stats["passing_cpoe"],
            "away_sacks_pg": away_stats["sacks_pg"],
            "away_carries_pg": away_stats["carries_pg"],
            "away_rushing_yds_pg": away_stats["rushing_yds_pg"],
            "away_rushing_tds_pg": away_stats["rushing_tds_pg"],
            "away_rushing_epa_pg": away_stats["rushing_epa_pg"],
            "away_targets_pg": away_stats["targets_pg"],
            "away_receiving_yds_pg": away_stats["receiving_yds_pg"],
            "away_receiving_tds_pg": away_stats["receiving_tds_pg"],
            "away_receiving_epa_pg": away_stats["receiving_epa_pg"],
            "away_turnovers_pg": away_stats["turnovers_pg"],
            "away_penalty_yds_pg": away_stats["penalty_yds_pg"],
            "away_dst_tds_pg": away_stats["dst_tds_pg"],
            "away_passing_yds_allowed_pg": away_stats["passing_yds_allowed_pg"],
            "away_passing_tds_allowed_pg": away_stats["passing_tds_allowed_pg"],
            "away_rushing_yds_allowed_pg": away_stats["rushing_yds_allowed_pg"],
            "away_rushing_tds_allowed_pg": away_stats["rushing_tds_allowed_pg"],
            "away_receiving_yds_allowed_pg": away_stats["receiving_yds_allowed_pg"],
            "away_receiving_tds_allowed_pg": away_stats["receiving_tds_allowed_pg"],
        }

        return game_features
    
    def create_dataset(self,
                       weekly_data, schedule_data,
                       return_type = "df"):
        """
        Create a complete dataset for model training from historical games.

        Iterates through all games in the schedule and creates feature vectors
        along with outcomes for model training.

        Args:
            weekly_data: DataFrame of weekly team statistics
            schedule_data: DataFrame of game schedules
            return_type: Format to return ("df" for DataFrame, else list of dicts)

        Returns:
            DataFrame or list of dictionaries containing game features and outcomes
        """
        
        dataset = []

        for _, game in schedule_data.iterrows():
            season = game['season']
            week = game['week']
            home_team = game['home_team']
            away_team = game['away_team']
            is_playoff = game['game_type'] != 'REG'
            home_rest = game['home_rest']
            away_rest = game['away_rest']
            home_score = game['home_score']
            away_score = game['away_score']
            spread = game['spread_line']

            if pd.isna(home_score) or pd.isna(away_score):
                continue # Skip games without scores

            if pd.isna(home_team) or pd.isna(away_team):
                continue # Skip games without team info

            team_features = self.create_team_features(weekly_data, season, week)
            if team_features is None:
                continue

            game_features = self.create_game_features(team_features, home_team, away_team, season, week, is_playoff)
            if game_features is None:
                continue
                
            if away_score + spread == home_score:
                home_spread_covered = 1 # Push
            elif away_score + spread < home_score:
                home_spread_covered = 2 # Home team covers
            else:
                home_spread_covered = 0 # Away team covers

            game_features['home_spread_covered'] = home_spread_covered
            game_features['home_win'] = 1 if home_score > away_score else 0
            game_features['home_rest'] = home_rest
            game_features['away_rest'] = away_rest
            #game_features['game_id'] = game['game_id']
            game_features['spread'] = spread

            dataset.append(game_features)

        # Return dataset in requested format
        if return_type == "df":
            # Convert dataset to DataFrame and return it
            return pd.DataFrame(dataset)
        else:
            # Return dataset as a list of dictionaries
            return dataset

    def dataset_df_to_csv(self,
                        dataset_df,
                        file_name=None):
        """
        Save dataset DataFrame to a CSV file.

        Args:
            dataset_df: DataFrame containing the dataset
            file_name: Name of file to save (auto-generated if None)

        Returns:
            File handle if successful, None otherwise
        """
        
        # Set default file name if not provided
        if file_name is None:
            file_name = f'nfl_game_dataset_{datetime.datetime.now().strftime("%Y%m%d")}.csv'

        # Save dataset to CSV
        try:
            with open(file_name, 'w', newline='') as csvfile:
                dataset_df.to_csv(path_or_buf=csvfile, index=False)
                print(f"Dataset saved to {file_name}")
                return csvfile
        except Exception as e:
            print(f"Error saving dataset to file: {e}")


    def select_features(self,
                        X_train, y_train,
                        min_features=5,
                        model_name = 'random_forest'):
        """
        Perform feature selection using Recursive Feature Elimination with Cross-Validation.

        Uses RFECV to find the optimal subset of features for the specified model.

        Args:
            X_train: Training features DataFrame
            y_train: Training labels
            min_features: Minimum number of features to select (default: 5)
            model_name: Type of model to use for selection (default: 'random_forest')

        Returns:
            List of selected feature names
        """

        # List for storing selected features for model
        model_features = []

        # Using RFECV to find optimal feature set for model
        print(f'Selecting features for {model_name} model using RFECV with minimum {min_features} features...')

        # Make sure model is recognized
        if model_name not in self.available_models:
            print(f"Model {model_name} not recognized. Skipping.")
            return []

        if model_name == 'gradient_boosting':
            estimator = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            )
        elif model_name == 'decision_tree':
            estimator = DecisionTreeClassifier(
                max_depth=5,
                random_state=42
            )
        elif model_name == 'random_forest':
            estimator = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42
            )
        elif model_name == 'logistic_regression':
            estimator = LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        elif model_name == 'xgboost':
            estimator = xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42,
                verbosity=0
            )

        # Define cross-validation strategy
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)

        # Perform RFECV
        print(f"Selecting features for {model_name} model...")
        rfecv = RFECV(
            estimator=estimator,
            step=1,
            cv=cv,
            scoring='accuracy',
            min_features_to_select=min_features,
            n_jobs=1
        )
        rfecv.fit(X_train, y_train)

        # Print and return selected features
        selected_features = X_train.columns[rfecv.support_].tolist()
        model_features = selected_features
        print(f"{model_name}: Selected {len(selected_features)} features.")
        print(f"Features: {selected_features}")

        return model_features
    
    def tune_model(self,
                   X_train, y_train, X_test, y_test,
                   model_features,
                   model_name='random_forest'):
        """
        Tune model hyperparameters using GridSearchCV.

        Performs exhaustive search over specified parameter values to find
        the best combination for the given model type.

        Args:
            X_train: Training features DataFrame
            y_train: Training labels
            X_test: Test features DataFrame
            y_test: Test labels
            model_features: List of feature names to use
            model_name: Type of model to tune (default: 'random_forest')

        Returns:
            Best estimator from grid search, or None if model not recognized
        """

        # Make sure model is available
        if model_name not in self.available_models:
            print(f"Model {model_name} not recognized. Skipping.")
            return None

        # Reformat feature set to selected features for given model
        X_train_selected = X_train[model_features]
        X_test_selected = X_test[model_features]

        # Select base model
        if model_name == 'gradient_boosting':
            model = GradientBoostingClassifier(random_state=42)
            param_grid = {
                'n_estimators': [200, 300, 400, 500],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [8, 9, 10, 11, 12]
            }
        elif model_name == 'decision_tree':
            model = DecisionTreeClassifier(random_state=42)
            param_grid = {
                'criterion': ['gini', 'entropy'],
                'max_depth': [5, 10, 15, 20, 25],
                'min_samples_split': [2, 5, 10, 20],
                'min_samples_leaf': [1, 2, 5, 10]
            }
        elif model_name == 'random_forest':
            model = RandomForestClassifier(random_state=42)
            param_grid = {
                'n_estimators': [200, 300, 400, 500],
                'max_depth': [8, 9, 10, 11, 12],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        elif model_name == 'logistic_regression':
            model = LogisticRegression(random_state=42, max_iter=1000)
            param_grid = {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            }
        elif model_name == 'xgboost':
            model = xgb.XGBClassifier(random_state=42, verbosity=0)
            param_grid = {
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'n_estimators': [100, 200, 300],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0],
                'reg_alpha': [0, 1],
                'reg_lambda': [1, 5]
            }

        print(f"Tuning {model} model...")

         # Define cross-validation strategy
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
            
        # Set up GridSearchCV
        grid_search = GridSearchCV(estimator=model, 
                                    param_grid=param_grid, 
                                    scoring='accuracy', cv=cv, 
                                    n_jobs=1)
        
        # Scale features if necessary
        if "logistic" in model_name:
            X_train_selected = StandardScaler().fit_transform(X_train_selected)
            X_test_selected = StandardScaler().fit_transform(X_test_selected)
            
        # Fit grid search
        grid_search.fit(X_train_selected, y_train)

        # Print results
        print(f"Best parameters for {model_name}: {grid_search.best_params_}")
        print(f"Best cross-validation accuracy for {model_name}: {grid_search.best_score_:.4f}")
        test_score = grid_search.score(X_test_selected, y_test)
        print(f"Test set accuracy for {model_name}: {test_score:.4f}")

        final_model = grid_search.best_estimator_

        return final_model

    def train_model(self,
                     dataset,
                     model_name = 'random_forest',
                     model_label = None,
                     create_model_package = True,
                     score_model = True,
                     model_target='home_win'):
        """
        Train a machine learning model on NFL game data.

        Complete training pipeline including feature selection, hyperparameter tuning,
        and model evaluation. Returns a model package ready for predictions or storage.

        Args:
            dataset: DataFrame containing game features and outcomes
            model_name: Type of model to train (default: 'random_forest')
            model_label: Label for the model package (auto-generated if None)
            create_model_package: Whether to package the model (default: True)
            score_model: Whether to evaluate model with cross-validation (default: True)
            model_target: Target variable to predict (default: 'home_win')

        Returns:
            Model package dictionary if create_model_package=True, else trained model
        """
        
        # Create default model label if not provided
        if model_label is None:
            model_label = f'{model_name}_{datetime.datetime.now().strftime("%m_%d_%Y")}'

        # Check if model is available
        if model_name not in self.available_models:
            print(f"{model_name} is not available for training. Skipping...")
            return None

        print(f"Starting training for {model_name} model")
        # Remove columns with NaN values or constant values and 'game_id' column 
        dataset = dataset.loc[:, dataset.var() > 0]
        dataset = dataset.fillna(dataset.mean())

        # Split dataset into features and target
        X = dataset.drop(columns=['home_spread_covered', 'home_win'], axis=1)
        if model_target == 'home_spread_covered':
            y = dataset['home_spread_covered']
        else:
            y = dataset['home_win']

        # Perform train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, 
            y, 
            test_size=0.2, 
            random_state=42, 
            stratify=dataset['home_win']
        )

        # Select model features using RFECV
        model_features = self.select_features(X_train=X_train, y_train=y_train, model_name=model_name)

        # Tune model hyperparameters using GridSearchCV
        model = self.tune_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                model_features=model_features, model_name=model_name)

        # Score model if specified
        model_scores = {}
        if score_model:
            print(f"Scoring {model_name} model labeled {model_label}...")

            # Specify CV strategy
            cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)

            # Scale features if necessary
            if "logistic" in model_name:
                X_train = StandardScaler().fit_transform(X_train)
                X_test = StandardScaler().fit_transform(X_test)

            # Score model and print/save scores
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=1)
            model_scores = {
                "mean_accuracy": float(np.mean(scores)),
                "std_accuracy": float(np.std(scores)),
                "scores": scores.tolist()  # Convert numpy array to Python list
            }
            print(f"{model_label}: Mean Accuracy = {scores.mean():.4f}, Std = {scores.std():.4f}")

        # Save and return model package if specified
        if create_model_package:
            encoded_model = self.encode_model(model)
            list_of_dicts = dataset.to_dict(orient='records')
            model_package = {
                "package_label": model_label,
                "model": encoded_model,
                "model_features": model_features,
                "model_scores": model_scores,
                "dataset": list_of_dicts,
                "model_target": model_target,
                "date_trained": datetime.date.today().strftime("%m-%d-%Y")
            }

            return model_package
        
        # Return model
        return model

    def predict_games(self,
                 weekly_data, schedule_data,
                 season, week,
                 model_package: dict,
                 save_to_csv=True
                 ):
        """
        Generate predictions for all games in a specific week.

        Uses the trained model to predict outcomes for each scheduled game,
        creating features from team statistics up to that point in the season.

        Args:
            weekly_data: DataFrame of weekly team statistics
            schedule_data: DataFrame of game schedules
            season: NFL season year
            week: Week number to predict
            model_package: Dictionary containing trained model and metadata
            save_to_csv: Whether to save predictions to CSV file (default: True)

        Returns:
            List of prediction dictionaries for each game
        """
        
        # Make sure model package exists
        if model_package is None:
            print(f"Model package does not exist. Unable to predict NFL {season} week {week} games.")
        
        # Obtain model from model package
        encoded_model = model_package["model"]
        model = self.decode_model(encoded_model)

        # List for storing predictions
        predictions = []

        # Prepare schedule data for the specified week
        schedule_data = schedule_data[["home_team", "away_team", "season", "week", "home_rest", "away_rest", "spread_line"]]
        games = schedule_data[(schedule_data['season'] == season) & (schedule_data['week'] == week)]

        # Create features for each upcoming game
        for _, game in games.iterrows():
            home_team = game['home_team']
            away_team = game['away_team']
            home_rest = game['home_rest']
            away_rest = game['away_rest']
            spread = game['spread_line']

            team_features = self.create_team_features(weekly_data, season, week)
            if team_features is None:
                continue

            game_features = self.create_game_features(team_features, home_team, away_team, season, week)
            if game_features is None:
                continue

            game_features['home_team'] = home_team
            game_features['away_team'] = away_team
            game_features['home_rest'] = home_rest
            game_features['away_rest'] = away_rest
            game_features['spread'] = spread
            game_df = pd.DataFrame([game_features])

            features = model_package["model_features"]
            X_pred = game_df[features]
            if "logistic" in model_package["package_label"]:
                X_pred = StandardScaler().fit_transform(X_pred)
            
            probability = model.predict_proba(X_pred)[:, 1][0]
            prediction = model.predict(X_pred)[0]

            predictions.append({
                "season": season,
                "week": week,
                "home_team": home_team,
                "away_team": away_team,
                "home_win": True if prediction == 1 else False,
                "confidence": probability,
                "model_used": model_package["package_label"],
                "is_correct": None
            })

        if save_to_csv:
            pd.DataFrame(predictions).to_csv(f"NFL_{season}_week{week}_{model_package.target_column}_predictions_by_{model_package.label}.csv",
                                      index=False)
            
        return predictions

    def encode_model(self, model) -> str:
        """
        Serialize a sklearn model using joblib and encode it to a base64 string.

        Args:
            model: A trained sklearn model object

        Returns:
            str: Base64 encoded string representation of the model
        """
        try:
            import joblib
            import base64
            from io import BytesIO

            # Create a BytesIO buffer to hold the serialized model
            buffer = BytesIO()

            # Serialize the model to the buffer using joblib
            joblib.dump(model, buffer)

            # Get the bytes from the buffer
            buffer.seek(0)
            model_bytes = buffer.read()

            # Encode to base64
            encoded_model = base64.b64encode(model_bytes).decode('utf-8')

            print(f"Model successfully encoded to base64 string (length: {len(encoded_model)})")
            return encoded_model

        except ImportError as e:
            print(f"Failed to import required libraries. Make sure joblib is installed using 'uv add joblib'.")
            return None
        except Exception as e:
            print(f"Error encoding model: {e}")
            return None

    def decode_model(self, encoded_model_str: str):
        """
        Decode a base64 string and deserialize it back to a sklearn model.

        Args:
            encoded_model_str: Base64 encoded string representation of a model

        Returns:
            The deserialized sklearn model object
        """
        try:
            import joblib
            import base64
            from io import BytesIO

            # Decode the base64 string to bytes
            model_bytes = base64.b64decode(encoded_model_str)

            # Create a BytesIO buffer from the bytes
            buffer = BytesIO(model_bytes)

            # Deserialize the model from the buffer using joblib
            model = joblib.load(buffer)

            print(f"Model successfully decoded from base64 string")
            return model

        except ImportError as e:
            print(f"Failed to import required libraries. Make sure joblib is installed using 'uv add joblib'.")
            return None
        except Exception as e:
            print(f"Error decoding model: {e}")
            return None
