from pyexpat import features
import nfl_data_py as nfl
import nflreadpy as nflr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

# XGBoost import
import xgboost as xgb


class NFLPredictor:
    def __init__(self, current_season=2025):
        self.models = {}
        self.model_features = {} # Model name -> list of features
        self.final_model = None
        self.current_season = current_season
        self.dataset = pd.DataFrame()
    

    def import_data(self, start_year=2014, end_year=2025):
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

    def create_team_features(self, weekly_data, season, week):
        
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

    def create_game_features(self, team_features, home_team, away_team, season, week, is_playoff=False, is_neutral=False):
        
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

    def create_dataset(self, weekly_data, schedule_data, save_data=True):
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

            game_features['home_win'] = 1 if home_score > away_score else 0
            game_features['home_rest'] = home_rest
            game_features['away_rest'] = away_rest
            #game_features['game_id'] = game['game_id']
            game_features['spread'] = spread

            dataset.append(game_features)

        if save_data:
            self.dataset = pd.DataFrame(dataset)
            self.dataset.to_csv('nfl_game_dataset.csv', index=False)
            print("Dataset saved to nfl_game_dataset.csv")
        return pd.DataFrame(dataset)

    def select_features(self, 
                        X_train, y_train,
                        min_features=5, 
                        models=['Gradient Boosting', 'Decision Tree', 'Random Forest', 'Logistic Regression', 'XGBoost']):
        
        # Dict for storing selected features for each model
        model_features = {}

        # Using RFECV to find optimal feature set for each model
        print(f'Selecting features using RFECV with minimum {min_features} features...')
        for model in models:
            # Make sure model is recognized
            if model not in ['Gradient Boosting', 'Decision Tree', 'Random Forest', 'Logistic Regression', 'XGBoost']:
                print(f"Model {model} not recognized. Skipping.")
                continue

            if model == 'Gradient Boosting':
                estimator = GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=3,
                    random_state=42
                )
            elif model == 'Decision Tree':
                estimator = DecisionTreeClassifier(
                    max_depth=5,
                    random_state=42
                )
            elif model == 'Random Forest':
                estimator = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=5,
                    random_state=42
                )
            elif model == 'Logistic Regression':
                estimator = LogisticRegression(
                    max_iter=1000,
                    random_state=42
                )
            elif model == 'XGBoost':
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
            print(f"Selecting features using {model}...")
            rfecv = RFECV(
                estimator=estimator,
                step=1,
                cv=cv,
                scoring='accuracy',
                min_features_to_select=min_features,
                n_jobs=1
            )
            rfecv.fit(X_train, y_train)

            selected_features = X_train.columns[rfecv.support_].tolist()
            model_features[model] = selected_features
            print(f"{model}: Selected {len(selected_features)} features.")

        self.model_features = model_features
        return model_features

    def train_and_compare_models(self, 
                     dataset, 
                     models=['Gradient Boosting', 'Decision Tree', 'Random Forest', 'Logistic Regression', 'XGBoost']):
        results = {}

        for model_name in models:
            if model_name not in self.model_features:
                print(f"No features selected for model {model_name}. Skipping training.")
                continue

            features = self.model_features[model_name]
            X = dataset[features].fillna(dataset[features].mean())
            y = dataset['home_win']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


            if model_name == 'Gradient Boosting':
                model = GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=3,
                    random_state=42
                )
            elif model_name == 'Decision Tree':
                model = DecisionTreeClassifier(
                    max_depth=5,
                    random_state=42
                )
            elif model_name == 'Random Forest':
                model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=5,
                    random_state=42
                )
            elif model_name == 'Logistic Regression':
                model = LogisticRegression(
                    max_iter=1000,
                    random_state=42
                )
            elif model_name == 'XGBoost':
                model = xgb.XGBClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=3,
                    random_state=42,
                    verbosity=0
                )
            else:
                model = xgb.XGBClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=3,
                    random_state=42,
                    verbosity=0
                )

            cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
            if "Logistic" in model_name:
                X_train = StandardScaler().fit_transform(X_train)
                X_test = StandardScaler().fit_transform(X_test)

            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=1)
            results[model_name] = {
                "mean_accuracy": np.mean(scores),
                "std_accuracy": np.std(scores),
                "scores": scores
            }

            print(f"{model_name}: Mean Accuracy = {scores.mean():.4f}, Std = {scores.std():.4f}")

        return models, results

    def tune_model(self, 
                   X_train, y_train, X_test, y_test,
                   models=['Gradient Boosting', 'Decision Tree', 'Random Forest', 'Logistic Regression', 'XGBoost']):

        for model_name in models:
            if model_name not in self.model_features:
                print(f"Model {model_name} not recognized. Skipping.")
                continue

            X_train_selected = X_train[self.model_features[model_name]]
            X_test_selected = X_test[self.model_features[model_name]]

            if model_name == 'Gradient Boosting':
                model = GradientBoostingClassifier(random_state=42)
                param_grid = {
                    'n_estimators': [200, 300, 400, 500],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [8, 9, 10, 11, 12]
                }
            elif model_name == 'Decision Tree':
                model = DecisionTreeClassifier(random_state=42)
                param_grid = {
                    'criterion': ['gini', 'entropy'],
                    'max_depth': [5, 10, 15, 20, 25],
                    'min_samples_split': [2, 5, 10, 20],
                    'min_samples_leaf': [1, 2, 5, 10]
                }
            elif model_name == 'Random Forest':
                model = RandomForestClassifier(random_state=42)
                param_grid = {
                    'n_estimators': [200, 300, 400, 500],
                    'max_depth': [8, 9, 10, 11, 12],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            elif model_name == 'Logistic Regression':
                model = LogisticRegression(random_state=42, max_iter=1000)
                param_grid = {
                    'C': [0.01, 0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear']
                }
            elif model_name == 'XGBoost':
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


            # Hyperparameter tuning logic would go here
            print(f"Tuning model {model}...")

            # Define cross-validation strategy
            cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
            
            # Set up GridSearchCV
            grid_search = GridSearchCV(estimator=model, 
                                       param_grid=param_grid, 
                                       scoring='accuracy', cv=cv, 
                                       n_jobs=1)
            
            if "Logistic" in model_name:
                X_train_selected = StandardScaler().fit_transform(X_train_selected)
                X_test_selected = StandardScaler().fit_transform(X_test_selected)
            
            grid_search.fit(X_train_selected, y_train)

            print(f"Best parameters for {model_name}: {grid_search.best_params_}")
            print(f"Best cross-validation accuracy for {model_name}: {grid_search.best_score_:.4f}")

            test_score = grid_search.score(X_test_selected, y_test)
            print(f"Test set accuracy for {model_name}: {test_score:.4f}")

            self.models[model_name] = grid_search.best_estimator_

    def predict_games(self, 
                 dataset, schedule_data,
                 season, week,
                 model_name='XGBoost'
                 ):
        
        # Check if model is trained
        if not (model_name in self.models and model_name in self.model_features):
            print(f"Model {model_name} not trained. Skipping predictions.")
            return None
        
        predictions = []

        # Prepare schedule data for the specified week
        schedule_data = schedule_data[["home_team", "away_team", "season", "week", "home_rest", "away_rest", "spread_line"]]
        games = schedule_data[(schedule_data['season'] == season) & (schedule_data['week'] == week)]

        # Create features for each game
        for _, game in games.iterrows():
            home_team = game['home_team']
            away_team = game['away_team']
            home_rest = game['home_rest']
            away_rest = game['away_rest']
            spread = game['spread_line']

            team_features = self.create_team_features(dataset, season, week)
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

            features = self.model_features[model_name]
            X_pred = game_df[features]
            if "Logistic" in model_name:
                X_pred = StandardScaler().fit_transform(X_pred)
            
            probability = self.models[model_name].predict_proba(X_pred)[:, 1][0]
            prediction = self.models[model_name].predict(X_pred)[0]
            pred_str = f"{home_team} WIN" if prediction == 1 else f"{away_team} WIN"


            predictions.append({
                "season": season,
                "week": week,
                "home_team": home_team,
                "away_team": away_team,
                "winner": pred_str,
                "home_probability": probability,
                "away_probability": 1 - probability,
                "prediction": prediction
            })

        return pd.DataFrame(predictions)     

def main():
    nfl_pred = NFLPredictor()
    weekly_data, team_dict, schedule_data = nfl_pred.import_data(2015, 2025)

    models = ['XGBoost', 'Random Forest']

    dataset = nfl_pred.create_dataset(weekly_data, schedule_data)

    # Remove columns with NaN values or constant values
    dataset = dataset.loc[:, dataset.var() > 0]
    dataset = dataset.fillna(dataset.mean())

    X_train, X_test, y_train, y_test = train_test_split(
        dataset.drop('home_win', axis=1), 
        dataset['home_win'], 
        test_size=0.2, 
        random_state=42, 
        stratify=dataset['home_win']
    )

    nfl_pred.select_features(X_train, y_train, min_features=5, models=models)

    nfl_pred.tune_model(X_train, y_train, X_test, y_test, models=models)

    for model_name in models:
        predictions = nfl_pred.predict_games(weekly_data, schedule_data, season=2025, week=7, model_name=model_name)
        if predictions is not None:
            print(f"\nPredictions for Week 7, 2025 using {model_name}:")
            print(predictions)

            # Save predictions to CSV
            predictions.to_csv(f"nfl_2025_week7_predictions_{model_name}.csv", index=False)


if __name__ == "__main__":
    main()
