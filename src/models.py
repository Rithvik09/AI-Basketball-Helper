from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import joblib
import os

class MLPredictor:
    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.scaler = StandardScaler()
        self.classification_model = None
        self.regression_model = None
        self._load_or_create_models()

    def _load_or_create_models(self):
        """Load existing models or create new ones"""
        try:
            self.classification_model = joblib.load(f'{self.model_dir}/classification_model.joblib')
            self.regression_model = joblib.load(f'{self.model_dir}/regression_model.joblib')
            self.scaler = joblib.load(f'{self.model_dir}/scaler.joblib')
        except:
            self.classification_model = RandomForestClassifier(
                n_estimators=100, 
                max_depth=10,
                random_state=42
            )
            self.regression_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )

    def prepare_features(self, player_stats, opponent_data=None, matchup_history=None, team_data=None):
        """Prepare features for prediction"""
        features = {}
        
        # Basic player performance features
        features.update({
            'recent_avg': float(player_stats.get('last5_avg', 0)),
            'season_avg': float(player_stats.get('avg', 0)),
            'max_recent': float(player_stats.get('max', 0)),
            'min_recent': float(player_stats.get('min', 0)),
            'stddev': float(np.std(player_stats.get('values', [0]))),
            'games_played': len(player_stats.get('values', [])),
        })

        # Matchup features
        if matchup_history:
            matchup_values = matchup_history.get('values', [])
            features.update({
                'vs_team_avg': float(np.mean(matchup_values)) if matchup_values else 0,
                'vs_team_max': float(np.max(matchup_values)) if matchup_values else 0,
                'vs_team_min': float(np.min(matchup_values)) if matchup_values else 0,
                'vs_team_games': len(matchup_values),
            })

        # Team context features
        if team_data:
            features.update({
                'team_recent_form': float(team_data.get('recent_form', 0)),
                'team_pace': float(team_data.get('pace', 0)),
                'team_off_rating': float(team_data.get('offensive_rating', 0)),
            })

        # Opponent context features
        if opponent_data:
            features.update({
                'opp_recent_form': float(opponent_data.get('recent_form', 0)),
                'opp_def_rating': float(opponent_data.get('defensive_rating', 0)),
                'matchup_advantage': float(team_data.get('offensive_rating', 0) - 
                                        opponent_data.get('defensive_rating', 0))
                                        if team_data and opponent_data else 0,
            })

        return features

    def train(self, training_data):
        """Train both classification and regression models"""
        if not training_data:
            raise ValueError("No training data provided")

        # Prepare features and targets
        X = pd.DataFrame([data['features'] for data in training_data])
        y_class = [1 if data['result'] > data['line'] else 0 for data in training_data]
        y_reg = [data['result'] for data in training_data]

        # Split data
        X_train, X_test, y_class_train, y_class_test = train_test_split(X, y_class, test_size=0.2)
        _, _, y_reg_train, y_reg_test = train_test_split(X, y_reg, test_size=0.2)

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train models
        self.classification_model.fit(X_train_scaled, y_class_train)
        self.regression_model.fit(X_train_scaled, y_reg_train)

        # Save models
        joblib.dump(self.classification_model, f'{self.model_dir}/classification_model.joblib')
        joblib.dump(self.regression_model, f'{self.model_dir}/regression_model.joblib')
        joblib.dump(self.scaler, f'{self.model_dir}/scaler.joblib')

    def predict(self, features, line):
        try:
            if not self.classification_model or not self.regression_model:
                self._load_or_create_models()

            features_df = pd.DataFrame([features])
            
            # Simple scaling
            scaled_features = features_df.copy()
            for col in scaled_features.columns:
                if scaled_features[col].std() != 0:
                    scaled_features[col] = (scaled_features[col] - scaled_features[col].mean()) / scaled_features[col].std()

            # Make predictions
            over_prob = self.classification_model.predict_proba(scaled_features)[0][1]
            predicted_value = self.regression_model.predict(scaled_features)[0]

            # Calculate confidence and recommendation
            confidence = self._calculate_confidence(over_prob, predicted_value, line)
            recommendation = self._generate_recommendation(over_prob, predicted_value, line)

            return {
                'over_probability': float(over_prob),
                'predicted_value': float(predicted_value),
                'recommendation': recommendation,
                'confidence': confidence,
                'edge': float((predicted_value - line) / line) if line > 0 else 0
            }
        except Exception as e:
            print(f"Prediction error: {e}")
            # Fallback to basic prediction
            return {
                'over_probability': 0.5,
                'predicted_value': features.get('season_avg', line),
                'recommendation': 'PASS',
                'confidence': 'LOW',
                'edge': 0.0
            }

    def _calculate_confidence(self, prob, pred_value, line):
        """Calculate prediction confidence"""
        prob_distance = abs(prob - 0.5)
        value_distance = abs(pred_value - line) / line if line > 0 else 0
        
        if prob_distance > 0.2 and value_distance > 0.1:
            return 'HIGH'
        elif prob_distance > 0.15 or value_distance > 0.07:
            return 'MEDIUM'
        return 'LOW'

    def _generate_recommendation(self, prob, pred_value, line):
        """Generate betting recommendation"""
        if prob > 0.65 or (pred_value - line) / line > 0.1:
            return 'OVER'
        elif prob < 0.35 or (line - pred_value) / line > 0.1:
            return 'UNDER'
        return 'PASS'