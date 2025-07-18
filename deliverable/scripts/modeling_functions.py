import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy import stats

def predict_game_outcomes(model_input):
    """
    Predicts team offensive efficiency (OE), defensive efficiency (DE), expected scores,
    spread, and probability based on trained models.
    
    Args:
        model_input (pd.DataFrame): Dataframe containing test game data.
    
    Returns:
        model_output (pd.DataFrame): Dataframe with predictions and calculated metrics.
        results_dict (dict): Dictionary containing key predictions for each game.
    """
    models_dir = "models"
    if not os.path.exists(models_dir):
        raise FileNotFoundError("Models directory not found. Please train models first.")

    exclude_cols = ["game_id", "team1", "team2", "team1_oe", "team2_oe", "team1_de", "team2_de", "team1_team2_spread"]
    X_test = model_input.drop(columns=exclude_cols)
    
    # Drop NaN values before applying transformations to maintain alignment
    X_test_clean = X_test.dropna().reset_index(drop=True)
    
    # Standardize features
    scaler = StandardScaler()
    X_test_scaled = scaler.fit_transform(X_test_clean)
    model_input_clean = model_input.loc[X_test_clean.index].reset_index(drop=True)
    
    predictions_df = model_input_clean[["game_id", "team1", "team2"]].copy()
    metrics = ["team1_oe", "team2_oe", "team1_de", "team2_de"]
    
    for metric in metrics:
        model_path = f"{models_dir}/best_model_{metric}.pkl"
        if not os.path.exists(model_path):
            print(f"Model for {metric} not found. Skipping...")
            continue
        
        best_model = joblib.load(model_path)
        feature_names = best_model.feature_names_in_
        X_test_selected = pd.DataFrame(X_test_scaled, columns=X_test_clean.columns)[feature_names]
        
        if len(X_test_selected) != len(predictions_df):
            print(f"Warning: Dropping {len(predictions_df) - len(X_test_selected)} records to match prediction size for {metric}")
            predictions_df = predictions_df.iloc[:len(X_test_selected)].reset_index(drop=True)
        
        predictions_df[f"pred_{metric}"] = best_model.predict(X_test_selected)
    
    model_output = model_input_clean.merge(predictions_df, on=["game_id", "team1", "team2"], how="left")
    
    # Calculate expected scores
    model_output["expected_team1_score"] = (model_output["pred_team1_oe"] * model_output["total_possessions_team1"]) / 100
    model_output["expected_team2_score"] = (model_output["pred_team2_oe"] * model_output["total_possessions_team2"]) / 100
    
    # Calculate predicted spread
    model_output["predicted_spread"] = model_output["expected_team1_score"] - model_output["expected_team2_score"]
    
    # Fit normal distribution for spread probability
    spread_mean = model_output["predicted_spread"].mean()
    spread_std = model_output["predicted_spread"].std()
    model_output["spread_probability"] = 1 - stats.norm.cdf(model_output["predicted_spread"], loc=spread_mean, scale=spread_std)
    
    # Construct results dictionary
    results_dict = {}
    for _, row in model_output.iterrows():
        game_id = row["game_id"]
        results_dict[game_id] = {
            "team1": row["team1"],
            "team2": row["team2"],
            "pred_team1_oe": row["pred_team1_oe"],
            "pred_team2_oe": row["pred_team2_oe"],
            "pred_team1_de": row["pred_team1_de"],
            "pred_team2_de": row["pred_team2_de"],
            "expected_team1_score": row["expected_team1_score"],
            "expected_team2_score": row["expected_team2_score"],
            "predicted_spread": row["predicted_spread"],
            "spread_probability": row["spread_probability"]
        }
    
    return model_output, results_dict
