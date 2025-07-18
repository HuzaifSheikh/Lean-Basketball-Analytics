import pandas as pd
import numpy as np
import shap
import optuna
from pathlib import Path
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge  # Using Ridge instead of plain LinearRegression
from sklearn.model_selection import KFold, TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set random seed for reproducibility
seed = 6748

# Ensure directories exist
os.makedirs("models", exist_ok=True)
os.makedirs("images", exist_ok=True)
os.makedirs("model_results", exist_ok=True)

def evaluate(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R²": r2_score(y_true, y_pred),
        "MAPE": np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    }

def train_and_evaluate(train_data, test_data, y_column, n_trials=20):
    """
    Train models dynamically based on the target feature (y_column), optimize models, evaluate performance, 
    save the best model, and generate a comparison plot.
    """
    
    # ------------------------------
    # Data Preparation & Alignment
    # ------------------------------
    # Drop rows with missing target values in training
    train_data = train_data.dropna(subset=[y_column])
    
    # Drop unwanted columns
    exclude_cols = ["game_id", "season", "team1", "team2", "team1_oe", "team2_oe", 
                    "team1_de", "team2_de",
                    "team1_team2_spread"]
    
    X_train = train_data.drop(columns=exclude_cols)
    y_train = train_data[y_column]
    X_test = test_data.drop(columns=exclude_cols)
    y_test = test_data[y_column]
    
    # Drop rows with missing feature values and align y accordingly
    valid_idx_train = X_train.dropna().index
    X_train = X_train.loc[valid_idx_train].reset_index(drop=True)
    y_train = y_train.loc[valid_idx_train].reset_index(drop=True)

    valid_idx_test = X_test.dropna().index
    X_test = X_test.loc[valid_idx_test].reset_index(drop=True)
    y_test = y_test.loc[valid_idx_test].reset_index(drop=True)

    
    # ------------------------------
    # Feature Scaling
    # ------------------------------
    scaler = StandardScaler()
    X_train_scaled_arr = scaler.fit_transform(X_train)  # Fit only on training data
    X_test_scaled_arr = scaler.transform(X_test)         # Transform test data
    
    # Convert back to DataFrame to preserve column names and indices (required for SHAP)
    X_train_scaled = pd.DataFrame(X_train_scaled_arr, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled_arr, columns=X_test.columns, index=X_test.index)
    
    # ------------------------------
    # Feature Selection using SHAP
    # ------------------------------
    base_model = XGBRegressor(n_estimators=100, random_state=seed)
    base_model.fit(X_train, y_train)

    # Compute SHAP values
    explainer = shap.Explainer(base_model)
    shap_values = explainer(X_train) 

    # Convert SHAP values to DataFrame
    shap_df = pd.DataFrame(shap_values.values, columns=X_train.columns)

    # Compute mean absolute SHAP values per feature
    shap_importance = shap_df.abs().mean().sort_values(ascending=False)

    # Select features with importance above the mean
    selected_features = shap_importance[shap_importance > shap_importance.mean()].index.tolist()

    # SHAP summary plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_train_scaled, feature_names=X_train.columns, show=False)
    plt.savefig(f"images/shap_summary_{y_column}.png", bbox_inches="tight", dpi=300)
    plt.close()

    # Use only selected features for modeling
    X_train_selected = X_train_scaled[selected_features]
    X_test_selected = X_test_scaled[selected_features]

    # Ensure y_test is aligned
    y_test = y_test.loc[X_test_selected.index].reset_index(drop=True)
    
    # ------------------------------
    # Model Optimization & Evaluation
    # ------------------------------
    # Define the models to optimize
    models_to_optimize = ["Random Forest", "XGBoost", "LightGBM", "MLP Regressor", "SVR", "Linear Regression"]
    
    # Define the Optuna objective function for time-series cross-validation
    def objective_kfold_cv(trial, model_name):
        if model_name == "Random Forest":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            }
            model = RandomForestRegressor(**params, random_state=seed)
        
        elif model_name == "XGBoost":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            }
            model = XGBRegressor(**params, random_state=seed)
        
        elif model_name == "LightGBM":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            }
            model = LGBMRegressor(**params, random_state=seed)
        
        elif model_name == "MLP Regressor":
            params = {
                "hidden_layer_sizes": trial.suggest_categorical("hidden_layer_sizes", [(64, 32), (128, 64), (256, 128)]),
                "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
                "solver": trial.suggest_categorical("solver", ["adam", "lbfgs"]),
            }
            model = MLPRegressor(max_iter=500, random_state=seed, **params)
        
        elif model_name == "SVR":
            params = {
                "kernel": trial.suggest_categorical("kernel", ["linear", "rbf"]),
                "C": trial.suggest_float("C", 0.1, 10.0),
                "epsilon": trial.suggest_float("epsilon", 0.01, 0.5),
            }
            model = SVR(**params)
        
        elif model_name == "Linear Regression":
            # Replace plain Linear Regression with Ridge Regression for regularization.
            params = {
                "alpha": trial.suggest_float("alpha", 0.001, 10.0, log=True)
            }
            model = Ridge(**params, random_state=seed)
        
        # Perform Time-Series Cross-Validation using neg MAE as the score.
        score = cross_val_score(model, X_train_selected, y_train, 
                                cv= TimeSeriesSplit(n_splits=5), 
                                scoring="neg_mean_absolute_error").mean()
        return -score

    # Optimize models
    evaluation_results = {}
    optimized_models = {}
    
    # Define mapping for later model instantiation
    model_mapping = {
        "Random Forest": RandomForestRegressor,
        "XGBoost": XGBRegressor,
        "LightGBM": LGBMRegressor,
        "MLP Regressor": MLPRegressor,
        "SVR": SVR,
        "Linear Regression": Ridge  # Use Ridge for regularized linear regression
    }
    
    # Optimize and train each model
    for model_name in models_to_optimize:
        study = optuna.create_study(direction="minimize", study_name=f'NCAA OE and DE Predictions {y_column}')
        study.optimize(lambda trial: objective_kfold_cv(trial, model_name), n_trials=n_trials)
        optimized_models[model_name] = study.best_params
        
        # Instantiate and train the optimized model on the selected features
        model_class = model_mapping[model_name]
        model = model_class(**optimized_models[model_name])
        model.fit(X_train_selected, y_train)
        
        # Evaluate training performance
        y_train_pred = model.predict(X_train_selected)
        evaluation_results[model_name] = evaluate(y_train, y_train_pred)
    
    # Save evaluation results to CSV
    metrics_df = pd.DataFrame(evaluation_results).T
    metrics_df.to_csv(f"model_results/evaluation_results_{y_column}.csv", index=True)
    
    # ------------------------------
    # Visualization of Model Performance
    # ------------------------------
    # Define Georgia Tech colors
    gt_gold = "#B3A369"
    gt_white = "#E5E4E2"
    gt_navy = "#003057"
    
    # Create bar plot for MAE and RMSE
    fig, ax = plt.subplots(figsize=(12, 6))
    metrics_df[["MAE", "RMSE"]].plot(kind="bar", ax=ax, color=[gt_gold, gt_navy],
                                      alpha=0.85, edgecolor="black")
    ax.set_ylim(0, metrics_df[["MAE", "RMSE"]].max().max() * 1.15)
    
    # Add data labels on each bar
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}",
                    (p.get_x() + p.get_width() / 2, p.get_height() + 0.1),
                    ha="center", va="bottom", fontsize=10, fontweight="bold",
                    color=gt_navy)
    
    plt.title(f"Model Performance Comparison ({y_column})", fontsize=14, fontweight="bold", color=gt_navy)
    plt.ylabel("Error", fontsize=12, color=gt_navy)
    plt.xticks(rotation=45, ha="right", fontsize=10, color=gt_navy)
    plt.yticks(fontsize=10, color=gt_navy)
    plt.legend(title="Metrics", fontsize=10, title_fontsize=12, facecolor=gt_white, edgecolor="black")
    plt.grid(axis="y", linestyle="--", alpha=0.7, color=gt_navy)
    plt.box(False)
    
    plt.savefig(f"images/model_comparison_{y_column}.png", bbox_inches="tight", dpi=300, facecolor=gt_white)
    
    # ------------------------------
    # Best Model Selection & Test Evaluation
    # ------------------------------
    # Here, we select the best model based on lowest MAE. 
    best_model_name = min(evaluation_results, key=lambda x: evaluation_results[x]["MAE"])
    best_model_class = model_mapping[best_model_name]
    best_model = best_model_class(**optimized_models[best_model_name])
    
    # Train best model on full training data (selected features)
    best_model.fit(X_train_selected, y_train)
    
    # Save best model to disk
    joblib.dump(best_model, f"models/best_model_{y_column}.pkl")
    
    # Evaluate test performance
    y_test_pred = best_model.predict(X_test_selected)
    test_results = evaluate(y_test, y_test_pred)
    
    # Save test performance results to CSV
    test_metrics_df = pd.DataFrame([test_results], index=[best_model_name])
    test_metrics_df.to_csv(f"model_results/test_evaluation_results_{y_column}.csv", index=True)
    
    print(f"Best model ({best_model_name}) saved for {y_column}.")
    print("Test Performance:", test_results)
    
    return best_model, evaluation_results, test_results
