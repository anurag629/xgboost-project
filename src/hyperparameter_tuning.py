from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

def tune_hyperparameters(X_train, y_train):
    """Perform hyperparameter tuning using GridSearchCV."""
    # Define the hyperparameter grid
    param_grid = {
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'n_estimators': [50, 100, 150],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    # Initialize GridSearchCV
    grid_search = GridSearchCV(
        estimator=XGBClassifier(objective="multi:softmax", num_class=3, eval_metric="mlogloss", random_state=42),
        param_grid=param_grid,
        scoring="accuracy",
        cv=3,
        n_jobs=-1,
        verbose=2
    )

    # Fit the model
    grid_search.fit(X_train, y_train)

    # Return best parameters
    print(f"Best Parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_
