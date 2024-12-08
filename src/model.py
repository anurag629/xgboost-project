from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

def train_model(X_train, y_train):
    """Train the XGBoost model."""
    model = XGBClassifier(objective="multi:softmax", num_class=3, eval_metric="mlogloss", random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and return accuracy and classification report."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, report
