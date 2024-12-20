from src.preprocess import load_data, preprocess_data
from src.model import train_model, evaluate_model
from src.visualize import plot_feature_importance
from src.utils import save_model

# Load the Iris dataset
data_path = "data/iris/iris.data"
X, y = load_data(data_path)

# Preprocess and split the dataset
X_train, X_test, y_train, y_test = preprocess_data(X, y)

# Train the model
model = train_model(X_train, y_train)

# Evaluate the model
accuracy, report = evaluate_model(model, X_test, y_test)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(report)

# Plot feature importance
plot_feature_importance(model, "results/feature_importance.png")

# Save the trained model
save_model(model, "models/xgboost_model.json")
