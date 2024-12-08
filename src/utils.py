import logging
import xgboost as xgb

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def save_model(model, file_path):
    """Save the trained model to a file."""
    logging.info("Saving the model to %s", file_path)
    model.save_model(file_path)

def load_model(file_path):
    """Load a saved model from a file."""
    logging.info("Loading the model from %s", file_path)
    model = xgb.XGBClassifier()
    model.load_model(file_path)
    return model
