import matplotlib.pyplot as plt
from xgboost import plot_importance

def plot_feature_importance(model, save_path):
    """Plot and save feature importance."""
    plt.figure(figsize=(8, 6))
    plot_importance(model)
    plt.title("Feature Importance")
    plt.savefig(save_path)
    plt.close()
