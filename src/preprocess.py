import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """Load the Iris dataset from the given path."""
    # Define column names for the dataset
    column_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
    data = pd.read_csv(file_path, header=None, names=column_names)
    
    # Map species names to numerical labels
    species_mapping = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
    data["species"] = data["species"].map(species_mapping)
    
    # Split features (X) and target variable (y)
    X = data.iloc[:, :-1]  # All columns except the last
    y = data.iloc[:, -1]   # The last column (species)
    return X, y

def preprocess_data(X, y):
    """Scale features and split data into training and testing sets."""
    # Standardize features using StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
