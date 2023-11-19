import pandas as pd
from sklearn.model_selection import train_test_split

# Load datasets
heart_data = pd.read_csv("data/heart.csv")
o2_data = pd.read_csv("data/o2saturation.csv")

# Classification: heart attack prediction
X_classification = heart_data.drop("output", axis=1)
y_classification = heart_data["ouput"]

# Split the data (train and test)
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
    X_classification, y_classification, test_size=0.2, random_state=50
)

# Regression: O2 saturation prediction
X_regression = o2_data.drop(o2_data.columns[0], axis=1)
y_regression = o2_data[o2_data.columns[0]]

# Split the data (train and test)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_regression, y_regression, test_size=0.2, random_state=50
)