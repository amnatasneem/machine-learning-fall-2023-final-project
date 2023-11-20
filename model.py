import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load datasets
heart_data = pd.read_csv("data/heart.csv", sep=",")
o2_data = pd.read_csv("data/o2saturation.csv")

# Classification: heart attack prediction
X_classification = heart_data.drop(heart_data.columns[13], axis=1)
y_classification = heart_data[heart_data.columns[13]]

# Split the data (train and test)
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
    X_classification, y_classification, test_size=0.2, random_state=50
)

# Standardize features
scaler = StandardScaler()
X_train_class_scaled = scaler.fit_transform(X_train_class)
X_test_class_scaled = scaler.transform(X_test_class)

# Logistic regression
log_reg_model = LogisticRegression()
log_reg_model.fit(X_train_class_scaled, y_train_class)
y_pred_labels = log_reg_model.predict(X_test_class_scaled)

# Print classification results
print("Logistic Regression - Classification Results: ")
print(classification_report(y_test_class, y_pred_labels))

# Regression: O2 saturation prediction
X_regression = o2_data.drop(o2_data.columns[0], axis=1)
y_regression = o2_data[o2_data.columns[0]]

# Split the data (train and test)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_regression, y_regression, test_size=0.2, random_state=50
)