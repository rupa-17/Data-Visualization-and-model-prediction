import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,mean_squared_error, r2_score)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.compose import ColumnTransformer

def classification_models(data):
    df = data

    # Separate features and target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Determine task type
    task_type = "regression" if y.dtype == 'float64' or y.dtype == 'int64' else "classification"

    if(task_type == "regression") :
        return None

    # Identify column types
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'bool']).columns

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ])

    X_processed = preprocessor.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

    max_iter_dynamic = X_train.shape[0] * X_train.shape[1]

    # Final result structure
    results = {
        "regression": {},
        "classification": {}
    }


    models = {
        "Logistic Regression": LogisticRegression(max_iter=max_iter_dynamic),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(),
        "k-NN": KNeighborsClassifier(),
        "Naïve Bayes": GaussianNB(),
        "XGBoost (Gradient Boosting)": GradientBoostingClassifier(),
        "Neural Networks": MLPClassifier(max_iter=max_iter_dynamic)
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average='macro', zero_division=0)
        recall = recall_score(y_test, predictions, average='macro', zero_division=0)
        f1 = f1_score(y_test, predictions, average='macro', zero_division=0)

        results["classification"][name] = {
            "Accuracy": round(acc, 4) * 100,
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1_Score": round(f1, 4)
        }

    print(results)

    return results
