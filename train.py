import seaborn as sns
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
penguins = sns.load_dataset("penguins")

# Drop missing values
penguins = penguins.dropna()

# Features and target
X = penguins.drop("species", axis=1)
y = penguins["species"]

# Identify column types
num_features = X.select_dtypes(include="number").columns.to_list()
cat_features = X.select_dtypes(exclude="number").columns.to_list()

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features),
        (
            "cat",
            OneHotEncoder(
                drop="first",
                sparse_output=False,
                dtype=np.float64,
                handle_unknown="ignore",
            ),
            cat_features,
        ),
    ]
)

# Full pipeline
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression()),
    ]
)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train
pipeline.fit(X_train, y_train)

# Predict
y_pred = pipeline.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
import pickle

with open(file="./pipeline.pkl", mode="wb") as file:
    pickle.dump(obj=pipeline, file=file)
