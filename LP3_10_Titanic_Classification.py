import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

import urllib.request

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
urllib.request.urlretrieve(url, "train.csv")

print("Dataset downloaded successfully!")

# Step 2: Load and inspect data
df = pd.read_csv("train.csv")

print("Dataset shape:", df.shape)
print(df.head())
print("\nData types:\n", df.dtypes)
print("\nMissing values:\n", df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())
print("\nSurvival distribution:\n", df["Survived"].value_counts())

# Step 3: Visualize survival count
sns.countplot(data=df, x="Survived")
plt.title("Survival Count")
plt.savefig("survival_count.png")
plt.show()

# Survival rate by gender
sns.barplot(data=df, x="Sex", y="Survived")
plt.title("Survival Rate by Gender")
plt.savefig("survival_by_gender.png")
plt.show()

# Survival rate by passenger class
sns.barplot(data=df, x="Pclass", y="Survived")
plt.title("Survival Rate by Passenger Class")
plt.savefig("survival_by_class.png")
plt.show()

# Age distribution by survival
sns.histplot(data=df, x="Age", hue="Survived", bins=20, kde=True)
plt.title("Age Distribution by Survival")
plt.savefig("age_distribution.png")
plt.show()

# Step 4: Clean data and create useful features
df = df.drop_duplicates().copy()

df["Title"] = df["Name"].str.extract(r",\s*([^.]*)\.", expand=False)
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

X = df.drop(columns=["Survived", "PassengerId", "Name", "Ticket", "Cabin"])
y = df["Survived"]

num = ["Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize", "IsAlone"]
cat = ["Sex", "Embarked", "Title"]

print("Features:", X.columns.tolist())
print("Target distribution:\n", y.value_counts())

# Step 5: Split data and prepare preprocessing pipelines
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

prep = ColumnTransformer([
    ("num", num_pipe, num),
    ("cat", cat_pipe, cat)
])

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# Step 6: Train classification models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC()
}

for name in models:
    models[name] = Pipeline([("prep", prep), ("clf", models[name])])
    models[name].fit(X_train, y_train)

print("All five models trained successfully.")

# Step 7: Calculate classification metrics
results = []

for name, model in models.items():
    pred = model.predict(X_test)
    results.append([
        name,
        accuracy_score(y_test, pred),
        precision_score(y_test, pred, zero_division=0),
        recall_score(y_test, pred, zero_division=0),
        f1_score(y_test, pred, zero_division=0)
    ])

scores = pd.DataFrame(results, columns=[
    "Model", "Accuracy", "Precision", "Recall", "F1-score"
])

print(scores.round(3))

best_name = scores.loc[scores["F1-score"].idxmax(), "Model"]
best_model = models[best_name]
print("\nBest model based on F1-score:", best_name)

# Step 8: Evaluate the selected model
pred = best_model.predict(X_test)

print("Classification Report:\n")
print(classification_report(y_test, pred, zero_division=0))

sns.heatmap(confusion_matrix(y_test, pred), annot=True,
            fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Survival")
plt.ylabel("Actual Survival")
plt.savefig("confusion_matrix.png")
plt.show()

# Step 9: Compare model performance
scores.plot(x="Model", y=["Accuracy", "Precision", "Recall", "F1-score"],
            kind="bar", figsize=(10, 5))
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig("model_comparison.png")
plt.show()

# Step 10: Identify important passenger features
rf = models["Random Forest"]
importance = rf.named_steps["clf"].feature_importances_
features = rf.named_steps["prep"].get_feature_names_out()

imp = pd.Series(importance, index=features).sort_values(ascending=False).head(10)
imp.sort_values().plot(kind="barh", figsize=(8, 5))
plt.title("Random Forest Feature Importance")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()

# Step 11: Predict survival for a sample passenger
sample = X_test.iloc[[0]]
prediction = best_model.predict(sample)[0]

print("Passenger details:\n", sample)
print("Predicted survival:", "Survived" if prediction == 1 else "Did not survive")
print("Actual survival:", "Survived" if y_test.iloc[0] == 1 else "Did not survive")
