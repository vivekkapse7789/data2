import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('emails.csv')
print(df.head())
print(df.shape)
print(df.isnull().sum().sum())
print(df['Prediction'].value_counts())

sns.countplot(x=df['Prediction'])
plt.xticks([0, 1], ['Not Spam (0)', 'Spam (1)'])
plt.title('Class Distribution'); plt.show()

# Pre-processing and Split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = df.drop(['Email No.', 'Prediction'], axis=1)
y = df['Prediction']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)
print("Train:", X_train.shape, " Test:", X_test.shape)

# K-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Choose k by test accuracy over a small range
k_values = range(1, 16, 2)
k_scores = []
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    k_scores.append(accuracy_score(y_test, knn.predict(X_test)))

plt.plot(list(k_values), k_scores, marker='o')
plt.xlabel('k'); plt.ylabel('Accuracy'); plt.title('KNN accuracy vs k'); plt.grid(alpha=0.3)
plt.show()
best_k = list(k_values)[int(np.argmax(k_scores))]
print("Best k =", best_k)

knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

print("KNN Accuracy:", round(accuracy_score(y_test, y_pred_knn), 4))
print("\nClassification Report:\n", classification_report(y_test, y_pred_knn, target_names=['Not Spam', 'Spam']))

# Support Vector Machine
from sklearn.svm import SVC

svm = SVC(kernel='linear', C=1.0)
svm.fit(X_train_s, y_train)
y_pred_svm = svm.predict(X_test_s)

print("SVM Accuracy:", round(accuracy_score(y_test, y_pred_svm), 4))
print("\nClassification Report:\n", classification_report(y_test, y_pred_svm, target_names=['Not Spam', 'Spam']))

# Performance Analysis
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, pred, name in zip(axes, [y_pred_knn, y_pred_svm], [f'KNN (k={best_k})', 'SVM (linear)']):
    sns.heatmap(confusion_matrix(y_test, pred), annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Not Spam', 'Spam'], yticklabels=['Not Spam', 'Spam'])
    ax.set_title(name); ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
plt.tight_layout(); plt.show()

from sklearn.metrics import precision_score, recall_score, f1_score
comparison = pd.DataFrame({
    'Model':     [f'KNN (k={best_k})', 'SVM (linear)'],
    'Accuracy':  [accuracy_score(y_test, y_pred_knn),  accuracy_score(y_test, y_pred_svm)],
    'Precision': [precision_score(y_test, y_pred_knn), precision_score(y_test, y_pred_svm)],
    'Recall':    [recall_score(y_test, y_pred_knn),    recall_score(y_test, y_pred_svm)],
    'F1 Score':  [f1_score(y_test, y_pred_knn),        f1_score(y_test, y_pred_svm)],
})
print(comparison)
