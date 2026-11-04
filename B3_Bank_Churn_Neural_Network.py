import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
tf.random.set_seed(42); np.random.seed(42)
print("TensorFlow version:", tf.__version__)

# 1. Read the Dataset
df = pd.read_csv('Churn_Modelling.csv')
print(df.head())
print(df.shape)
df.info()
print(df['Exited'].value_counts())

# 2. Features, Target and Train/Test Split
# Drop identifiers that carry no predictive information
X = df.drop(['RowNumber', 'CustomerId', 'Surname', 'Exited'], axis=1)
y = df['Exited']

# Encode categorical columns
X = pd.get_dummies(X, columns=['Geography', 'Gender'], drop_first=True, dtype=int)
print(X.head())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Train:", X_train.shape, " Test:", X_test.shape)

# 3. Normalize the Data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)
print(X_train[:3].round(3))

# 4a. Baseline Model
model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(6, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(X_train, y_train, validation_split=0.2, epochs=20, batch_size=32, verbose=2)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
y_pred_base = (model.predict(X_test, verbose=0) > 0.5).astype(int).ravel()
print("Baseline accuracy:", round(accuracy_score(y_test, y_pred_base), 4))
print(classification_report(y_test, y_pred_base, target_names=['Stayed', 'Exited']))

# 4b. Points of Improvement
# Deeper/wider network, Adam optimizer, Dropout, Early stopping, Class weights
from sklearn.utils.class_weight import compute_class_weight
cw = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train)
class_weight = {0: cw[0], 1: cw[1]}
print("Class weights:", class_weight)

improved = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(16, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')
])
improved.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                 loss='binary_crossentropy', metrics=['accuracy'])
improved.summary()

early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history2 = improved.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=32,
                        class_weight=class_weight, callbacks=[early], verbose=2)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].plot(history2.history['loss'], label='train'); axes[0].plot(history2.history['val_loss'], label='validation')
axes[0].set_title('Loss'); axes[0].set_xlabel('Epoch'); axes[0].legend()
axes[1].plot(history2.history['accuracy'], label='train'); axes[1].plot(history2.history['val_accuracy'], label='validation')
axes[1].set_title('Accuracy'); axes[1].set_xlabel('Epoch'); axes[1].legend()
plt.tight_layout(); plt.show()

# 5. Accuracy Score and Confusion Matrix
y_pred = (improved.predict(X_test, verbose=0) > 0.5).astype(int).ravel()
print("Improved model accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['Stayed', 'Exited']))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, pred, name in zip(axes, [y_pred_base, y_pred], ['Baseline model', 'Improved model']):
    sns.heatmap(confusion_matrix(y_test, pred), annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Stayed', 'Exited'], yticklabels=['Stayed', 'Exited'])
    ax.set_title(name); ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
plt.tight_layout(); plt.show()
