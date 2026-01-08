import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def standardize(X):
    X = X.astype(np.float64)
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / (std + 1e-8)


def one_hot(y, num_classes):
    m = y.shape[0]
    y_oh = np.zeros((m, num_classes))
    y_oh[np.arange(m), y] = 1
    return y_oh

def softmax(z):
    exp = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)

def forward(X, W, b):
    z = np.dot(X, W) + b
    return softmax(z)

def cross_entropy(y_true, y_pred):
    epsilon = 1e-9
    return -np.mean(np.sum(y_true * np.log(y_pred + epsilon), axis=1))

def backward(X, y_true, y_pred):
    m = X.shape[0]
    dZ = y_pred - y_true
    dW = np.dot(X.T, dZ) / m
    db = np.sum(dZ, axis=0, keepdims=True) / m
    return dW, db

def predict(X, W, b):
    probs = forward(X, W, b)
    return np.argmax(probs, axis=1)


df = pd.read_csv("data.csv")
df.drop("User_ID", axis=1, inplace=True)
label_map = {
    "Negative": 0,
    "Neutral": 1,
    "Positive": 2
}
df["Performance_Impact"] = df["Performance_Impact"].map(label_map)

categorical_cols = [
    "Gender",
    "Occupation",
    "Game_Type",
    "Primary_Gaming_Time"
]

df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

X = df.drop("Performance_Impact", axis=1).astype(np.float64).values
y = df["Performance_Impact"].values

X = standardize(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


num_classes = 3
y_train_oh = one_hot(y_train, num_classes)

np.random.seed(42)

n_features = X_train.shape[1]
W = np.random.randn(n_features, num_classes) * 0.01
b = np.zeros((1, num_classes))


learning_rate = 0.01
epochs = 2000

for epoch in range(epochs):
    y_pred = forward(X_train, W, b)
    loss = cross_entropy(y_train_oh, y_pred)

    dW, db = backward(X_train, y_train_oh, y_pred)

    W -= learning_rate * dW
    b -= learning_rate * db

    if epoch % 200 == 0:
        print(f"Epoch {epoch} | Loss: {loss:.4f}")


y_test_pred = predict(X_test, W, b)

print("\nAccuracy:", accuracy_score(y_test, y_test_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred))
