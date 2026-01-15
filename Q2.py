from Q1 import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import numpy as np

x_train,x_val,x_test,y_train,y_val,y_test,y_train_full_encoded,y_test_encoded = preprocessing()

#Flatten the images
X_train_flat = x_train.reshape(x_train.shape[0], -1)
X_test_flat = x_test.reshape(x_test.shape[0], -1)

# Convert the labels to integers
if y_train.ndim > 1:
    y_train_integers = np.argmax(y_train, axis=1)
else:
    y_train_integers = y_train

# Check y_test separately
if y_test.ndim > 1:
    y_test_integers = np.argmax(y_test, axis=1)
else:
    y_test_integers = y_test

#Train the model
print("Training Logistic Regression... (this may take 30-60 seconds)")
log_reg = LogisticRegression(solver='lbfgs', max_iter=1000)
log_reg.fit(X_train_flat, y_train_integers)
print("Training Complete.")

y_pred_test = log_reg.predict(X_test_flat)
print("--- Model Performance ---")
print(f"Accuracy: {accuracy_score(y_test_integers, y_pred_test):.4f}")
print("\nClassification Report:\n")
print(classification_report(y_test_integers, y_pred_test))