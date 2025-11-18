import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# Create simple 1D data
np.random.seed(42)
X = np.sort(5 * np.random.rand(80, 1), axis=0)
y_class = (X[:, 0] > 2.5).astype(int)         # class labels: 0 or 1
y_reg = np.sin(X[:, 0]) + 0.1 * np.random.randn(80)  # continuous values

# Train models
clf = DecisionTreeClassifier(max_depth=3)
clf.fit(X, y_class)

reg = DecisionTreeRegressor(max_depth=3)
reg.fit(X, y_reg)

# Create prediction grid
X_test = np.linspace(0, 5, 500).reshape(-1, 1)
y_pred_class = clf.predict_proba(X_test)[:, 1]  # probability for class 1
y_pred_reg = reg.predict(X_test)

# Plot
fig, axs = plt.subplots(1, 2, figsize=(10, 4))

# --- Classifier ---
axs[0].scatter(X, y_class, c=y_class, cmap="coolwarm", edgecolor="k")
axs[0].plot(X_test, y_pred_class, color="black", linewidth=2)
axs[0].set_title("Decision Tree Classifier")
axs[0].set_xlabel("X")
axs[0].set_ylabel("Class probability")

# --- Regressor ---
axs[1].scatter(X, y_reg, color="gray", label="Data")
axs[1].plot(X_test, y_pred_reg, color="red", linewidth=2, label="Prediction")
axs[1].set_title("Decision Tree Regressor")
axs[1].set_xlabel("X")
axs[1].set_ylabel("Predicted value")
axs[1].legend()

plt.tight_layout()
plt.show()
