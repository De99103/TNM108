# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import fetch_olivetti_faces
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.utils.validation import check_random_state
from sklearn.tree import DecisionTreeRegressor

# Load the faces datasets
try:
    data, targets = fetch_olivetti_faces(return_X_y=True, download_if_missing=True)
except Exception as err:
    print("Could not fetch Olivetti faces dataset:", err)
    print("Ensure you have internet access or the dataset cached locally.")
    raise

train = data[targets < 30]
test = data[targets >= 30]  # Test on independent people

# Test on a subset of people
n_faces = 5
rng = check_random_state(4)
face_ids = rng.randint(test.shape[0], size=(n_faces,))
test = test[face_ids, :]

n_pixels = data.shape[1]
# Upper half of the faces
X_train = train[:, : (n_pixels + 1) // 2]
# Lower half of the faces
y_train = train[:, n_pixels // 2 :]
X_test = test[:, : (n_pixels + 1) // 2]
y_test = test[:, n_pixels // 2 :]

# Fit estimators
ESTIMATORS = {
    "Extra trees": ExtraTreesRegressor(
        n_estimators=10, max_features=32, random_state=0
    ),
    "K-nn": KNeighborsRegressor(),
    "Linear regression": LinearRegression(),
    "Ridge": RidgeCV(),
    
    # (a) Decision tree: max_depth=10, max_features=50
    "DT depth=10, feat=50": DecisionTreeRegressor(
        max_depth=10, max_features=50, random_state=0
    ),
    # (b) Decision tree: max_depth=20, max_features=50
    "DT depth=20, feat=50": DecisionTreeRegressor(
        max_depth=20, max_features=100, random_state=0
    ),
    # (c) Decision tree: max_depth=20, max_features=25
    "DT depth=20, feat=25": DecisionTreeRegressor(
        max_depth=20, max_features=25, random_state=0
    ),
    # (d) Random forest: max_depth=10, max_features=50
    "RF depth=10, feat=50": RandomForestRegressor(
        n_estimators=10, max_depth=10, max_features=50, random_state=0
    ),
    # (e) Random forest: max_depth=20, max_features=50
    "RF depth=20, feat=50": RandomForestRegressor(
        n_estimators=10, max_depth=20, max_features=50, random_state=0
    ),
    # (f) Random forest: max_depth=20, max_features=25
    "RF depth=20, feat=25": RandomForestRegressor(
        n_estimators=10, max_depth=20, max_features=25, random_state=0
    ),
}

y_test_predict = dict()
print("Training models...")
for name, estimator in ESTIMATORS.items():
    print(f"  Training {name}...")
    estimator.fit(X_train, y_train)
    y_test_predict[name] = estimator.predict(X_test)
print("Training complete!")

# Plot the completed faces
image_shape = (64, 64)

n_cols = 1 + len(ESTIMATORS)
plt.figure(figsize=(2.0 * n_cols, 2.26 * n_faces))
plt.suptitle("Face completion with multi-output estimators", size=16)

# Make subplot titles much smaller globally
plt.rcParams['axes.titlesize'] = 6

for i in range(n_faces):
    true_face = np.hstack((X_test[i], y_test[i]))

    if i:
        sub = plt.subplot(n_faces, n_cols, i * n_cols + 1)
    else:
        sub = plt.subplot(n_faces, n_cols, i * n_cols + 1, title="true faces")

    sub.axis("off")
    sub.imshow(
        true_face.reshape(image_shape), cmap='gray', interpolation="nearest"
    )

    for j, est in enumerate(sorted(ESTIMATORS)):
        completed_face = np.hstack((X_test[i], y_test_predict[est][i]))

        if i:
            sub = plt.subplot(n_faces, n_cols, i * n_cols + 2 + j)
        else:
            sub = plt.subplot(n_faces, n_cols, i * n_cols + 2 + j, title=est)

        sub.axis("off")
        sub.imshow(
            completed_face.reshape(image_shape),
            cmap='gray',
            interpolation="nearest",
        )

plt.tight_layout()
plt.show()