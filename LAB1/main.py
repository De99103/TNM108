# Dependencies
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt


# Load the train and test datasets to create two DataFrames
train_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
train = pd.read_csv(train_url)
test_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv"
test = pd.read_csv(test_url)

# 1) Imputera numeriskt robust för Pandas 1.5/2.x
train = train.fillna(train.mean(numeric_only=True))
test  = test.fillna(test.mean(numeric_only=True))

# 2) Droppa icke-informativa kolumner
train = train.drop(['Name','Ticket','Cabin','Embarked'], axis=1)
test  = test.drop(['Name','Ticket','Cabin','Embarked'], axis=1)

# 3) LabelEncoder: fit på train, transformera båda
from sklearn.preprocessing import LabelEncoder, StandardScaler
le = LabelEncoder()
le.fit(train['Sex'])
train['Sex'] = le.transform(train['Sex'])
test['Sex']  = le.transform(test['Sex'])

# 4) Bygg X/y utan PassengerId
y = train['Survived'].to_numpy()
X = train.drop(['Survived','PassengerId'], axis=1).astype(float).to_numpy()

# 5) Skala features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 6) KMeans med stabila inställningar
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, algorithm='lloyd', max_iter=600, n_init=50, random_state=42)
labels = kmeans.fit_predict(X_scaled)

# 7) Mappa klusteretiketter till klassetiketter och räkna "accuracy"
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
cm = confusion_matrix(y, labels)
r, c = linear_sum_assignment(-cm)
mapping = {c_i: r_i for r_i, c_i in zip(r, c)}
y_pred = np.vectorize(mapping.get)(labels)
print("Accuracy:", (y_pred == y).mean())
