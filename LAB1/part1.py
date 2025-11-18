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

# print("***** Train_Set *****")
# print(train.head())
# print("\n")
# print("***** Test_Set *****")
#print(test.head())


# Fill missing values with mean column values in the train set
train.fillna(train.select_dtypes(include=[np.number]).mean(), inplace=True)
# Fill missing values with mean column values in the test set
test.fillna(test.select_dtypes(include=[np.number]).mean(), inplace=True)

# Dropping non-relevant features 
train = train.drop(['Name','Ticket', 'Cabin','Embarked'], axis=1)
test = test.drop(['Name','Ticket', 'Cabin','Embarked'], axis=1)

# Label encoding
#from sklearn.preprocessing import LabelEncoder, StandardScaler
le = LabelEncoder()
le.fit(train['Sex'])
train['Sex'] = le.transform(train['Sex'])
test['Sex']  = le.transform(test['Sex'])

X = np.array(train.drop(['Survived'], axis=1).astype(float)) #all info BUT survived
y = np.array(train['Survived'])                              #no info but survived
# Scale the data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# --- Elbow Method to determine optimal k ---
inertia_values = []
k_values = range(1, 11)  # Try k = 1 to 10

for k in k_values:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    inertia_values.append(km.inertia_)

# # Plot elbow curve
plt.figure(figsize=(8, 5))
plt.plot(k_values, inertia_values, 'bo-', linewidth=2)
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia (Sum of squared distances)')
plt.title('Elbow Method for Optimal k')
plt.show()

# --- Choose the optimal k manually after checking the plot ---
optimal_k = 2  # example; pick based on the elbow in your plot

# Final KMeans with chosen k
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans.fit(X_scaled)
#print(kmeans.cluster_centers_) 
#print(kmeans.labels_)
# print(kmeans.inertia_)
# print(kmeans.n_iter_)

KMeans(algorithm='lloyd', copy_x=True, init='k-means++', max_iter=600,
    n_clusters=2, n_init=10, random_state=None, tol=0.0001, verbose=0)
correct = 0

for i in range(len(X_scaled)):
    predict_me = np.array(X_scaled[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = kmeans.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1
print(correct/len(X_scaled))

# Analyze feature relevance for survival
import scipy.stats as stats
# Combine train and test for consistent feature analysis
data = train.copy()

# # List of features to analyze
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']

# # Encode categorical features for correlation analysis
data_encoded = data.copy()
data_encoded['Sex'] = LabelEncoder().fit_transform(data_encoded['Sex'])

#if data_encoded['Embarked'].isnull().any():
#   data_encoded['Embarked'].fillna(data_encoded['Embarked'].mode()[0], inplace=True)
#   data_encoded['Embarked'] = LabelEncoder().fit_transform(data_encoded['Embarked'])

# # Correlation with survival
correlations = data_encoded[features + ['Survived']].corr()['Survived'].sort_values(ascending=False)
print("Correlation with Survived:\n", correlations)

# # Visualize relationships
for feature in features:
    plt.figure(figsize=(6, 4))
    if data[feature].dtype == 'object':
        sns.barplot(x=feature, y='Survived', data=data)
    else:
        sns.histplot(data, x=feature, hue='Survived', multiple='stack', bins=20)
    plt.title(f'Survival vs {feature}')
    plt.show()
