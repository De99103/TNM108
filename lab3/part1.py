#p(class/ feature) = p(class) * p(feature|class) / p(feature)
# posterior probability = prior probability * likelihood / evidence / predictor prior probability

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.datasets  import make_blobs

x, y = make_blobs(100,2, centers=2, random_state=2, cluster_std=1.5)
# number of samples, number of features(dimensions), centers = How many clusters (or “classes”). Can also be array of center coordinates.
#  cluster_std = Standard deviation (spread) of each cluster. Larger values = more overlap.
# random_state = Seed for random number generator. Makes results reproducible.

plt.scatter(x[:,0], x[:,1], c=y, s=50, cmap='RdBu')
#plt.plot()
plt.show()

from sklearn.naive_bayes import GaussianNB
model= GaussianNB()
model.fit(x,y)

rng= np.random.RandomState(0)
Xnew =[-6,-14] + [14,18] * rng.rand(2000,2)
ynew= model.predict(Xnew)

plt.scatter(x[:,0], x[:,1], c=y, s=50, cmap='RdBu')
lim = plt.axis()
plt.scatter(Xnew[:,0], Xnew[:,1], c=ynew, s=20, cmap='RdBu', alpha=0.1)
plt.axis(lim)

plt.show()