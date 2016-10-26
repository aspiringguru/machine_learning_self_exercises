import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from sklearn.datasets import make_blobs
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cross_validation import train_test_split


n_samples = 50000
n_bins = 3  # use 3 bins for calibration_curve as we have 3 clusters here

# Generate 3 blobs with 2 classes where the second blob contains
# half positive samples and half negative samples. Probability in this
# blob is therefore 0.5.
centers = [(-5, -5), (0, 0), (5, 5)]
X, y = make_blobs(n_samples=n_samples, n_features=2, cluster_std=1.0, centers=centers, shuffle=False, random_state=42)
#http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html

print "type(X)", type(X), "X.shape", X.shape
print "type(y)=", type(y), "y.shape=", y.shape

#convert first half of y values to 0, second half of y values to 1.
y[:n_samples // 2] = 0
y[n_samples // 2:] = 1

sample_weight = np.random.RandomState(42).rand(y.shape[0])

# split train, test for calibration
X_train,X_test,y_train,y_test,sw_train,sw_test = train_test_split(X, y, sample_weight, test_size=0.9, random_state=42)
#NB: this is splitting 3 THREE input variables between train/test.

# Gaussian Naive-Bayes with no calibration
clf = GaussianNB()
#http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
clf.fit(X_train, y_train)  # GaussianNB itself does not support sample-weights
prob_pos_clf = clf.predict_proba(X_test)[:, 1]
#Return probability estimates for the test vector X.
#NB probability estimates <> classification [performed by predict(X)
print "type(prob_pos_clf)=", type(prob_pos_clf), "prob_pos_clf.shape=", prob_pos_clf.shape

# Gaussian Naive-Bayes with isotonic calibration
clf_isotonic = CalibratedClassifierCV(clf, cv=2, method='isotonic')
#http://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html
#Probability calibration with isotonic regression or sigmoid.
clf_isotonic.fit(X_train, y_train, sw_train)
#Fit the calibrated model
prob_pos_isotonic = clf_isotonic.predict_proba(X_test)[:, 1]
#Posterior probabilities of classification
print "type(prob_pos_isotonic)=", type(prob_pos_isotonic), "prob_pos_isotonic.shape=", prob_pos_isotonic.shape

# Gaussian Naive-Bayes with sigmoid calibration
clf_sigmoid = CalibratedClassifierCV(clf, cv=2, method='sigmoid')
clf_sigmoid.fit(X_train, y_train, sw_train)
prob_pos_sigmoid = clf_sigmoid.predict_proba(X_test)[:, 1]
print "type(prob_pos_sigmoid)=", type(prob_pos_sigmoid), "prob_pos_sigmoid.shape=", prob_pos_sigmoid.shape


print("Brier scores: (the smaller the better)")

clf_score = brier_score_loss(y_test, prob_pos_clf, sw_test)
#http://scikit-learn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html
#The smaller the Brier score, the better, hence the naming with 'loss'.
#The Brier score is appropriate for binary and categorical outcomes that can be structured as true or false,
# but is inappropriate for ordinal variables which can take on three or more values
#brier_score_loss(y_true, y_prob, sample_weight=None, pos_label=None)[source]
#
print("brier_score_loss prob_pos_clf: No calibration: %1.3f" % clf_score)#format to 3 decimal places.

clf_isotonic_score = brier_score_loss(y_test, prob_pos_isotonic, sw_test)
print("brier_score_brier_score_loss With isotonic calibration: %1.3f" % clf_isotonic_score)

clf_sigmoid_score = brier_score_loss(y_test, prob_pos_sigmoid, sw_test)
print("brier_score_loss With sigmoid calibration: %1.3f" % clf_sigmoid_score)

###############################################################################
# Plot the data and the predicted probabilities
plt.figure()
y_unique = np.unique(y)
colors = cm.rainbow(np.linspace(0.0, 1.0, y_unique.size))
for this_y, color in zip(y_unique, colors):
    this_X = X_train[y_train == this_y]
    this_sw = sw_train[y_train == this_y]
    plt.scatter(this_X[:, 0], this_X[:, 1], s=this_sw * 50, c=color, alpha=0.5,
                label="Class %s" % this_y)
plt.legend(loc="best")
plt.title("Data")

plt.figure()
order = np.lexsort((prob_pos_clf, ))
plt.plot(prob_pos_clf[order], 'r', label='No calibration (%1.3f)' % clf_score)
plt.plot(prob_pos_isotonic[order], 'g', linewidth=3,
         label='Isotonic calibration (%1.3f)' % clf_isotonic_score)
plt.plot(prob_pos_sigmoid[order], 'b', linewidth=3,
         label='Sigmoid calibration (%1.3f)' % clf_sigmoid_score)
plt.plot(np.linspace(0, y_test.size, 51)[1::2],
         y_test[order].reshape(25, -1).mean(1),
         'k', linewidth=3, label=r'Empirical')
plt.ylim([-0.05, 1.05])
plt.xlabel("Instances sorted according to predicted probability "
           "(uncalibrated GNB)")
plt.ylabel("P(y=1)")
plt.legend(loc="upper left")
plt.title("Gaussian naive Bayes probabilities")

plt.show()