__author__ = 'angad'

import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


def predict(m, dimension, images_reduced, y, SAVE=True):
    #m = 5000
    #dimension = 512
    size = str(m)+'-'+str(dimension)+'x'+str(dimension)
    #print "Loading images data"
    #images_red = np.load('images40-'+size+'.npy')
    #print "Loading y data"
    #y = np.load('y-'+size+'.npy')

    #clf = SVC(class_weight='auto')
    clf = LogisticRegression(class_weight='auto', C=10)
    clf.verbose=1
    data_split = int(m/2)
    #print "Running SVM..."
    print "Running Logistic Regression..."
    (unique_values, counts) = np.unique(y, return_counts=True)
    weights = 1 - (counts.astype('float')/m)
    weight_dict = {}
    for i, weight in enumerate(weights):
        weight_dict[i] = weight
    sample_weight = []
    for i in range(m):
        sample_weight += [weight_dict[y[i]]]

    #print y
    #print sample_weight

    #clf.fit(images_reduced[:data_split],y[:data_split], sample_weight=sample_weight[:data_split])
    clf.fit(images_reduced[:data_split],y[:data_split])
    score = clf.score(images_reduced[data_split:], y[data_split:])
    #print 'Done. Score:', score

    predictions = clf.predict(images_reduced)

    if SAVE:
        np.savetxt('predictions-'+size+'.csv', np.array((clf.predict(images_reduced), y)).T, delimiter=',', fmt='%d', header='Prediction, y')

    return (predictions, score)
