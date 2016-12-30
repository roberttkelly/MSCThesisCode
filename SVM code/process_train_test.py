__author__ = 'angad'

import pre_process
import pca
import svm
from quadratic_weighted_kappa import quadratic_weighted_kappa


def do(m, dimension, n_components, FIX_INVERTED=True, FIX_RIGHT_LEFT=True, SAVE=True, n_components_min=0):
    #m = 1000
    #dimension = 256
    (images, y) = pre_process.extract(m, dimension, FIX_INVERTED, FIX_RIGHT_LEFT, SAVE)

    #n_components = 100
    #images_reduced = pca.fit_transform(m, dimension, images, n_components, SAVE, n_components_min)

    #(pred, svm_score) = svm.predict(m, dimension, images_reduced, y, SAVE)
    (pred, svm_score) = svm.predict(m, dimension, images, y, SAVE)

    kappa_score_train = quadratic_weighted_kappa(pred[:m/2], y[:m/2], min_rating=0, max_rating=4)
    kappa_score_test = quadratic_weighted_kappa(pred[m/2:], y[m/2:], min_rating=0, max_rating=4)
    kappa_score_all = quadratic_weighted_kappa(pred, y, min_rating=0, max_rating=4)

    print "kappa score for train: ", kappa_score_train
    print "kappa score for test: ", kappa_score_test
    print "kappa score for all data: ", kappa_score_all
    print "svm score: ", svm_score

if __name__ == '__main__':
    m = 20000
    dimension = 256
    n_components = 40
    n_components_min = 5
    FIX_INVERTED = True
    FIX_RIGHT_LEFT = True
    SAVE=False
    do(m, dimension, n_components, FIX_INVERTED, FIX_RIGHT_LEFT, SAVE)
