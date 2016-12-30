__author__ = 'angad'

from os import walk, path
import pandas as pd
import numpy as np
from scipy import misc
import cv2
from sklearn.preprocessing import scale


def is_inverted_vert(image, filename):
    # Output: True or False based on whether the image is 'inverted'
    # as defined by the Kaggle contest on Diabetic Retinopath
    # This function is based on Ravi's idea of testing
    # to see if the bright spot is on the left or the
    # right of the vertical center line
    # Input: Expects images with equalized histograms as inpu
    (r_max, c_max) = image.shape
    maxval = image.max()
    (t, image) = cv2.threshold(image, maxval-10, maxval, cv2.THRESH_BINARY)
    (r, c) = image.nonzero()
    if 'right' in filename:
        if c.mean() > (c_max/2):
            inverted = False
        else:
            inverted = True
    elif 'left' in filename:
        if c.mean() < (c_max/2):
            inverted = False
        else:
            inverted = True
    else:
        inverted = None
    return inverted


def get_image_filenames(dir):
    # Output: List of filenames with .jpeg extension
    # Input: Relative directory path to search for filenames
    filenames = []
    for (dp, dn, fs) in walk(dir):
        for f in fs:
            if '.jpeg' in f:
                filenames += [f]
    return filenames


def extract(m, dimension, FIX_INVERTED=True, FIX_RIGHT_LEFT=True, SAVE=True):
    #m = 5000
    #dimension = 512
    size = str(m)+'-'+str(dimension)+'x'+str(dimension)
    dir = 'processed/data/train-'+size+'/'
    filenames = get_image_filenames(dir)

    trainLabels = pd.read_csv('data/trainLabels.csv', index_col=0)

    images = []
    y = []
    inverted = 0
    right = 0
    print "Importing images..."
    for (i, filename) in enumerate(filenames):
        #print i
        image = misc.imread(path.join(dir, filename), flatten=1)
        if FIX_INVERTED:
            if is_inverted_vert(image, filename):
                image = np.fliplr(np.flipud(image))
                inverted += 1
        if FIX_RIGHT_LEFT:
            if 'right' in filename:
                image = np.fliplr(image)
                right += 1
        (r, c) = image.shape
        image = image.reshape(r*c)
        images += [image]
        y_val = trainLabels.loc[filename.split('.')[0]][0]
        y += [y_val]
    print "%d inverted images fixed." % inverted
    print "%d right images flipped." % right
    print "Converting images to a numpy array..."
    images = np.array(images)

    # Feature scaling
    print "Scaling images..."
    images = scale(images, axis=0, with_mean=True, with_std=True, copy=False)
    images = scale(images, axis=1, with_mean=True, with_std=True, copy=False)

    print "Converting y to a numpy array..."
    y = np.array(y)
    if SAVE:
        print "Saving images"
        np.save('images-'+size, images)
        print "Saving y"
        np.save('y-'+size, y)
        print 'Image binary and Y values saved: images-'+size+' and y-'+size
    return (images, y)
