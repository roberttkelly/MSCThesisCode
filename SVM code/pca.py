__author__ = 'angad'

import numpy as np
from sklearn.decomposition import PCA
# from sklearn.decomposition import IncrementalPCA


def fit_transform(m, dimension, images, n_components, SAVE=True, n_components_min=0):
    #m = 5000
    #dimension = 512
    size = str(m)+'-'+str(dimension)+'x'+str(dimension)
    #print "Loading images data..."
    #images = np.load('images-'+size+'.npy')
    #n_components = 40
    # batch_size = 100
    # pca = IncrementalPCA(n_components=n_components, batch_size=batch_size)

    print "Running PCA..."

    # TODO: Review the next four lines and uncomment to enable n_components_min
    #pca = PCA()
    #pca.fit(images)
    #transform_matrix = np.dot(pca.components_.T[n_components_min:n_components], images)
    #images_reduced = np.dot(images, transform_matrix.T)

    pca = PCA(n_components=n_components)
    images_reduced = pca.fit_transform(images)

    if SAVE:
        np.save('images'+str(n_components)+'-'+size, images_reduced)
        print 'Image binary with PCA saved: images'+str(n_components)+'-'+size
    return images_reduced
