# Run SVM on retina images
from sklearn import svm
from matplotlib.mlab import PCA as mlabPCA
import numpy as np
import os
from scipy.misc import imsave,imread
from sklearn.grid_search import GridSearchCV
from datetime import datetime
import cPickle
from sklearn.cross_validation import ShuffleSplit

# Define the local directory used
def load_subset(subset):
	images_labels = {}
	path_to_images = '' + subset # Image diectory here
	labels_file = '' + subset + '.txt' # Path to label file
	images_labels = {}
	with open(labels_file, 'r') as f:
		dict_labels = dict([line.strip().split() for line in f.readlines()])
	# List files in this directory
	files = os.listdir(path_to_images)
	#files = files[0:10000]

	# Create image holding structure
	images = np.zeros((len(files), 256*256*3), dtype=np.uint8)
	labels = np.zeros(len(files), dtype=np.uint8)
	for fid, file in enumerate(files):
        	if fid % 1000 == 0:
			print fid
		image = imread(path_to_images + '/' + file)
		if image.shape == (256, 256, 3):
			images[fid] = image.flatten()
			labels[fid] = int(dict_labels[file])
	return images, labels, 

pca = RandomizedPCA(n_components=n_components, whiten=True)

param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf = GridSearchCV(SVC(kernel='rbf', class_weight='auto'),param_grid)

def kappa(labels, predictions):
	labels = np.asarray(labels)
	predictions = np.asarray(predictions)
        
	ratings = np.matrix((labels, predictions)).T

        categories = int(np.amax(ratings)) + 1
        subjects = ratings.size / 2

        # Build weight matrix
        weighted = np.empty((categories, categories))
        for i in range(categories):
                for j in range(categories):
                        weighted[i, j] = abs(i - j) ** 2

        # Build observed matrix
        observed = np.zeros((categories, categories))
        distributions = np.zeros((categories, 2))
        for k in range(subjects):
                observed[ratings[k, 0], ratings[k, 1]] += 1
                distributions[ratings[k, 0], 0] += 1
                distributions[ratings[k, 1], 1] += 1

        # Normalize observed and distribution arrays
        observed = observed / subjects
        distributions = distributions / subjects

        # Build expected array
        expected = np.empty((categories, categories))
        for i in range(categories):
                for j in range(categories):
                        expected[i, j] = distributions[i, 0] * distributions[j, 1]

        # Calculate kappa
        kappa = 1.0 - (sum(sum(weighted * observed)) / sum(sum(weighted * expected)))
	return kappa

train_images, train_labels, train_files = load_subset('train')

  

param_grid = {'C':[1], 'gamma':[0.0001],}
clf = GridSearchCV(svm.SVC(kernel='rbf', class_weight='auto'), param_grid, n_jobs=1, cv=ShuffleSplit(test_size=0.20, n_iter=1, random_state=0, n=len(train_images)))
clf = clf.fit(train_images, train_labels)


with open('', 'w') as f: # Storage locationn
        cPickle.dump(clf, f)
print 'Model saved as ' + '' # File Name
