import os
import pickle
import PIL
from PIL import Image
import gzip
import numpy as np
import matplotlib.pyplot as plt
from scipy import spatial
from os import listdir
from os.path import isfile, join
from os import walk
import scipy
import pylab
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
import gensim
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering

bug_mappings = {
    'failure': 0,
    'mosquito': 1,
    'cockroach': 2,
    'parasite': 3,
    'bacterium': 4,
    'wire': 5,
    'pest': 6,
    'glitch': 7,
    'virus': 8,
    'error': 9,
    'microphone': 10,
    'microbe': 11,
    'insect': 12,
    'beetle': 13,
    'malfunction': 14,
    'informer': 15,
    'tracker': 16,
    'mistake': 17,
    'snitch': 18,
    'fault': 19,
}


bug_list = [
    'failure',
    'mosquito',
    'cockroach',
    'parasite',
    'bacterium',
    'wire',
    'pest',
    'glitch',
    'virus',
    'error',
    'microphone',
    'microbe',
    'insect',
    'beetle',
    'malfunction',
    'informer',
    'tracker',
    'mistake',
    'snitch',
    'fault',
]

images_matrix = np.load('computed_distances.npy')
word2vec_matrix = np.load('word2vec_similarities.npy')

def get_features(dir1, dir2):
    a = open('features/' + dir1 + '/' + dir2, 'rb')
    b = pickle._Unpickler(a)
    b.encoding = 'latin1'
    c = b.load()
    return c

def get_image_similarities():

    all_dirs = os.listdir('features')
    print(all_dirs)
    word_images = []

    for d in all_dirs:
        temp = [f for f in listdir('features/' + d) if isfile(join('features/' + d, f))]
        word_images.append(temp)

    print('word_images')

    num_words = len(word_images)
    distance_matrix = np.zeros((num_words, num_words))
    progress = np.zeros((num_words, num_words))

    curr_iter = 0
    total_iters = num_words ** 2


    for i in range(0, num_words):
        for j in range(0, num_words):
            if progress[i][j] == 0:
                progress[i][j] = 1
                all_max = []

                for m in range(0, len(word_images[i])):
                    #word_images[i]
                    v = get_features(all_dirs[i], word_images[i][m])
                    if np.isfinite(v).all() == False: # Skip iteration if numpy.nan, numpy.inf or -numpy.inf values found
                        continue
                    for n in range(0, len(word_images[j])):
                        temp_similarities = []
                        w = get_features(all_dirs[j], word_images[j][n])
                        if np.isfinite(w).all() == False: # Skip iteration if numpy.nan, numpy.inf or -numpy.inf values found
                            continue
                        cos_similarity = 1 - spatial.distance.cosine(v, w)
                        temp_similarities.append(cos_similarity)
                        max_of_temps = max(temp_similarities)
                        all_max.append(max_of_temps)

                curr_iter = curr_iter + 1
                print('Progess: ' + str(curr_iter) + '/' + str(total_iters))

            distance_matrix[i][j] = sum(all_max)/len(all_max)
    print(distance_matrix)
    np.save('computed_distances', distance_matrix)

def get_word2vec_similarities():
    model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    word2vec_similarities = np.zeros((20, 20))
    progress = 1
    for i in range(0, len(bug_list)):
        for j in range(0, len(bug_list)):
            word2vec_similarities[i, j] = model.similarity(bug_list[i], bug_list[j])
            print('Progess: ' + str(progress) + '/' + str(len(bug_list) ** 2))
            progress = progress + 1
    np.save('word2vec_similarities', word2vec_similarities)
    return word2vec_similarities

def hierarchical_clustering_dendrogram():
    distance_matrix = np.load('computed_distances.npy')
    Z = sch.linkage(distance_matrix, method='centroid')


    plt.figure(figsize=(25, 10))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    dendrogram(
        Z,
        leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=8.,  # font size for the x axis labels
    )
    plt.show()

def DBSCAN_clustering():
    distance_matrix = np.load('coputed_distances.npy')
    #print(distance_matrix)
    db = DBSCAN(eps=.1, min_samples=3).fit_predict(distance_matrix)
    #print(db.labels_)
    #print(db)


def spectral_clustering(distance_matrix):
    #distance_matrix = np.load('computed_distances.npy')
    return SpectralClustering(affinity='precomputed', n_init = 25, assign_labels='discretize').fit_predict(distance_matrix)

    #get_distance_matrix()
    #hierarchical_clustering_dendrogram()

if __name__== '__main__':
    print('Hello World!')
    #get_word2vec_similarities()
    #print(get_image_similarities())
