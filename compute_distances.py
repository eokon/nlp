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
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn import preprocessing


#images_matrix = np.load('function_images_matrix.npy')
#word2vec_matrix = np.load('function_word2vec_matrix.npy')

def get_features(dir1, dir2, paraphrase):
    a = open('features/' + paraphrase + '/' + dir1 + '/' + dir2, 'rb')
    #a = open('features/bug' + dir1 + '/' + dir2, 'rb')
    b = pickle._Unpickler(a)
    b.encoding = 'latin1'
    c = b.load()
    return c

def get_image_matrix(paraphrase):
    all_dirs = os.listdir('features/' + paraphrase)
    #print(all_dirs)
    word_images = []

    for d in all_dirs:
        temp = [f for f in listdir('features/' + paraphrase + '/' + d) if isfile(join('features/' + paraphrase + '/' + d, f))]
        word_images.append(temp)

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
                    v = get_features(all_dirs[i], word_images[i][m], paraphrase)
                    if np.isfinite(v).all() == False: # Skip iteration if numpy.nan, numpy.inf or -numpy.inf values found
                        continue
                    for n in range(0, len(word_images[j])):
                        temp_similarities = []
                        w = get_features(all_dirs[j], word_images[j][n], paraphrase)
                        if np.isfinite(w).all() == False: # Skip iteration if numpy.nan, numpy.inf or -numpy.inf values found
                            continue
                        cos_similarity = 1 - spatial.distance.cosine(v, w)
                        temp_similarities.append(cos_similarity)
                        max_of_temps = max(temp_similarities)
                        all_max.append(max_of_temps)

                curr_iter = curr_iter + 1
                print('Progess: ' + str(curr_iter) + '/' + str(total_iters))
            if len(all_max) > 0:
                distance_matrix[i][j] = sum(all_max)/len(all_max)
            else:
                distance_matrix[i][j] = 0
    #print(distance_matrix)
    np.save('affinity_matrices/' + paraphrase + '_image_matrix', distance_matrix)

def get_word2vec_matrix(paraphrase):
    #model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    #pickle.dump(model, open('word2vec_model', 'wb'))
    model = pickle.load(open('word2vec_model', 'rb'))

    all_dirs = os.listdir('features/' + paraphrase)
    num_paraphrases = len(all_dirs)

    word2vec_similarities = np.zeros((num_paraphrases, num_paraphrases))
    progress = 1

    not_in_vocab = set()
    sum_of_included = 0
    num_included_in_vocab = 0
    average_included = 0

    for i in range(0, num_paraphrases):
        for j in range(0, num_paraphrases):
            if all_dirs[i] in model.wv.vocab and all_dirs[j] in model.wv.vocab: #Double check how I handle this
                word2vec_similarities[i,j] = model.similarity(all_dirs[i], all_dirs[j])
            else:
                not_in_vocab.add((i,j))
            print('Progess: ' + str(progress) + '/' + str(num_paraphrases ** 2))
            progress = progress + 1

    for i in range(0, num_paraphrases):
        for j in range(0, num_paraphrases):
            if (i,j) not in not_in_vocab:
                sum_of_included = sum_of_included + word2vec_similarities[i,j]
                num_included_in_vocab = num_included_in_vocab + 1

    avg_included = sum_of_included/num_included_in_vocab

    for point in not_in_vocab:
        word2vec_similarities[point[0], point[1]] = avg_included

    print(sum_of_included)
    print(num_included_in_vocab)
    print(avg_included)

    np.save('affinity_matrices/' + paraphrase + '_word2vec_matrix', word2vec_similarities)

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
    distance_matrix = np.load('computed_distances.npy')
    #print(distance_matrix)
    db = DBSCAN(eps=.1, min_samples=3).fit_predict(distance_matrix)
    #print(db.labels_)
    #print(db)

def find_k_pca(distance_matrix):
    k = 1
    X_scaled = preprocessing.scale(distance_matrix)
    pca = PCA()
    pca.fit(X_scaled)
    explained = pca.explained_variance_ratio_.cumsum()
    #print(explained)
    for i in range(0, len(explained)):
        if explained[i] > 0.8:
            k = i + 1
            break
    return k

def find_k_silhouette(distance_matrix):
    k = 2
    max_score = -1

    for i in range(2, min(11, len(distance_matrix))):
        clusters = SpectralClustering(n_clusters=i, affinity='precomputed', n_init=25,
                                      assign_labels='discretize').fit_predict(distance_matrix)
        silouette_score = metrics.silhouette_score(distance_matrix, clusters)
        if silouette_score > max_score:
            max_Score = silouette_score
            k = i
    return k

def spectral_clustering(distance_matrix):
    k = find_k_pca(distance_matrix)
    print(k)
    #print(find_k_silhouette(distance_matrix))
    clusters = SpectralClustering(n_clusters=k, affinity='precomputed', n_init=25,
                                  assign_labels='discretize').fit_predict(distance_matrix)
    return clusters

if __name__== '__main__':
    print('Hello World!')
    get_image_matrix('break')
    #c = spectral_clustering(np.load('affinity_matrices/note_image_matrix.npy'))
    #print(c)
    #get_word2vec_matrix('break')
    #get_image_matrix('break')
