import os
import pickle
import PIL
from PIL import Image
import gzip
import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy import spatial
from os import listdir
from os.path import isfile, join
from os import walk
import scipy
import pylab
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import dendrogram
import bcubed
from nltk.corpus import wordnet as wn
import pickle as pkl
import inspect
import re
import compute_distances
import bcubed_evaluation
import multiview_clustering
from io import StringIO
import pandas


'''
    1. Manually create folder for paraphrase
    2. Produce a script by running gather_images_features(parapaphrase, part_of_speech)
    3. Run the script with bash
'''
def gather_image_features(paraphrase, part_of_speech):
    clusters = bcubed_evaluation.get_wordnet_clusters(paraphrase, part_of_speech)
    mappings = bcubed_evaluation.get_wordnet_mappings(clusters, paraphrase)
    bcubed_evaluation.get_ssh_script(paraphrase, mappings[2])

'''
    Computes the affinity matrices, stores them
'''
def get_matrices(paraphrase):
    compute_distances.get_image_matrix(paraphrase)
    compute_distances.get_word2vec_matrix(paraphrase)


def get_all_affinity_matrices():
    all_dirs = os.listdir('features/')
    num_words = len(all_dirs)
    curr_progress = 1

    for dir in all_dirs[29:]:
        get_matrices(dir)
        print(str(curr_progress) + ' word(s) completed.')
        curr_progress += 1

def get_concreteness_map():
    all_words = os.listdir('features/')
    concreteness_ratings = pandas.read_excel('Concreteness_ratings_Brysbaert_et_al_BRM.xlsx')
    concreteness_ratings = concreteness_ratings.as_matrix()
    concreteness_vocab = [row[0] for row in concreteness_ratings]

    concreteness_map = {}

    # Create map of form {key = word, value = [concreteness_mean, concreteness_SD]}
    for i in range(0, len(concreteness_vocab)):
        for j in range(0, len(all_words)):
            if concreteness_vocab[i] == all_words[j]:
                concreteness_map[all_words[j]] = [concreteness_ratings[i][2], concreteness_ratings[i][3]]

    return concreteness_map

'''
    Returns clusters from clustering algorithms
'''
def get_clusters(paraphrase, part_of_speech):
    images_matrix = np.load('affinity_matrices/' + paraphrase + '_image_matrix.npy') #this must be precomputed
    word2vec_matrix = np.load('affinity_matrices/' + paraphrase + '_word2vec_matrix.npy') #this must be precomputed


    concreteness_map = get_concreteness_map()
    #'''

    with open('linear_regression.pkl', 'rb') as fd:
        linear_regression = pkl.load(fd)

    X = list(concreteness_map[paraphrase])
    image_variance = np.var(images_matrix)
    word2vec_variance = np.var(word2vec_matrix)
    X.append(image_variance)
    X.append(word2vec_variance)
    image_weight = linear_regression.predict(X)
    print('image_weight: ', str(image_weight))
    #print('prob: ', probabilities)
    #
    #'''

    ground_truth_clusters = bcubed_evaluation.get_wordnet_clusters(paraphrase, part_of_speech)
    mappings = bcubed_evaluation.get_wordnet_mappings(ground_truth_clusters, paraphrase)
    ground_truth_clusters = bcubed_evaluation.cull_ground_truth_clusters(ground_truth_clusters, mappings[0])
    ground_truth_clusters = bcubed_evaluation.formatted_ground_truth_clusters(ground_truth_clusters)

    [output_clusters_images, output_clusters_word2vec] = bcubed_evaluation.get_output_clusters(images_matrix, word2vec_matrix, paraphrase)
    output_clusters_multiview = multiview_clustering.get_multiview_clusters(paraphrase, part_of_speech, image_weight=image_weight)
    return [ground_truth_clusters, output_clusters_images, output_clusters_word2vec, output_clusters_multiview]


'''
    Returns bcubed evaluation for clusters
'''
def get_eval(output_clusters, ground_truth_clusters):
    results = bcubed_evaluation.eval(output_clusters, ground_truth_clusters)
    return results


def variable_name(p):
  for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
    m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
    if m:
      return m.group(1)

def view_clusters(y, y_hat_images, y_hat_word2vec, y_hat_multiview, results_images, results_word2vec, results_multiview):
    all_y = [y, y_hat_images, y_hat_word2vec, y_hat_multiview]
    all_results = [results_images, results_word2vec, results_multiview]

    for i in range(0, len(all_y)):
        print('*******************')
        print('*******************')
        print('*******************')
        if i == 0:
            print('y')
        elif i == 1:
            print('y_hat_images')
        elif i == 2:
            print('y_hat_word2vec')
        else:
            print('y_hat_multiview')
        print('*******************')

        max_value = 0

        for value in all_y[i].values():
            temp = list(value)[0]
            if temp > max_value:
                max_value = temp

        for j in range(0, max_value+1):
            for key in all_y[i].keys():
                #temp_list = all_y[i]
                if list((all_y[i])[key])[0] == j:
                    #print('hi')
                    print(key)
            print('-------------')

        #print(max_value)
        print('*******************')
        print('*******************')
        print('*******************')

    for i in range(0, len(all_results)):
        print('----------------')
        if i == 0:
            print('results_images')
        elif i == 1:
            print('results_word2vec')
        elif i == 2:
            print('results_multiview')
        else:
            print('y_hat_multiview')

        print(all_results[i])
        print('----------------')




def get_average_results(all_words):

    num_words = len(all_words)
    #print('num_words: ', num_words)
    sum_precision_images = 0
    sum_precision_word2vec = 0
    sum_precision_multiview = 0
    sum_recall_images = 0
    sum_recall_word2vec = 0
    sum_recall_multiview = 0
    sum_bcubed_images = 0
    sum_bcubed_word2vec = 0
    sum_bcubed_multiview = 0

    for word in all_words:
        paraphrase = word
        part_of_speech = 'n'
        print('paraphrase (debugging): ', paraphrase)
        [y, y_hat_images, y_hat_word2vec, y_hat_multiview] = get_clusters(paraphrase, part_of_speech)

        temp_list = list(y.keys())
        #for y_key in temp_list:
        #    if y_key not in y_hat_images.keys():
        #        y.pop(y_key, None)


        print('-------')
        print(word)
        print(y)
        print(y_hat_images)
        print('-------')
        results_images = get_eval(y_hat_images, y)

        results_word2vec = get_eval(y_hat_word2vec, y)
        results_multiview = get_eval(y_hat_multiview, y)

        sum_precision_images += results_images[0]
        sum_precision_word2vec += results_word2vec[0]
        sum_precision_multiview += results_multiview[0]
        sum_recall_images += results_images[1]
        sum_recall_word2vec += results_word2vec[1]
        sum_recall_multiview += results_multiview[1]
        sum_bcubed_images += results_images[2]
        sum_bcubed_word2vec += results_word2vec[2]
        sum_bcubed_multiview += results_multiview[2]

    map = {}

    map['images'] = [sum_precision_images/num_words, sum_recall_images/num_words, sum_bcubed_images/num_words]
    map['word2vec'] = [sum_precision_word2vec/num_words, sum_recall_word2vec/num_words, sum_bcubed_word2vec/num_words]
    map['multiview'] = [sum_precision_multiview/num_words, sum_recall_multiview/num_words, sum_bcubed_multiview/num_words]

    return map





if __name__== '__main__':
    all_words = os.listdir('features/')
    #gather_image_features('marvel', 'n')
    #print(get_all_affinity_matrices())

    #test = np.genfromtxt('Concreteness_ratings_Brysbaert_et_al_BRM.txt', delimiter=' ', dtype=None)


    #print(get_average_results(all_words[88:110]))

    '''
    x = get_clusters(all_words[105], 'n')
    ground_truth_clusters = bcubed_evaluation.get_wordnet_clusters(all_words[105], 'n')
    mappings = bcubed_evaluation.get_wordnet_mappings(ground_truth_clusters, all_words[105])
    ground_truth_clusters = bcubed_evaluation.cull_ground_truth_clusters(ground_truth_clusters, mappings[0])
    ground_truth_clusters = bcubed_evaluation.formatted_ground_truth_clusters(ground_truth_clusters)
    print(get_eval(x[3], ground_truth_clusters))
    '''
    #print(x[1])
    #print(x[2])
    #print(x[3])

    '''
    all_words = os.listdir('features/')
    print(all_words)
    print(len(all_words))
    print(get_average_results(all_words=all_words))
    '''


    '''
    paraphrase = 'break'
    part_of_speech = 'n'
    #gather_image_features(paraphrase, part_of_speech)
    #get_matrices(paraphrase)
    [y, y_hat_images, y_hat_word2vec, y_hat_multiview] = get_clusters(paraphrase, part_of_speech)

    results_images = get_eval(y_hat_images, y)
    results_wod2vec = get_eval(y_hat_word2vec, y)
    results_multiview = get_eval(y_hat_multiview, y)


    view_clusters(y=y, y_hat_images=y_hat_images, y_hat_word2vec=y_hat_word2vec, y_hat_multiview=y_hat_multiview,
                  results_images=results_images, results_word2vec= results_wod2vec, results_multiview=results_multiview)

    '''
