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
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import dendrogram
import bcubed
from nltk.corpus import wordnet as wn
import compute_distances


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


def get_ground_truth_clusters(): #returns a dictionary of clusters
    clusters = {}

    for paraphrase in bug_mappings:
        cluster = set()
        for element in get_ground_truth_clusters_helper(paraphrase, 'n'):
            cluster.add(element.name().split('.')[0])
            #if paraphrase == 'microbe':
            #    print(element)
        clusters[paraphrase] = cluster

    remove_extraneous_paraphrases(clusters)
    #print(clusters)
    modified_clusters = condense_clusters(clusters)

    #for key in modified_clusters:
    #    print(key)
    #    print(modified_clusters[key])
    #    print('----------')

    return modified_clusters

def remove_extraneous_paraphrases(clusters):
    for key in clusters: #remove paraphrases that don't appear in bug_mappings
        temp_list = []
        temp_list = list(clusters[key])
        for value in clusters[key]:
            #print(temp_list)
            if value not in bug_mappings:
                temp_list.remove(value)

        #print(temp_list)
        clusters[key] = set(temp_list)


def condense_clusters(clusters):
    modified_clusters = {}

    for key in clusters:
        modified_clusters[key] = clusters[key]
        temp_list = []
        temp_list.append(key)
        temp_list.extend(clusters[key])

    while True:
        #if break_while: #break from while
        #    break
        break_outer_for = False
        condensed_one = False
        for first_key in modified_clusters: #first list
            first_values = modified_clusters[first_key]

            if break_outer_for: #break from outer for loop
                break
            condensed_one = False
            for second_key in [x for x in modified_clusters if x != first_key]:
            #for second_key in [item for item in list_of_tuples[1] if item != first_list]:  #second list
                second_values = modified_clusters[second_key]
                #check if there is overlap AND first_values != second_values
                if (set(first_values) & set(second_values)) and (first_values != second_values):
                    combined_set = first_values | second_values
                    modified_clusters[first_key] = combined_set
                    modified_clusters[second_key] = combined_set
                    break_outer_for = True
                    condensed_one = True
                    break
        if condensed_one == False:
            break

    return modified_clusters


def get_ground_truth_clusters_helper(paraphrase, part_of_speech): #returns of a set of items for a paraphrase
    cluster = set()
    
    #add all syns and lemmas of syns to cluster
    for syn in wn.synsets(paraphrase, pos = part_of_speech):
        cluster.add(syn)
        cluster |= set(list(syn.lemmas()))

    #add all hyponyms and hypernyms to cluster
    for element in list(cluster):
        for hypo in element.hyponyms():
            cluster.add(hypo)
            hypo_lemmas = hypo.lemmas()
            cluster |= set(hypo_lemmas)
            #cluster |= set(wn.synsets(hypo.name().split('.')[0], pos=part_of_speech))
            for item in hypo_lemmas:
                #cluster |= set(wn.synsets(item.name().split('.')[0], pos=part_of_speech))
                cluster |= set(item.hyponyms())
                cluster |= set(item.hypernyms())

        for hyper in element.hypernyms():
            cluster.add(hyper)
            hyper_lemmas = hyper.lemmas()
            cluster |= set(hyper_lemmas)
            #cluster |= set(wn.synsets(hyper.name().split('.')[0], pos=part_of_speech))
            for item in hyper_lemmas:
                #cluster |= set(wn.synsets(item.name().split('.')[0], pos=part_of_speech))
                cluster |= set(item.hyponyms())
                cluster |= set(item.hypernyms())

    return cluster


def get_ground_truth_clusters_old(paraphrase, part_of_speech):
    # retrieve synsets for bug (noun)
    wn.synsets('bug', 'n')

    # retrieve lemmas for all synsets of bug, store in a dict
    syn_lems = {}
    for syn in wn.synsets(paraphrase, pos = part_of_speech):
        syn_lems[syn.name().split('.')[0]] = set([l.name() for l in syn.lemmas()]) #set modification

    # retrieve hypernyms for each synset and append to each cluster
    for syn in wn.synsets(paraphrase, pos = part_of_speech):
        temp = set(syn_lems[syn.name().split('.')[0]]) #set modification
        for hypo in syn.hyponyms():
            temp |= set([l.name() for l in hypo.lemmas()])
        for hyper in syn.hypernyms():
            temp |= set([l.name() for l in hyper.lemmas()])
        syn_lems[syn.name().split('.')[0]] = temp



    # The result of the above would be a dict encoding the
    # gold sense clusters for your target word ( in this case bug).
    #

    return syn_lems

def get_output_clusters():
    dict_images= {}
    dict_word2vec = {}
    y_images = compute_distances.spectral_clustering(compute_distances.images_matrix)
    y_word2vec = compute_distances.spectral_clustering(compute_distances.word2vec_matrix)

    for i in range(0, len(y_images)):
        if i in dict_images or i in dict_word2vec:
            dict_images[i] = set.add(y_images[i])
            dict_word2vec[i] = set.add(y_word2vec[i])
        else:
            dict_images[i] = set([y_images[i]])
            dict_word2vec[i] = set([y_word2vec[i]])

    temp_dict = {v: k for k, v in bug_mappings.items()}
    values = bug_mappings.values()  # numbers

    for value in values: #change number keys to word keys
        dict_images[temp_dict[value]] = dict_images.pop(value)
        dict_word2vec[temp_dict[value]] = dict_word2vec.pop(value)


    return [dict_images, dict_word2vec]


def eval(ground_truth_clusters, output_clusters):
    precision = bcubed.precision(output_clusters, ground_truth_clusters)
    recall = bcubed.recall(output_clusters, ground_truth_clusters)
    fscore = bcubed.fscore(precision, recall)

    return [precision, recall, fscore]


if __name__== '__main__':
    '''
    ground_truth_clusters = get_ground_truth_clusters()

    [output_clusters_images, output_clusters_word2vec]= get_output_clusters()
    results_images = eval(output_clusters_images, ground_truth_clusters)
    results_word2vec = eval(output_clusters_word2vec, ground_truth_clusters)

    print(ground_truth_clusters)
    print('------------------------------------')
    print(output_clusters_images)
    print(output_clusters_word2vec)
    print('------------------------------------')
    print(results_images)
    print(results_word2vec)
    '''
    print(get_ground_truth_clusters_old('bug', 'n'))