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
    for syn in wn.synsets(paraphrase, pos =
    part_of_speech):
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


def get_wordnet_clusters(paraphrase, part_of_speech):
    # retrieve synsets for bug (noun)
    wn.synsets('affair', 'n')

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

    for key in syn_lems:
        temp_list = list(syn_lems[key])
        for value in temp_list:
            if '_' in value:
                syn_lems[key].remove(value)

    return syn_lems

def get_wordnet_mappings(syn_lems, paraphrase):
    superset_mappings = {}
    line_mappings = {}

    mappings = {}

    for key in syn_lems:
        superset_mappings[key] = None
        line_mappings[key] = None
        for value in syn_lems[key]:
            superset_mappings[value] = None
            line_mappings[value] = None

    word_mapping_keys = line_mappings.keys()

    dir = os.listdir('english_supersets')
    #print(dir)
    #os.listdir('english_supersets')
    for file_name in dir:
        #print('english_supersets/' + filpickle.dump(model, open(filename, 'wb'))e_name)
        with open('english_supersets/' + file_name) as file:
            for num, line in enumerate(file, 1):
                #print(line)
                for key in word_mapping_keys:
                    if key == line.strip():
                        superset_mappings[key] = file_name
                        line_mappings[key] = num-1
                        mappings[key] = superset_mappings[key] + '/' + str(line_mappings[key])

    '''
    with open('script_' + paraphrase + '.txt', 'w') as file:
        prefix = 'scp -r edidiong@nlpgrid.seas.upenn.edu:/nlp/data/bcal/features/alexnet/English-'
        superset_index = len('english.superset')

        for key in mappings.keys():
            suffix = ' features/' + paraphrase + '/' + key
            file.write(prefix + mappings[key][superset_index:] + suffix + '\n')

        file.close()
    '''

    return [superset_mappings, line_mappings, mappings]

def get_ssh_script(paraphrase, mappings):
    with open('new_scripts/script_' + paraphrase + '.txt', 'w') as file:
        prefix = 'scp -r edidiong@nlpgrid.seas.upenn.edu:/nlp/data/bcal/features/alexnet/English-'
        superset_index = len('english.superset')

        for key in mappings.keys():
            suffix = ' features/' + paraphrase + '/' + key
            file.write(prefix + mappings[key][superset_index:] + suffix + '\n')

        file.close()

def get_output_clusters(images_matrix, word2vec_matrix, paraphrase):
    dict_images= {}
    dict_word2vec = {}
    y_images = compute_distances.spectral_clustering(images_matrix)
    y_word2vec = compute_distances.spectral_clustering(word2vec_matrix)

    all_dirs = os.listdir('features/' + paraphrase)
    mappings = {}

    for i in range(0, len(all_dirs)):
        mappings[i] = all_dirs[i]

    for i in range(0, len(y_images)):
        if i in dict_images or i in dict_word2vec:
            dict_images[mappings[i]] = set.add(y_images[i])
            dict_word2vec[mappings[i]] = set.add(y_word2vec[i])
        else:
            dict_images[mappings[i]] = set([y_images[i]])
            dict_word2vec[mappings[i]] = set([y_word2vec[i]])

    return [dict_images, dict_word2vec]


def cull_ground_truth_clusters(ground_truth_clusters, mapping):
    culled_clusters = ground_truth_clusters.copy()
    for mapping_key in mapping.keys():
        if mapping[mapping_key] == None:
            for cluster_key in ground_truth_clusters.keys():
                temp_set = culled_clusters[cluster_key].copy()
                temp_set.discard(mapping_key)
                culled_clusters[cluster_key] = temp_set
    return culled_clusters

def pre_eval(output_clusters, ground_truth_clusters):
    ground_truth_keys = list(ground_truth_clusters.keys())
    output_keys = list(output_clusters.keys())

    for key in ground_truth_keys:
        if key not in output_keys:
            ground_truth_clusters.pop(key, None)

    ground_truth_keys = list(ground_truth_clusters.keys())

    for key in output_keys:
        if key not in ground_truth_keys:
            output_clusters.pop(key, None)

def formatted_ground_truth_clusters(ground_truth_clusters):
    formatted_ground_truth_clusters = {}
    i = 0
    for key in ground_truth_clusters.keys():
        for value in ground_truth_clusters[key]:
            formatted_ground_truth_clusters[value] = {i}
        i = i + 1
    return formatted_ground_truth_clusters


def eval(output_clusters, ground_truth_clusters):
    precision = bcubed.precision(output_clusters, ground_truth_clusters)
    recall = bcubed.recall(output_clusters, ground_truth_clusters)
    fscore = bcubed.fscore(precision, recall)
    return [precision, recall, fscore]

if __name__== '__main__':

    paraphrase = 'function'

    images_matrix = np.load('function_image_matrix.npy')
    word2vec_matrix = np.load('function_word2vec_matrix.npy')

    ground_truth_clusters = get_wordnet_clusters(paraphrase, 'n')
    mappings = get_wordnet_mappings(ground_truth_clusters, paraphrase)
    ground_truth_clusters = cull_ground_truth_clusters(ground_truth_clusters, mappings[0])
    ground_truth_clusters = formatted_ground_truth_clusters(ground_truth_clusters)

    [output_clusters_images, output_clusters_word2vec] = get_output_clusters(images_matrix, word2vec_matrix, paraphrase)

    results_images = eval(output_clusters_images, ground_truth_clusters)
    results_word2vec = eval(output_clusters_word2vec, ground_truth_clusters)

    print(ground_truth_clusters)
    print('------------------------------------')
    print(output_clusters_images)
    print(output_clusters_word2vec)
    print('------------------------------------')
    print(results_images)
    print(results_word2vec)

    print(len(ground_truth_clusters.items()))
    print(len(output_clusters_images.items()))



    '''
    clusters = get_wordnet_clusters(paraphrase, 'n')
    mappings = get_wordnet_mappings(clusters, paraphrase)
    mappings_index2 = mappings[2]


    print(mappings[0])
    print(mappings[1])
    print(mappings[2])
    print(clusters)
    print(culled_clusters)
    '''