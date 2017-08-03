import os
import sys
import gzip
import json
import numpy as np
import scipy.io as sio
from oct2py import octave
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score
from copy import deepcopy
from nltk.stem.wordnet import WordNetLemmatizer
import pickle as pkl
#from word_pos import *
from clus_nmi import nmi
from clus_bcubed import score_bcubed
#from wn import WN
#from utils import read_sents
from nltk.corpus import wordnet
#import addcos
#from settings import SETTINGS
from sklearn.linear_model import LogisticRegression
import pandas
import compute_distances
import bcubed_evaluation
from sklearn import linear_model, datasets
import pipeline

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

def count_nonzero(l, ignore=None):
    if not ignore:
        return sum([ll for ll in l if ll > 0])
    else:
        return sum([ll for i, ll in enumerate(l) if ll > 0 and i != ignore])


def flatten(l):
    return [item for sublist in l for item in sublist]


class TargetParaphrases:
    def __init__(self, word, pos, paraphrase): #Added the 4th parameter paraphrase
        self.baseword = word
        self.pos = pos
        self.paraphrase = paraphrase #Added this
        self.paraphrases = os.listdir('features/' + paraphrase) ##CHANGE THIS FOR DIFFERENT PARAPHRASES
        self.paraphrases_vecs = None
        self.paraphrases_vecs_norm = None
        self.pp_pp_mat = None  # PPDB 2.0 Score
        self.pp_pp_distrib_mat = None  # embedding cosine sim

        self.minscore = 0.0

        self.sense_clustering = None
        self.sense_clustering_withsyns = None
        self.pp_clus_assignments = None
        self.syn_clus_assignments = None

        self.synsets = None
        self.synsets_vecs = None
        self.synsets_vecs_norm = None
        self.synlemmas = None
        self.pp_syn_mat = None  # Distributional similarity

        self.sentences = None
        self.pp_sent_mat = None  # LexSub score

        self.translations = None
        self.pp_trans_mat = None


        self.gold_clus = {}

        self.inc_pp_pp = None
        self.inc_pp_syn = None
        self.inc_pp_sent = None
        self.inc_pp_trans = None
        self.inc_pp_distrib = None
        self.pp_clus_probs = None
        self.syn_clus_probs = None

        self.V_centroid = None
        self.U_finals = None
        self.V_finals = None
        self.type2ind = None

        self.pp_images_mat = None
        self.inc_pp_images = None
        self.pp_word2vec_mat = None
        self.inc_pp_word2vec = None

    def load_gold_clus(self, lexsubdata):
        '''
        Load gold clusters from lexsub data, where gold clusters is the
        superset of annotated substitutions
        :param lexsubdata:
        :return:
        '''
        self.gold_clus = {}

        def safesplit(s):
            splt = s.split('_')
            return '_'.join(splt[:-1]), splt[-1]

        for i in range(len(lexsubdata.idents)):
            tgtw, tgtp = safesplit(lexsubdata.targets[i])
            if tgtw == self.baseword:
                if tgtp[0].lower() == self.pos[0].lower():
                    golddict = lexsubdata.golds[i]
                    clusnum = len(self.gold_clus)
                    self.gold_clus[clusnum] = [safesplit(k)[0]
                                               for k, sc in golddict.items()
                                               if sc > 0]

        if len(self.gold_clus) > 0:
            return 1
        else:
            return 0

    '''
    def load_paraphrases(self, ppdbobj, embedmodel, minscore=1.0, cleanPOS=True):
    # remember minscore
    self.minscore = minscore

    # load paraphrases
    ppdict = ppdbobj.ppsets.get(word_pos(self.baseword, self.pos), {})

    # filter by minscore
    ppdict = {w: s for w, s in ppdict.iteritems() if s >= minscore}
    if len(ppdict) == 0:
        sys.stderr.write('No PPDB paraphrases found for target %s (%s)\n'
                         % (self.baseword, self.pos))
        return 0

    if cleanPOS:
        ## Remove paraphrases that are likely the wrong POS
        posmap = {'N': 'n', 'V': 'v', 'J': 'a', 'R': 'r'}
        errorpps = set([])
        for p in ppdict:
            if len(wordnet.synsets(p, posmap[self.pos[0]])) > 0:
                continue
            for wrongpos in set(['n', 'v', 'a', 'r']) - set([posmap[self.pos[0]]]):
                if len(wordnet.synsets(p, wrongpos)) > 0:
                    errorpps.add(p)
                    break
        for e in errorpps:
            del ppdict[e]

    self.paraphrases = sorted(ppdict.keys())
    m = len(self.paraphrases)

    # load paraphrase vecs
    embedding_dim = embedmodel.syn0.shape[1]
    self.paraphrases_vecs = np.random.uniform(size=(m, embedding_dim))
    for i, p in enumerate(self.paraphrases):
        if p in embedmodel.vocab:
            self.paraphrases_vecs[i] = embedmodel[p]
    self.paraphrases_vecs_norm = normalize(self.paraphrases_vecs, axis=1)

    # create pp_pp mat (using PPDB 2.0 Score as sim)
    self.pp_pp_mat = np.zeros((m, m))
    for i, p1 in enumerate(self.paraphrases):
        self.pp_pp_mat[i][i] = 6.
        p1_dict = ppdbobj.ppsets.get(word_pos(p1, self.pos))
        for j, p2 in enumerate(self.paraphrases):
            p1_p2_ppdbscore = p1_dict.get(p2, 0.0)
            if p1_p2_ppdbscore >= minscore:
                self.pp_pp_mat[i][j] = p1_p2_ppdbscore
                self.pp_pp_mat[j][i] = p1_p2_ppdbscore  # maintain symmetry

    # create distributional pp-pp mat (using embeddinig cosine sim as similarity)
    self.pp_pp_distrib_mat = np.dot(self.paraphrases_vecs_norm, self.paraphrases_vecs_norm.T)

    return 1
    '''

    '''
    def load_translations(self, transobj):

        transdict = transobj.ppsets.get(word_pos(self.baseword, self.pos), {})

        if len(transdict) == 0:
            sys.stderr.write('No translations found for target %s (%s)\n'
                             % (self.baseword, self.pos))
            return 0
        m = len(self.paraphrases)

        fword_list = list(set(flatten([v.keys() for k, v in transdict.items()
                                       if k.word in self.paraphrases])))
        self.translations = fword_list
        f2i = {f: i for i, f in enumerate(fword_list)}
        n = len(fword_list)

        self.pp_trans_mat = np.zeros((m, n))

        for i, pp in enumerate(self.paraphrases):
            pp_wt = word_pos(pp, self.pos)
            for fw, sc in transdict.get(pp_wt, {}).items():
                self.pp_trans_mat[i][
                    f2i[fw]] = -sc  # scores are log prob, so smaller is better; will be transformed later
        return 1
    '''
    '''
    def load_synsets(self, word_embeddings, ppdbobj, depth=1):

        wn = WN(self.baseword, self.pos)
        self.synsets, self.synlemmas = wn.load_synsets_tgt(depth)

        if len(self.synsets) == 0:
            return 0
        self.synsets_vecs = wn.load_vecs(word_embeddings, ppdbobj)
        if self.synsets_vecs is None:
            return 0
        self.synsets_vecs_norm = normalize(self.synsets_vecs, axis=1)

        if self.paraphrases is None:
            sys.stderr.write('Paraphrases have not been loaded; cannot create'
                             'paraphrase-synset matrix\n')
        else:
            self.pp_syn_mat = np.dot(self.paraphrases_vecs_norm, self.synsets_vecs_norm.T)

        wnl = WordNetLemmatizer()
        for i, pp in enumerate(self.paraphrases):
            pp_lem = wnl.lemmatize(pp)
            if pp_lem == self.baseword:
                continue
            for j, synset in enumerate(self.synsets):
                if pp_lem in self.synlemmas[synset]:
                    self.pp_syn_mat[i][j] = 1.0

        return 1
    '''


    def rescale(self, mat):
        '''
        Rescale matrix to range 0,1
        :param mat:
        :return:
        '''
        matmin = np.min(mat)
        matmax = np.max(mat)
        if matmax == matmin:
            sys.stderr.write('Error when rescaling matrix: max==min\n')
            return mat
        scalemat = (mat - matmin) / (matmax - matmin)
        return scalemat


    def cluster(self, imgfile, w2vfile, inc_pp_pp=False, inc_pp_syn=False, inc_pp_sent=False,
                inc_pp_trans=False, inc_pp_distrib=False, inc_pp_images = True,
                inc_pp_word2vec = True, fileprefix='.', choosek='silhouette', image_weight = 0.5):
        '''
        Use Multi-view Nonnegative Matrix Factorization to cluster along
        multiple views:
        - PP-PP (PPDB2.0Score)
        - PP-PP (embedding cosine similarity)
        - PP-Synset
        - PP-Sentences
        - PP-Translations
        Reference:
        @inproceedings{sdm2013_liu,
          author = {Liu, Jialu and Wang, Chi and Gao, Jing and Han, Jiawei},
          title = {Multi-View Clustering via Joint Nonnegative Matrix Factorization},
          booktitle = {Proc. of 2013 SIAM Data Mining Conf.},
          year = {2013},
        }
        :param inc_pp_pp: boolean
        :param inc_pp_syn: boolean
        :param inc_pp_sent: boolean
        :param inc_pp_trans: boolean
        :param inc_pp_distrib: boolean
        :param fileprefix: str
        :param eng: matlab.engine instance (if not supplied, will start and
               stop one within the fn execution)
        :param choosek: str
        :return:
        '''
        assert inc_pp_pp or inc_pp_syn or inc_pp_sent or inc_pp_trans or inc_pp_distrib or inc_pp_images or inc_pp_word2vec
        assert choosek in ['nmi', 'bcubed', 'silhouette']

        self.inc_pp_pp = inc_pp_pp
        self.inc_pp_syn = inc_pp_syn
        self.inc_pp_sent = inc_pp_sent
        self.inc_pp_trans = inc_pp_trans
        self.inc_pp_distrib = inc_pp_distrib

        self.inc_pp_images = inc_pp_images
        self.inc_pp_word2vec = inc_pp_word2vec

        inputfile = os.path.join(fileprefix, 'input.mat')
        outputfile = os.path.join(fileprefix, 'output.mat')

        # Write output matrices to file

        arrlen = self.inc_pp_pp + self.inc_pp_syn + self.inc_pp_sent + self.inc_pp_trans + self.inc_pp_distrib + self.inc_pp_images + self.inc_pp_word2vec

        matlab_arr = np.zeros((arrlen,), dtype=np.object)
        i = 0
        self.type2ind = {}
        if self.inc_pp_sent:
            matlab_arr[i] = self.rescale(self.pp_sent_mat.T)
            i += 1
            self.type2ind['pp_sent'] = len(self.type2ind)
        if self.inc_pp_trans:
            matlab_arr[i] = self.rescale(self.pp_trans_mat.T)
            i += 1
            self.type2ind['pp_trans'] = len(self.type2ind)
        if self.inc_pp_pp:
            matlab_arr[i] = self.rescale(self.pp_pp_mat.T)
            i += 1
            self.type2ind['pp_pp'] = len(self.type2ind)
        if self.inc_pp_syn:
            matlab_arr[i] = self.rescale(self.pp_syn_mat.T)
            i += 1
            self.type2ind['pp_syn'] = len(self.type2ind)
        if self.inc_pp_distrib:
            matlab_arr[i] = self.rescale(self.pp_pp_distrib_mat.T)
            i += 1
            self.type2ind['pp_pp_distrib'] = len(self.type2ind)
        if self.inc_pp_images:
            matlab_arr[i] = self.rescale(np.load(imgfile).T)
            self.type2ind['pp_images'] = len(self.type2ind)
            i += 1
        if self.inc_pp_word2vec:
            matlab_arr[i] = self.rescale(np.load(w2vfile).T)
            self.type2ind['pp_word2vec'] = len(self.type2ind)
            i += 1


        sio.savemat(inputfile, {'data': matlab_arr})

        # Cluster at a range of K's, and choose clustering that gives best score
        maxscore = -1.
        best_k = -1.
        best_y = None
        bestresult = None
        pred_clus = None
        result_contents = None

        '''

        for k in range(2, min(len(self.paraphrases), 10)):
            try:
                # Run matlab code on output matrices
                __ = octave.run_MultiNMF(k, inputfile, outputfile)  # for some reason returning something different
                # Read matlab result
                result_contents = sio.loadmat(outputfile)
                v_centroid = result_contents['V_centroid']
                y_pps = np.argmax(v_centroid, axis=1)

                pred_clus = {clusnum: list(np.array(self.paraphrases)[y_pps == clusnum])
                             for clusnum in range(k)}
                pred_clus = {i: v for i, v in enumerate(pred_clus.values()) if len(v) > 0}

                if choosek == 'nmi':
                    nmiscore = nmi(pred_clus, self.gold_clus)
                    print
                    k, nmiscore, json.dumps(pred_clus, indent=2)
                    if not nmiscore:
                        continue
                    if nmiscore[0] >= maxscore:
                        maxscore = nmiscore[0]
                        best_k = k
                        bestresult = result_contents
                        best_y = y_pps
                elif choosek == 'bcubed':
                    res = score_bcubed(self.gold_clus, pred_clus, beta=0.5)
                    if not res:
                        continue
                    fscore, precision, recall, __ = res
                    print
                    k, precision, json.dumps(pred_clus, indent=2)

                    if fscore >= maxscore:
                        maxscore = fscore
                        best_k = k
                        bestresult = result_contents
                        best_y = y_pps
                elif choosek == 'silhouette':
                    nlabels = len(set(y_pps))
                    if nlabels not in range(2, len(self.paraphrases)):
                        continue
                    silscore = silhouette_score(self.pp_pp_mat, y_pps, metric='cosine')
                    if not silscore:
                        continue
                    print
                    k, silscore, json.dumps(pred_clus, indent=2)
                    if silscore >= maxscore:
                        maxscore = silscore
                        best_k = k
                        bestresult = result_contents
                        best_y = y_pps
            except:
                sys.stderr.write('Error clustering %s at k=%d\n' % (self.baseword, k))
                continue
        '''



        if bestresult is None: #**** In this block, "self.paraphrase" replaced with imgfile[0]****
            sys.stderr.write(
                'Could not get best K for target %s based on NMI (maybe no overlapping words?). Defaulting to number of WordNet synsets.\n' % self.baseword)
            poslookup = {'N': 'n', 'V': 'v', 'R': 'r', 'J': 'a'}
            #k = min(max(2, len(imgfile[0]) - 1), len(wordnet.synsets(self.baseword, poslookup[self.pos[0]])))
            k_image = compute_distances.find_k_pca(np.load('affinity_matrices/' + self.paraphrase + '_image_matrix.npy'))
            k_word2vec = compute_distances.find_k_pca(np.load('affinity_matrices/' + self.paraphrase + '_word2vec_matrix.npy'))
            #k = int((k_image*image_weight + k_word2vec*(1-image_weight))/2)
            k = int((k_image + k_word2vec)/2)
            if k == 0: #error trap
                k = 1
            __ = octave.run_MultiNMF(k, inputfile, outputfile, image_weight)
            result_contents = sio.loadmat(outputfile)
            v_centroid = result_contents['V_centroid']
            y_pps = np.argmax(v_centroid, axis=1)
            best_k = k
            best_y = y_pps
            bestresult = result_contents

        # consolidate results
        self.V_centroid = bestresult['V_centroid']
        self.U_finals = bestresult['U_final'][0]
        self.V_finals = bestresult['V_final'][0]

        self.pp_clus_probs = bestresult['V_centroid']
        self.pp_clus_assignments = dict(zip(self.paraphrases, best_y)) #replaced "self.paraphrases" with "imgfile[0]"
        self.all_assignments = self.pp_clus_assignments
        print("error check best_k: ", best_k)
        self.sense_clustering = {clusnum: list(np.array(self.paraphrases)[best_y == clusnum])
                                 for clusnum in range(best_k)}
        self.sense_clustering = {i: v for i, v in enumerate(self.sense_clustering.values()) if len(v) > 0}
        self.all_clustering = deepcopy(self.sense_clustering)







        if self.inc_pp_syn:
            self.syn_clus_probs = bestresult['U_final'][0][self.type2ind['pp_syn']]

            y_syn = np.argmax(self.syn_clus_probs, axis=1)
            self.syn_clus_assignments = dict(zip(self.synsets, y_syn))
            self.all_assignments.update(self.syn_clus_assignments)

            syn_clus = {clusnum: list(np.array(self.synsets)[y_syn == clusnum])
                        for clusnum in range(best_k)}

            for clusnum in range(best_k):
                self.all_clustering[clusnum] = self.all_clustering.get(clusnum, []) + syn_clus.get(clusnum, [])

            self.all_clustering = {clusnum: cluslist for clusnum, cluslist in self.all_clustering.items() if
                                   len(cluslist) > 0}

    def write(self, dirname, ignoresyns=False):
        filename = os.path.join(dirname, 'ppdb-2.0-sub-clusters-%s.gz' % self.pos)
        outpos = '[%s]' % self.pos
        with gzip.open(filename, 'a') as fout:
            if ignoresyns:
                print('before')
                print(self.sense_clustering)
                print('after')
                output_clusters = {}
                for key in self.sense_clustering.keys():
                    for value in self.sense_clustering[key]:
                        output_clusters[value] = set([key])


                ground_truth_clusters = bcubed_evaluation.get_wordnet_clusters('function', 'n')
                mappings = bcubed_evaluation.get_wordnet_mappings(ground_truth_clusters, self.paraphrase)
                ground_truth_clusters = bcubed_evaluation.cull_ground_truth_clusters(ground_truth_clusters, mappings[0])
                ground_truth_clusters = bcubed_evaluation.formatted_ground_truth_clusters(ground_truth_clusters)
                results = bcubed_evaluation.eval(ground_truth_clusters, output_clusters)
                #return [output_clusters, results] #ADDED THIS

                fout.write("%s ||| %s ||| %s\n"
                           % (outpos, self.baseword,
                              json.dumps(self.sense_clustering)))
            else:
                sense_clustering_withsyns = {
                clusnum: ['PP:' + i if type(i) in [str, np.string_] else 'SYN:' + i.name().encode('utf8') for i in clus]
                for clusnum, clus in self.all_clustering.items()}
                fout.write("%s ||| %s ||| %s\n"
                           % (outpos, self.baseword,
                              json.dumps(sense_clustering_withsyns)))

    def write_data(self, dirname):
        '''
        Write important results in pkl format
        '''
        self.synsetnames = [s.name() for s in self.synsets]
        outobj = [self.baseword, self.pos, self.paraphrases, self.pp_pp_mat,
                  self.minscore, self.sense_clustering, self.sense_clustering_withsyns,
                  self.synsetnames, self.pp_syn_mat, self.sentences, self.pp_sent_mat,
                  self.translations, self.pp_trans_mat, self.pp_pp_distrib_mat, self.gold_clus,
                  self.type2ind, self.V_centroid, self.U_finals, self.V_finals]

        with open(os.path.join(dirname, '.'.join((self.baseword, self.pos))), 'w') as fout:
            pkl.dump(outobj, fout)

    def read_data(self, fname):
        with open(fname, 'r') as fin:
            inobj = pkl.load(fin)
        self.baseword, self.pos, self.paraphrases, self.pp_pp_mat, \
        self.minscore, self.sense_clustering, self.sense_clustering_withsyns, \
        self.synsetnames, self.pp_syn_mat, self.sentences, self.pp_sent_mat, \
        self.translations, self.pp_trans_mat, self.pp_pp_distrib_mat, self.gold_clus, \
        self.type2ind, self.V_centroid, self.U_finals, self.V_finals = inobj



def write_clusters(ppsets, outdir, ignoresyns=True):
    posset = set([p.pos for p in ppsets])
    posdict = {pos: [p for p in ppsets if p.pos == pos] for pos in posset}

    for pos, pplist in posdict.items():
        outpos = '[%s]' % pos
        with gzip.open(os.path.join(outdir, 'ppdb-2.0-sub-clusters-%s.gz' % pos), 'w') as fout:
            for ppobj in pplist:
                if ignoresyns:
                    fout.write("%s ||| %s ||| %s\n"
                               % (outpos, ppobj.baseword,
                                  json.dumps(ppobj.sense_clustering)))
                else:
                    sense_clustering_withsyns = {
                    clusnum: ['PP:' + i if type(i) == str else 'SYN:' + i.name() for i in clus]
                    for clusnum, clus in ppobj.all_clustering.items()}
                    fout.write("%s ||| %s ||| %s\n"
                               % (outpos, ppobj.baseword,
                                  json.dumps(sense_clustering_withsyns)))


def write_json(ppsets, outdir, methodprefix=''):
    '''
    Write clusters to json files in format for d3 viewer
    :param ppsets:
    :param outdir:
    :return:
    '''
    for ppobj in ppsets:
        # set filename
        word_PP = '_'.join((ppobj.baseword, ppobj.pos))
        ppthr = '%0.2f' % ppobj.minscore
        if ppobj.use_syns:
            method = 'mss%s-mpp%s-mps%s' % (str(ppobj.mask_syn_syn), str(ppobj.mask_pp_pp), str(ppobj.mask_pp_syn))
        else:
            method = 'mpp%s-ppsOnly' % str(ppobj.mask_pp_pp)
        method = methodprefix + method
        filename = '.'.join((word_PP, ppthr, method))

        # construct node / link format
        if ppobj.use_syns:
            pps_labeled = ['PP:' + p for p in ppobj.paraphrases]
            syn_labeled = ['SYN:' + s.name() for s in ppobj.synsets]
            all_labeled = pps_labeled + syn_labeled
            clus_assignments = [int(i) for i in ppobj.pp_clus_assignments] \
                               + [int(i) for i in ppobj.syn_clus_assignments]
        else:
            pps_labeled = ['PP:' + p for p in ppobj.paraphrases]
            all_labeled = pps_labeled
            clus_assignments = [int(i) for i in ppobj.pp_clus_assignments]

        nodes = [{'name': n, 'group': c} for n, c in zip(all_labeled, clus_assignments)]
        links = []
        n = len(all_labeled)
        for i in range(n):
            for j in range(n):
                if i > j:
                    continue
                if ppobj.affinity_matrix[i][j] != 0:
                    newlink = {'source': i,
                               'target': j,
                               'value': ppobj.affinity_matrix[i][j]}
                links.append(newlink)
        outobj = {'nodes': nodes, 'links': links, 'nmi': 0.0}
        # write to filea
        with open(os.path.join(outdir, filename), 'w') as fout:
            print >> fout, json.dumps(outobj)


def get_multiview_clusters(paraphrase, part_of_speech, image_weight=0.5):
    imgfile = 'affinity_matrices/' + paraphrase + '_image_matrix.npy'
    w_file = 'affinity_matrices/' + paraphrase + '_word2vec_matrix.npy'
    tp = TargetParaphrases(paraphrase, 'NN', paraphrase)
    tp.cluster(imgfile=imgfile, w2vfile=w_file, image_weight=image_weight)

    output_clusters = {}
    for key in tp.sense_clustering.keys():
        for value in tp.sense_clustering[key]:
            output_clusters[value] = set([key])

    return output_clusters

    #tp.write('output', ignoresyns=True)

def train_weights(training_set, filename = 'model'):
    X = []
    best_image_weights = [0] * len(training_set)
    part_of_speech = 'n'

    possible_image_weights = np.arange(0, 1.05, 0.05)

    num_iters = 0
    total_iters = len(possible_image_weights)* len(training_set)

    concreteness_map = pipeline.get_concreteness_map()

    for i in range(0, len(training_set)):
        #print('Debugging: ', training_set[i])
        ground_truth_clusters = bcubed_evaluation.get_wordnet_clusters(training_set[i], part_of_speech)
        mappings = bcubed_evaluation.get_wordnet_mappings(ground_truth_clusters, training_set[i])
        ground_truth_clusters = bcubed_evaluation.cull_ground_truth_clusters(ground_truth_clusters, mappings[0])
        ground_truth_clusters = bcubed_evaluation.formatted_ground_truth_clusters(ground_truth_clusters)
        imgfile = 'affinity_matrices/' + training_set[i] + '_image_matrix.npy'
        w_file = 'affinity_matrices/' + training_set[i] + '_word2vec_matrix.npy'
        tp = TargetParaphrases(training_set[i], 'NN', training_set[i])
        tp.cluster(imgfile=imgfile, w2vfile=w_file)
        variance_images = np.var(np.load('affinity_matrices/' + training_set[i] + '_image_matrix.npy'))
        variance_word2vec = np.var(np.load('affinity_matrices/' + training_set[i] + '_word2vec_matrix.npy'))
        temp_X = list(concreteness_map[training_set[i]])
        temp_X.append(variance_images)
        temp_X.append(variance_word2vec)
        #print(type(concreteness_map[training_set[i]]))
        #print(type(variance_images))
        #print(type(temp_X))

        X.append(temp_X)
        max_bcubed = 0
        for j in range(0, len(possible_image_weights)):
            output_clusters = get_multiview_clusters(training_set[i], 'n', possible_image_weights[j])
            temp_list = list(ground_truth_clusters.keys())
            #for y_key in temp_list:
            #    if y_key not in output_clusters.keys():
            #        ground_truth_clusters.pop(y_key, None)
            #print('ground_truth_clusters: ', ground_truth_clusters)
            #print('output_clusters: ', output_clusters)
            #print('--------------')
            bcubed_evaluation.pre_eval(output_clusters, ground_truth_clusters)
            results = bcubed_evaluation.eval(output_clusters, ground_truth_clusters)
            if results[2] > max_bcubed:
                max_bcubed = results[2]
                best_image_weights[i] = possible_image_weights[j]
            print('Progress: ' + str(num_iters+1) + '/' + str(total_iters))
            num_iters = num_iters + 1

    #print(image_variances)
    #print(best_image_weights)

    linear_regression = linear_model.LinearRegression()
    linear_regression.fit(np.asarray(X), np.asarray(best_image_weights))
    #linear_regression.fit(np.asarray(X).reshape(len(X), 1), np.asarray(best_image_weights, dtype="|S6"))

    #save model
    with open(filename + '.pkl', 'wb') as fd:
        pkl.dump(linear_regression, fd)

    #print(possible_image_weights)
    part_of_speech = 'n'


'''
def cluster(self, inc_pp_pp=False, inc_pp_syn=False, inc_pp_sent=False,
            inc_pp_trans=False, inc_pp_distrib=False, inc_pp_images=True,
            inc_pp_word2vec=True, fileprefix='.', choosek='silhouette'):
'''

def train_cross_validation_models(num_models, sample_size):
    all_words = os.listdir('features/')
    hold_out_size = int(sample_size/num_models)

    for i in range(0, num_models):
        if i == 0:
            print('Training model ' + str(i + 1) + ' of ' + str(num_models) + '.')
            training_set = list(all_words)
            temp = hold_out_size*i
            hold_out_set = list(all_words[temp: temp + hold_out_size])
            training_set = [x for x in training_set if x not in hold_out_set]
            name = 'linear_model' + str(i+1)
            train_weights(training_set, filename=name)
            print('Done training model ' + str(i + 1) + 'cof ' + str(num_models) + '.')

def test_cross_validation_models(num_models, sample_size):
    all_words = os.listdir('features/')
    hold_out_size = int(sample_size / num_models)

    for i in range(0, num_models):
        training_set = list(all_words)
        temp = hold_out_size * i
        hold_out_set = list(all_words[temp: temp + hold_out_size])
        training_set = [x for x in training_set if x not in hold_out_set]





if __name__== '__main__':
    all_words = os.listdir('features/')
    #training_set = all_words[:88]

    #concreteness_map = pipeline.get_concreteness_map()
    train_cross_validation_models(5, len(all_words))

    #print(concreteness_map)
    #train_weights(training_set)


    '''

    # save model
    with open('logistic_regression.pkl', 'wb') as fd:
        pkl.dump(logistic_regression, fd)
    '''

    #print(get_multiview_clusters('break', 'n'))
    #imgfile = 'function_image_matrix.npy'
    #w_file = 'function_word2vec_matrix.npy'
    #tp = TargetParaphrases('bug','NN')
    #tp.cluster(imgfile=imgfile, w2vfile=w_file)
    #tp.write('output', ignoresyns = True)
