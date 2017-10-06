#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Satoshi Tsutsui <stsutsui@indiana.edu>
class to get the data in a batch way
includes the sampler for word2vec negative sampling
'''


import numpy as np
import random
# import scipy
import os

class Dataset(object):
    def __init__(self,random_walk_txt,node_type_mapping_txt,window_size):
        self.nodeid2type = self.parse_node_type_mapping_txt(node_type_mapping_txt)
        index2token,token2index,word_and_counts,index2frequency,node_context_pairs= self.parse_random_walk_txt(random_walk_txt,window_size)
        self.window_size = window_size
        self.nodeid2index = token2index
        self.index2nodeid = index2token
        self.nodeid2frequency = word_and_counts
        self.index2frequency = index2frequency
        self.node_context_pairs= node_context_pairs
        self.shffule()
        self.count = 0
        self.epoch = 1

    def parse_node_type_mapping_txt(self,node_type_mapping_txt):
        #this method does not modify any class variables
        nodeid2type={}
        with open(node_type_mapping_txt) as f:
            for line in f:
                pair = [entry.strip() for entry in line.split('\t')]
                nodeid2type[pair[0]]=pair[1]
        return nodeid2type

    def parse_random_walk_txt(self,random_walk_txt,window_size):
        #this method does not modify any class variables
        #this will NOT make any <UKN> so don't use for NLP.
        word_and_counts = {}
        with open(random_walk_txt) as f:
            for line in f:
                sent = [word.strip() for word in line.split(' ')]
                for word in sent:       
                    if len(word) == 0:
                        continue
                    if word_and_counts.has_key(word):
                        word_and_counts[word] += 1
                    else:
                        word_and_counts[word] = 1


        print("The number of unique words:%d"%len(word_and_counts))
        index2token = dict((i, word) for i, word in enumerate(word_and_counts.keys()) )
        token2index = dict((v, k) for k, v in index2token.items())
        index2frequency = dict((token2index[word],freq) for word,freq in word_and_counts.items() )

        #word_word=scipy.sparse.lil_matrix((len(token2index), len(token2index)), dtype=np.int32)
        node_context_pairs = []#let's use naive way now

        print("window size %d"%window_size)

        with open(random_walk_txt) as f:
            for line in f:
                sent = [token2index[word.strip()] for word in line.split(' ') if word.strip() in token2index]
                sent_length=len(sent)
                for target_word_position,target_word_idx in enumerate(sent):
                    start=max(0,target_word_position-window_size)
                    end=min(sent_length,target_word_position+window_size+1)
                    context=sent[start:target_word_position]+sent[target_word_position+1:end+1]
                    for contex_word_idx in context:
                        node_context_pairs.append((target_word_idx,contex_word_idx))
                        #word_word[target_word_idx,contex_word_idx]+=1

        #word_word=word_word.tocsr()
        return index2token,token2index,word_and_counts,index2frequency,node_context_pairs

    def get_one_batch(self):
        if self.count == len(self.node_context_pairs):
            self.count=0
            self.epoch+=1
        node_context_pair = self.node_context_pairs[self.count]
        self.count+=1
        return node_context_pair

    def get_batch(self,batch_size):
        pairs = np.array([self.get_one_batch() for i in range(batch_size)])
        return pairs[:,0],pairs[:,1]

    def shffule(self):
        random.shuffle(self.node_context_pairs)

    def negative_sampler():
        pass
        #samplign rate is here
        #http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/

if __name__ == '__main__':
    #test code  
    dataset=Dataset(random_walk_txt="../data/test_data/random_walks.txt",node_type_mapping_txt="../data/test_data/node_type_mapings.txt",window_size=1)
    print(dataset.get_batch(2))
