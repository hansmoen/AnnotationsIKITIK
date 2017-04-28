from __future__ import division

import argparse
import sys
from os import listdir
import w2v_handler
from tabulate import tabulate
import numpy as np


def convert_from_words_to_num(model_filename, mapping_filename, save_name, lowercasing=True):

    num_to_word_dict = {}
    num_list = []
    print('\nFetching vocabulary from mapping file ... ')
    with open(mapping_filename, 'rb') as file:
        for line in file:
            line = line.decode('utf-8').strip()
            line_parts = line.split()
            num_id = line_parts[0]
            word = line_parts[1]
            num_to_word_dict[num_id] = word
            num_list.append(num_id)

    print('\nLoading word space model ... ')
    word_model = w2v_handler.W2vModel()
    word_model.load_w2v_model(model_filename)

    print('\nMaking the dictionary ... ')
    num_vocab = w2v_handler.make_vocab(num_list)
    num_vectors = np.empty((len(num_vocab), word_model.get_dim()), dtype=np.float32)
    for i_num in num_vocab:
        i_word = num_to_word_dict[i_num].lower()
        if lowercasing:
            i_word = i_word.lower()
        if word_model.in_vocab(i_word):
            word_vec = word_model.get_vec(i_word)
            num_vectors[num_vocab[i_num].index] = word_vec

    print('\nSave the model ... ')
    filename_no_ending = save_name
    w2v_handler.save_word2vec_format(vectors=num_vectors, vocab=num_vocab, save_filename=filename_no_ending + '.w2v.bin',
                                     fvocab=filename_no_ending + '.w2v.vocab', binary=True)



if __name__ == "__main__":
    ####################################%%%%%%%%%%%%%%%%%%%%%
    parser = argparse.ArgumentParser(description='map_embeddings.py')
    parser.add_argument('-model', type=str, help='Word model to fetch embeddings from.', required=True)
    parser.add_argument('-mapping', type=str, help='File containing the mappings from numbers to words.', required=True)
    parser.add_argument('-save', type=str, help='Save the model and vocabulary.', required=True)
    ####################################%%%%%%%%%%%%%%%%%%%%%
    args = parser.parse_args(sys.argv[1:])

    print("\nStart ... \n")
    convert_from_words_to_num(model_filename=args.model, mapping_filename=args.mapping, save_name=args.save)
    print("\nDone!")


