from __future__ import division

import argparse
import sys
from os import listdir
import numpy as np


class X_y_dataHandler:

    def __init__(self, ann_set, include_negatives=0):

        self.X_word_np = None
        self.X_lemma_np = None
        self.X_pos_np = None
        self.y_n_hot_np = None

        self.X_word_table = []
        self.X_lemma_table = []
        self.X_pos_table = []
        self.y_ann_table = []

        self.X_max_len = 0
        self.X_max_word_value = 0
        self.X_max_lemma_value = 0
        self.X_max_pos_value = 0
        self.y_max_len = 0
        self.y_max_value = 0

        self.ann_set = ann_set # choices = ['kipu', 'sekavuus', 'infektio']
        self.o_label_id = self.get_o_label_id()
        self.include_negatives = include_negatives

    def get_o_label_id(self):
        """
        Get the ID (placeholder) for the O label in the given annotation/data set.
        Hardcoded based on the current data set.
        ann_set choices = ['kipu', 'sekavuus', 'infektio'].
        :return: O label ID
        """
        if self.ann_set == 'kipu':
            return 7
        elif self.ann_set == 'sekavuus':
            return 18
        elif self.ann_set == 'infektio':
            return 16
        else:
            print('NOTE, O labels will not be removed!')
            return -1

    def load_data_set(self, filename):
        """
        Format:
        121 177 13	46 97 2	7 7 8	65 156 2	|-| 1 2 6 8
        121 177 13	46 97 2	7 7 8	65 156 2	|-| 5
        """
        with open(filename, 'rb') as file:
            for line in file:
                line = line.decode('utf-8').strip()
                text, ann = line.split('\t|-|\t')
                word_lemma_pos_list = text.split('\t')
                ann_list = map(int, ann.split(' '))

                # X
                words = []
                lemmas = []
                pos = []
                for i_word_lemma_pos in word_lemma_pos_list:
                    i_word_int, i_lemma_int, i_pos_int = map(int, i_word_lemma_pos.split(' '))
                    words.append(i_word_int)
                    lemmas.append(i_lemma_int)
                    pos.append(i_pos_int)

                    if i_word_int > self.X_max_word_value:
                        self.X_max_word_value = i_word_int
                    if i_lemma_int > self.X_max_lemma_value:
                        self.X_max_lemma_value = i_lemma_int
                    if i_pos_int > self.X_max_pos_value:
                        self.X_max_pos_value = i_pos_int

                if len(words) > self.X_max_len:
                    self.X_max_len = len(words)

                self.X_word_table.append(words)
                self.X_lemma_table.append(lemmas)
                self.X_pos_table.append(pos)

                # Y
                self.y_ann_table.append(ann_list)
                max_ann = max(ann_list)
                if max_ann > self.y_max_value:
                    self.y_max_value = max_ann
                if len(ann_list) > self.y_max_len:
                    self.y_max_len = len(ann_list)

        """
                print('\n')
                print('words', words)
                print('lemmas', lemmas)
                print('pos', pos)
                print('anns', ann_list)
        print('\n\n-------------------------------')
        print('X_max_len:', self.X_max_len)
        print('X_max_word_value', self.X_max_word_value)
        print('y_max_len', self.y_max_len)
        print('y_max_value', self.y_max_value)
        """

    def make_numpy_arrays(self, X_dim_preset, y_dim_preset, padding_side='right'):
        X_word_padded = []
        X_lemma_padded = []
        X_pos_padded = []
        for i in range(0, len(self.X_word_table)):
            if len(self.X_word_table[i]) < X_dim_preset:
                padding_count = X_dim_preset - len(self.X_word_table[i])  # Should be the same for word, lemma and pos
                if padding_side == 'right':
                    # Pad with zeros from the right
                    X_word_padded.append(self.X_word_table[i] + [0] * padding_count)
                    X_lemma_padded.append(self.X_lemma_table[i] + [0] * padding_count)
                    X_pos_padded.append(self.X_pos_table[i] + [0] * padding_count)
                if padding_side == 'left':
                    # Pad with zeros from the left
                    X_word_padded.append([0] * padding_count + self.X_word_table[i])
                    X_lemma_padded.append([0] * padding_count + self.X_lemma_table[i])
                    X_pos_padded.append([0] * padding_count + self.X_pos_table[i])
            else:
                # Keep the leftmost part of the row, discard the rest
                X_word_padded.append(self.X_word_table[i][:X_dim_preset])
                X_lemma_padded.append(self.X_lemma_table[i][:X_dim_preset])
                X_pos_padded.append(self.X_pos_table[i][:X_dim_preset])
        self.X_word_np = np.array(X_word_padded, dtype=np.int32)
        self.X_lemma_np = np.array(X_lemma_padded, dtype=np.int32)
        self.X_pos_np = np.array(X_pos_padded, dtype=np.int32)
        #print(self.X_word_np) #------

        # N hot vectors reflecting the y values

        y_dim = self.y_max_value
        if y_dim_preset > 0:
            y_dim = y_dim_preset
        self.y_n_hot_np = np.zeros([len(self.y_ann_table), y_dim], dtype=np.int32)
        for i in range(0, len(self.y_ann_table)):
            for hot_index in self.y_ann_table[i]:
                if (self.include_negatives) or (hot_index != self.o_label_id):
                    self.y_n_hot_np[i, hot_index - 1] = 1
        #print(self.y_one_hot_np)  # ------

        # Check if everything went as planned ...
        assert(len(self.X_word_np) == len(self.X_lemma_np) == len(self.X_pos_np) == len(self.y_ann_table) == len(self.y_n_hot_np))
        assert(self.X_word_np.shape == self.X_lemma_np.shape == self.X_pos_np.shape)

    def get_X_word_np_array(self):
        return self.X_word_np

    def get_X_lemma_np_array(self):
        return self.X_lemma_np

    def get_X_pos_np_array(self):
        return self.X_pos_np

    def get_y_n_hot_np_array(self):
        return self.y_n_hot_np

    def get_X_max_len(self):
        return self.X_max_len

    def get_X_max_word_value(self):
        return self.X_max_word_value

    def get_X_max_lemma_value(self):
        return self.X_max_lemma_value

    def get_X_max_pos_value(self):
        return self.X_max_pos_value

    def get_y_max_len(self):
        return self.y_max_len

    def get_y_max_value(self):
        return self.y_max_value

    def get_size(self):
        return len(self.X_word_table)



if __name__ == "__main__":

    # sent
    data = X_y_dataHandler('')
    data.load_data_set('../Preprocess/data/train-with-keras/sent-train-nersuite.txt')
    print('Set size: ' + str(data.get_size()))

    print('Max X word value: ' + str(data.get_X_max_word_value()))
    print('Max X lemma value: ' + str(data.get_X_max_lemma_value()))
    print('Max X pos value: ' + str(data.get_X_max_pos_value()))
    print('Max y value: ' + str(data.get_y_max_value()))
    data.make_numpy_arrays(10, 10)
    # doc
    #data = X_y_data_handler('../Preprocess/data/train-with-keras/doc-train-nersuite.txt', 10)