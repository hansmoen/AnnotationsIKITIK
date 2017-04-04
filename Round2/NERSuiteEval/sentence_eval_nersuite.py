from __future__ import print_function
from __future__ import division

import argparse
import sys
import numpy as np
from sklearn.metrics import f1_score


unique_label_set = set()

def read_conll_file(filename, label_index):
    all_sentence_label_list = []
    in_sent = False
    sent_labels = set()
    with open(filename, 'rb') as file:
        for line in file:
            line = line.decode('utf-8').strip()

            if len(line) > 0:
                in_sent = True
                line_parts = line.split()
                label_string = line_parts[label_index]

                if label_string != 'O': # Do not include these
                    label_list = label_string[2:].split('-AND-')
                    for label in label_list:
                        sent_labels.add(label)
                        unique_label_set.add(label)
            else: # End of sentence ...
                all_sentence_label_list.append(sent_labels)
                in_sent = False
                sent_labels = set()
    if in_sent:
        # In case the file does not end with a empty line ...
        all_sentence_label_list.append(sent_labels)
        in_sent = False
        sent_labels = set()
    return all_sentence_label_list


def evaluate_nersuite(predictions_filename, gold_filename):

    # NERSuite predictions, labes are at the right, element 6
    pred = read_conll_file(predictions_filename, 6)

    # Gold, labes are at the left, element 0
    gold = read_conll_file(gold_filename, 0)

    # Sentence counts has to match!
    assert len(pred) == len(gold)

    # Make label to index mappings for use in N hot arrays for sklearn metrics
    label2index = {}
    index2label = {}
    for index, label in enumerate(sorted(unique_label_set)):
        label2index[label] = index
        index2label[index] = label

    # Convert into 2d numpy arrays of n hots
    pred_np_array = np.zeros((len(pred), len(unique_label_set)), dtype=np.int8)
    gold_np_array = np.zeros((len(pred), len(unique_label_set)), dtype=np.int8)

    for i in range(0, len(pred)):
        # Predictions
        i_pred_sent_label_set = pred[i]
        for pred_label in i_pred_sent_label_set:
            pred_np_array[i, label2index[pred_label]] = 1

        # Gold
        i_gold_sent_label_set = gold[i]
        for gold_label in i_gold_sent_label_set:
            gold_np_array[i, label2index[gold_label]] = 1

    # Calculate scores
    f1_score_macro = f1_score(gold_np_array, pred_np_array, average='macro')
    print('f1_score_macro', f1_score_macro)
    f1_score_micro = f1_score(gold_np_array, pred_np_array, average='micro')
    print('f1_score_micro', f1_score_micro)
    f1_score_weighted = f1_score(gold_np_array, pred_np_array, average='weighted')
    print('f1_score_weighted', f1_score_weighted)
    f1_score_samples = f1_score(gold_np_array, pred_np_array, average='samples')
    print('f1_score_samples', f1_score_samples)


    f1_score_class_list = f1_score(gold_np_array, pred_np_array, average=None)
    print('\nf1 score for individual classes')
    for i_class, class_f1_score in enumerate(f1_score_class_list):
        print('\t' + index2label[i_class] + ': ' + str(class_f1_score))


if __name__ == "__main__":
    ####################################%%%%%%%%%%%%%%%%%%%%%
    parser = argparse.ArgumentParser(description='sentence_eval_nersuite.py')
    parser.add_argument('-pred', type=str, help='Filename for predicted data.', required=True) # default='data/pred.txt')
    parser.add_argument('-gold', type=str, help='Filename for gold data.', required=True) # default='data/gold.txt')
    ####################################%%%%%%%%%%%%%%%%%%%%%
    args = parser.parse_args(sys.argv[1:])


    print('\nStart ... \n')

    evaluate_nersuite(predictions_filename=args.pred, gold_filename=args.gold)

    print('\nDone!')