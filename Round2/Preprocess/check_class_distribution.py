from __future__ import division

import argparse
import sys
from os import listdir
from collections import OrderedDict
from tabulate import tabulate


def print_annotation_class_counts(foldername, save_filename=None, mode='org'):
    """
    Format:
    O	308	313	Hyvin	hyvin	ADV	O
    O	314	318	meni	menna	VERB	O
    O	318	319	.	.	PUNCT	O

    O	320	326	Nahnyt	nahda	VERB	O
    O	327	332	vahan	vahan	ADV	O
    B-Harhaisuus	333	340	harhoja	harha	NOUN	O
    O	340	341	.	.	PUNCT	O
    """
    class_dict = {}
    class_sum = [0, 0, 0]

    token_count = [0, 0, 0]
    sent_count = [0, 0, 0]
    doc_count = [1, 1, 1]

    for filename in listdir(foldername):

        if filename.startswith('train'):
            full_filepath = foldername + '/' + filename
            with open(full_filepath, 'rb') as file:
                empty_sent = False
                sent_labels = []
                for line in file:
                    line = line.decode('utf-8').strip()
                    if len(line) == 0:
                        if empty_sent:
                            doc_count[0] += 1
                        else:
                            sent_count[0] += 1
                        empty_sent = True
                        if mode == 'sent':
                            sent_labels = set(sent_labels)
                        for l in sent_labels:
                            if l in class_dict:
                                class_dict[l][0] += 1
                            else:
                                class_dict[l] = [1, 0, 0]
                            class_sum[0] += 1
                        sent_labels = []
                    elif len(line) > 0:
                        empty_sent = False
                        if line.startswith('B-'):
                            ann_class = line.split()[0][2:]
                            if mode in ['single', 'sent']:
                                ann_classes = ann_class.split('-AND-')
                                for i_class in ann_classes:
                                    sent_labels.append(i_class)
                            elif mode == 'org':
                                sent_labels.append(ann_class)
                        token_count[0] += 1

        elif filename.startswith('devel'):
            full_filepath = foldername + '/' + filename
            with open(full_filepath, 'rb') as file:
                empty_sent = False
                sent_labels = []
                for line in file:
                    line = line.decode('utf-8').strip()
                    if len(line) == 0:
                        if empty_sent:
                            doc_count[1] += 1
                        else:
                            sent_count[1] += 1
                        empty_sent = True
                        if mode == 'sent':
                            sent_labels = set(sent_labels)
                        for l in sent_labels:
                            if l in class_dict:
                                class_dict[l][1] += 1
                            else:
                                class_dict[l] = [0, 1, 0]
                            class_sum[1] += 1
                        sent_labels = []
                    elif len(line) > 0:
                        empty_sent = False
                        if line.startswith('B-'):
                            ann_class = line.split()[0][2:]
                            if mode in ['single', 'sent']:
                                ann_classes = ann_class.split('-AND-')
                                for i_class in ann_classes:
                                    sent_labels.append(i_class)
                            elif mode == 'org':
                                sent_labels.append(ann_class)
                        token_count[1] += 1

        elif filename.startswith('test'):
            full_filepath = foldername + '/' + filename
            with open(full_filepath, 'rb') as file:
                empty_sent = False
                sent_labels = []
                for line in file:
                    line = line.decode('utf-8').strip()
                    if len(line) == 0:
                        if empty_sent:
                            doc_count[2] += 1
                        else:
                            sent_count[2] += 1
                        empty_sent = True
                        if mode == 'sent':
                            sent_labels = set(sent_labels)
                        for l in sent_labels:
                            if l in class_dict:
                                class_dict[l][2] += 1
                            else:
                                class_dict[l] = [0, 0, 1]
                            class_sum[2] += 1
                        sent_labels = []
                    elif len(line) > 0:
                        empty_sent = False
                        if line.startswith('B-'):
                            ann_class = line.split()[0][2:]
                            if mode in ['single', 'sent']:
                                ann_classes = ann_class.split('-AND-')
                                for i_class in ann_classes:
                                    sent_labels.append(i_class)
                            elif mode == 'org':
                                sent_labels.append(ann_class)
                        token_count[2] += 1

    # Sort
    class_dict = OrderedDict(sorted(class_dict.items()))

    # Add sum to the bottom
    class_dict['SUM'] = class_sum

    # Print as a table using tabulate
    tab_output = []
    # Tabulate example: print(tabulate([['Alice', 24], ['Bob', 19]], headers=['Name', 'Age']))
    for ann_class, (train_count, devel_count, test_count) in class_dict.items():
        tab_output.append([ann_class, train_count, devel_count, test_count, sum([train_count, devel_count, test_count])])

    # Add some word, sentence and document statistics
    tab_output.append(['-', '-', '-', '-', '-'])
    tab_output.append(['Tokens'] + token_count + [sum(token_count)])
    tab_output.append(['Sentences'] + sent_count + [sum(sent_count)])
    tab_output.append(['Documents'] + doc_count + [sum(doc_count)])

    if save_filename is not None:
        f_save = open(save_filename, 'wb')
        f_save.write(tabulate(tab_output, headers=['Class', 'Train', 'Devel', 'Test', 'Total']))
        f_save.close()
    else:
        print(tabulate(tab_output, headers=['Class', 'Train', 'Devel', 'Test', 'Total']))


if __name__ == "__main__":
    ####################################%%%%%%%%%%%%%%%%%%%%%
    parser = argparse.ArgumentParser(description='check_class_distribution.py')
    parser.add_argument('-folder', type=str, help='Folder containing the NERSuite formated files (ending with .conll).', default='data/train-with-nersuite')
    parser.add_argument('-save', type=str, help='Provide a filename to save the table; default = None (no save but print to screen).', default=None)
    parser.add_argument('-mode', type=str, help='How to count classes; choices={"org", "single", "sent"}, where org:original classes, single:single classes only, sent:sentence level.', choices=['org', 'single', 'sent'], default='org')
    ####################################%%%%%%%%%%%%%%%%%%%%%
    args = parser.parse_args(sys.argv[1:])

    print("\nStart ... \n")
    print_annotation_class_counts(foldername=args.folder, save_filename=args.save, mode=args.mode)
    print("\nDone!")