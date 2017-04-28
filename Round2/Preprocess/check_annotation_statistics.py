from __future__ import division

import argparse
import sys
from os import listdir
from collections import OrderedDict
from tabulate import tabulate


def print_annotation_statistics(filename, save_filename=None):
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
    class_B_ann_count_dict = {}
    class_word_span_count_dict = {}
    class_vocab_dict = {}

    with open(filename, 'rb') as file:
        for line in file:
            line = line.decode('utf-8').strip()
            if len(line) > 0:
                if line.startswith('B-'):
                    ann_class = line.split()[0][2:]
                    ann_classes = ann_class.split('-AND-')
                    for i_class in ann_classes:
                        if i_class in class_B_ann_count_dict:
                            class_B_ann_count_dict[i_class] += 1
                        else:
                            class_B_ann_count_dict[i_class] = 1

                if line.startswith('B-') or line.startswith('I-'):
                    word = line.split()[3].lower()
                    ann_class = line.split()[0][2:]
                    ann_classes = ann_class.split('-AND-')
                    for i_class in ann_classes:
                        if i_class in class_word_span_count_dict:
                            class_word_span_count_dict[i_class] += 1
                            class_vocab_dict[i_class].add(word)
                        else:
                            class_word_span_count_dict[i_class] = 1
                            class_vocab_dict[i_class] = {word}


    # Sort
    class_B_ann_count_dict = OrderedDict(sorted(class_B_ann_count_dict.items()))
    class_word_span_count_dict = OrderedDict(sorted(class_word_span_count_dict.items()))
    class_vocab_dict = OrderedDict(sorted(class_vocab_dict.items()))

    # Sanity check
    assert len(class_B_ann_count_dict) == len(class_word_span_count_dict) == len(class_vocab_dict)

    # Print as a table using tabulate
    tab_output = []
    # Tabulate example: print(tabulate([['Alice', 24], ['Bob', 19]], headers=['Name', 'Age']))
    for i_class in class_B_ann_count_dict:
        class_ann_count = class_B_ann_count_dict[i_class]
        class_ann_span = class_word_span_count_dict[i_class]
        class_vocab_set = class_vocab_dict[i_class]

        tab_output.append([i_class, class_ann_span / class_ann_count, len(class_vocab_set)])

    if save_filename is not None:
        f_save = open(save_filename, 'wb')
        f_save.write(tabulate(tab_output, headers=['Class', 'Avg span', 'Vocab size']))
        f_save.close()
    else:
        print(tabulate(tab_output, headers=['Class', 'Avg span', 'Vocab size']))


if __name__ == "__main__":
    ####################################%%%%%%%%%%%%%%%%%%%%%
    parser = argparse.ArgumentParser(description='check_annotation_statistics.py')
    parser.add_argument('-f', type=str, help='File containing the NERSuite formated text (ending with .conll).', default='data/train-with-nersuite/train-nersuite.conll')
    parser.add_argument('-save', type=str, help='Provide a filename to save the table; default = None (no save but print to screen).', default=None)
    ####################################%%%%%%%%%%%%%%%%%%%%%
    args = parser.parse_args(sys.argv[1:])

    print("\nStart ... \n")
    print_annotation_statistics(filename=args.f, save_filename=args.save)
    print("\nDone!")