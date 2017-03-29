from __future__ import division

import argparse
import sys
from os import listdir
from collections import OrderedDict
from tabulate import tabulate


def print_annotation_class_counts(foldername, save_filename=None):
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
    ann_dict = {}
    sum = [0, 0, 0]
    for filename in listdir(foldername):
        if filename.startswith('train'):
            full_filepath = foldername + '/' + filename
            with open(full_filepath, 'rb') as file:
                for line in file:
                    line = line.decode('utf-8').strip()
                    if len(line) > 0 and line.startswith('B-'):
                        ann_class = line.split('\t')[0][2:]
                        if ann_class in ann_dict:
                            ann_dict[ann_class][0] += 1
                        else:
                            ann_dict[ann_class] = [1, 0, 0]
                        sum[0] += 1

        elif filename.startswith('devel'):
            full_filepath = foldername + '/' + filename
            with open(full_filepath, 'rb') as file:
                for line in file:
                    line = line.decode('utf-8').strip()
                    if len(line) > 0 and line.startswith('B-'):
                        ann_class = line.split('\t')[0][2:]
                        if ann_class in ann_dict:
                            ann_dict[ann_class][1] += 1
                        else:
                            ann_dict[ann_class] = [0, 1, 0]
                        sum[1] += 1

        elif filename.startswith('test'):
            full_filepath = foldername + '/' + filename
            with open(full_filepath, 'rb') as file:
                for line in file:
                    line = line.decode('utf-8').strip()
                    if len(line) > 0 and line.startswith('B-'):
                        ann_class = line.split('\t')[0][2:]
                        if ann_class in ann_dict:
                            ann_dict[ann_class][2] += 1
                        else:
                            ann_dict[ann_class] = [0, 0, 1]
                        sum[2] += 1

    # Sort
    ann_dict = OrderedDict(sorted(ann_dict.items()))

    # Add sum to the bottom
    ann_dict['SUM'] = sum

    # Print as a table using tabulate
    tab_output = []
    # Tabulate example: print(tabulate([['Alice', 24], ['Bob', 19]], headers=['Name', 'Age']))
    for ann_class, (train_count, devel_count, test_count) in ann_dict.items():
        tab_output.append([ann_class, train_count, devel_count, test_count])

    if save_filename is not None:
        f_save = open(save_filename, 'wb')
        f_save.write(tabulate(tab_output, headers=['Class', 'Train', 'Devel', 'Test']))
        f_save.close()
    else:
        print(tabulate(tab_output, headers=['Class', 'Train', 'Devel', 'Test']))





if __name__ == "__main__":
    ####################################%%%%%%%%%%%%%%%%%%%%%
    parser = argparse.ArgumentParser(description='check_class_distribution.py')
    parser.add_argument('-folder', type=str, help='Folder containing the NERSuite formated files (ending with .conll).', default='data/combined')
    parser.add_argument('-save', type=str, help='Give a filename to save the table; default = None (no save but print to screen).', default=None)
    ####################################%%%%%%%%%%%%%%%%%%%%%
    args = parser.parse_args(sys.argv[1:])

    print("\nStart ... ")
    print_annotation_class_counts(args.folder, args.save)
    print("\nDone!")