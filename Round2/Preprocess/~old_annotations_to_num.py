from __future__ import division

import argparse
import sys
from os import listdir


def map_to_num(conll_folder, save_num_folder, save_word_mappings, save_ann_mappings):
    # Go through the data and create dictionaries from the unique words
    word_dict = {}
    ann_dict = {}
    for filename in listdir(conll_folder):
        if filename.endswith('.conll'):
            fn_text = conll_folder + '/' + filename
            with open(fn_text, 'rb') as file:
                for line in file:
                    line = line.decode('utf-8').strip()
                    if len(line) > 0:
                        line_parts = line.split('\t')
                        word = line_parts[0]
                        if word in word_dict:
                            word_dict[word] += 1
                        else:
                            word_dict[word] = 1

                        ann = line_parts[1]
                        if len(ann) > 1: # Do not save the O-s
                            ann_prefix = ann[:2]
                            ann_class = ann[2:]
                            if ann_class in ann_dict:
                                ann_dict[ann_class] += 1
                            else:
                                ann_dict[ann_class] = 1


    # Save mappings
    f_word_mappings = open(save_word_mappings, 'wb')
    for i, (word, count) in enumerate(word_dict.items()):
        f_word_mappings.write((str(i + 1) + '\t' + word + '\t' + str(count) + '\n').encode('utf-8'))
        word_dict[word] = (i + 1, count)
    f_word_mappings.close()
    f_ann_mappings = open(save_ann_mappings, 'wb')
    for i, (ann, count) in enumerate(ann_dict.items()):
        f_ann_mappings.write((str(i + 1) + '\t' + ann + '\t' + str(count) + '\n').encode('utf-8'))
        ann_dict[ann] = (i + 1, count)
    f_ann_mappings.close()


    # Second itteration, replace words with numbers, save

    #for filename in listdir(conll_foldername):
    #    fn_text = conll_foldername + '/' + filename
    #    if filename.lower().endswith('.txt'):
    for filename in listdir(conll_folder):
        if filename.endswith('.conll'):
            fn_text = conll_folder + '/' + filename
            f_num_conll = open(save_num_folder + '/' + filename + '.num', 'wb')
            with open(fn_text, 'rb') as file:
                for line in file:
                    line = line.decode('utf-8').strip()
                    if len(line) > 0:
                        line_parts = line.split('\t')
                        word = line_parts[0]
                        ann = line_parts[1]
                        if len(ann) > 1:  # Check if it is an annotation, not an empty O
                            ann_prefix = ann[:2]
                            ann_class = ann[2:]
                            f_num_conll.write((str(word_dict[word][0]) + '\t' + ann_prefix + str(ann_dict[ann_class][0]) + '\n').encode('utf-8'))
                        else:
                            # Should be a O
                            f_num_conll.write((str(word_dict[word][0]) + '\tO' + '\n').encode('utf-8'))
                    else:
                        f_num_conll.write(('\n').encode('utf-8'))
            f_num_conll.close()




if __name__ == "__main__":
    ####################################%%%%%%%%%%%%%%%%%%%%%
    parser = argparse.ArgumentParser(description='annotations_to_num.py')
    parser.add_argument('-conll', type=str, help='Folder with the original text documents (.conll).', default='data/test_combine')  # required=True)
    parser.add_argument('-save', type=str, help='Folder for where to save the enumerated conll file (.conll.num).', default='data/test_enumerate') #required=True)
    parser.add_argument('-word_mappings', type=str, help='Filename to save the word-to-num mappings.', default='data/test_enumerate/word_mappings.txt') #required=True)
    parser.add_argument('-ann_mappings', type=str, help='Filename for saving the annotaions-to-num mappings', default='data/test_enumerate/ann_mappings.txt') #required=True)
    ####################################%%%%%%%%%%%%%%%%%%%%%
    args = parser.parse_args(sys.argv[1:])

    print("Start ... ")
    map_to_num(conll_folder=args.conll, save_num_folder=args.save, save_word_mappings=args.word_mappings, save_ann_mappings=args.ann_mappings)
    print("Done!\n")