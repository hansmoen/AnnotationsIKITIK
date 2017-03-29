from __future__ import division

import argparse
import sys
from os import listdir
import random
random.seed(1337)


def map_to_num(conll_folder, save_num_folder, simple): # save_word_mappings, save_lemma_mappings, save_pos_mappings, save_ann_mappings,
    # Go through the data and create dictionaries from the unique words
    word_dict = {}
    lemma_dict = {}
    pos_dict = {}
    ann_dict = {}

    for filename in listdir(conll_folder):
        if filename.endswith('.conll'):
            fn_text = conll_folder + '/' + filename
            with open(fn_text, 'rb') as file:
                for line in file:
                    line = line.decode('utf-8').strip()
                    if len(line) > 0:
                        line_parts = line.split('\t')
                        #print(line_parts) #----------
                        word = line_parts[3]
                        lemma = line_parts[4]
                        pos = line_parts[5]
                        ann = line_parts[0]

                        if word in word_dict:
                            word_dict[word] += 1
                        else:
                            word_dict[word] = 1
                        if lemma in lemma_dict:
                            lemma_dict[lemma] += 1
                        else:
                            lemma_dict[lemma] = 1
                        if pos in pos_dict:
                            pos_dict[pos] += 1
                        else:
                            pos_dict[pos] = 1

                        if len(ann) > 1:
                            ann_prefix = ann[:2]
                            ann_class = ann[2:]
                            ann_class_parts = ann_class.split('-AND-')
                            for i_ann_class in ann_class_parts:
                                if i_ann_class in ann_dict:
                                    ann_dict[i_ann_class] += 1
                                else:
                                    ann_dict[i_ann_class] = 1
                        else:
                            ann_class = ann[0]
                            if ann_class in ann_dict:
                                ann_dict[ann_class] += 1
                            else:
                                ann_dict[ann_class] = 1


    # Save mappings
    save_word_mappings = save_num_folder + '/word-mappings.txt'
    save_lemma_mappings= save_num_folder + '/lemma-mappings.txt'
    save_pos_mappings= save_num_folder + '/pos-mappings.txt'
    save_ann_mappings= save_num_folder + '/ann-mappings.txt'

    word_key_list = list(word_dict.keys())
    random.shuffle(word_key_list)
    word_key_values_shuffled = [(key, word_dict[key]) for key in word_key_list]
    f_word_mappings = open(save_word_mappings, 'wb')
    for i, (word, count) in enumerate(word_key_values_shuffled):
        f_word_mappings.write((str(i + 1) + '\t' + word + '\t' + str(count) + '\n').encode('utf-8'))
        word_dict[word] = (i + 1, count)
    f_word_mappings.close()

    ann_key_list = list(ann_dict.keys())
    random.shuffle(ann_key_list)
    ann_key_values_shuffled = [(key, ann_dict[key]) for key in ann_key_list]
    f_ann_mappings = open(save_ann_mappings, 'wb')
    for i, (ann, count) in enumerate(ann_key_values_shuffled):
        f_ann_mappings.write((str(i + 1) + '\t' + ann + '\t' + str(count) + '\n').encode('utf-8'))
        ann_dict[ann] = (i + 1, count)
    f_ann_mappings.close()

    if not simple:
        lemma_key_list = list(lemma_dict.keys())
        random.shuffle(lemma_key_list)
        lemma_key_values_shuffled = [(key, lemma_dict[key]) for key in lemma_key_list]
        f_lemma_mappings = open(save_lemma_mappings, 'wb')
        for i, (lemma, count) in enumerate(lemma_key_values_shuffled):
            f_lemma_mappings.write((str(i + 1) + '\t' + lemma + '\t' + str(count) + '\n').encode('utf-8'))
            lemma_dict[lemma] = (i + 1, count)
        f_lemma_mappings.close()

        pos_key_list = list(pos_dict.keys())
        random.shuffle(pos_key_list)
        pos_key_values_shuffled = [(key, pos_dict[key]) for key in pos_key_list]
        f_pos_mappings = open(save_pos_mappings, 'wb')
        for i, (pos, count) in enumerate(pos_key_values_shuffled):
            f_pos_mappings.write((str(i + 1) + '\t' + pos + '\t' + str(count) + '\n').encode('utf-8'))
            pos_dict[pos] = (i + 1, count)
        f_pos_mappings.close()



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

                        if simple: # Simple version
                            if len(line) > 0:
                                line_parts = line.split('\t')
                                word = line_parts[3]
                                #lemma = line_parts[4]
                                #pos = line_parts[5]
                                ann = line_parts[0]
                                if len(ann) > 1:  # Check if it is an annotation, not an empty O
                                    ann_prefix = ann[:2]
                                    ann_class = ann[2:]

                                    #---------#
                                    ann_class_parts = ann_class.split('-AND-')
                                    ann_classes_str = '-AND-'.join([str(ann_dict[i_ann_class][0]) for i_ann_class in ann_class_parts])
                                    # ---------#
                                    f_num_conll.write((str(word_dict[word][0]) + '\t' + ann_prefix + ann_classes_str + '\n').encode('utf-8'))
                                else:
                                    ann_class = ann[0] # Ann should be O
                                    f_num_conll.write((str(word_dict[word][0]) + '\t' + 'O' + '\n').encode('utf-8'))
                        else: # Full version
                            start_word_index = line_parts[1]
                            end_word_index = line_parts[2]
                            word = line_parts[3]
                            lemma = line_parts[4]
                            pos = line_parts[5]
                            ann = line_parts[0]
                            last_element = line_parts[6]
                            if len(ann) > 1:  # Check if it is an annotation, not an empty O
                                ann_prefix = ann[:2]
                                ann_class = ann[2:]
                                # ---------#
                                ann_class_parts = ann_class.split('-AND-')
                                ann_classes_str = '-AND-'.join([str(ann_dict[i_ann_class][0]) for i_ann_class in ann_class_parts])
                                # ---------#
                                #f_num_conll.write((str(word_dict[word][0]) + '\t' + ann_prefix + str(ann_dict[ann_class][0]) + '\n').encode('utf-8'))
                                f_num_conll.write((ann_prefix + ann_classes_str + '\t' + str(start_word_index) + '\t' + end_word_index + '\t' + str(word_dict[word][0]) + '\t' + str(lemma_dict[lemma][0]) + '\t' + str(pos_dict[pos][0]) + '\t' + last_element + '\n').encode('utf-8'))
                            else:
                                ann_class = ann[0] # Should be 'O', here we do the mapping to num also for 'O'
                                f_num_conll.write((str(ann_dict[ann_class][0]) + '\t' + str(start_word_index) + '\t' + end_word_index + '\t' + str(word_dict[word][0]) + '\t' + str(lemma_dict[lemma][0]) + '\t' + str(pos_dict[pos][0]) + '\t' + last_element + '\n').encode('utf-8'))
                    else:
                        f_num_conll.write(('\n').encode('utf-8'))
            f_num_conll.close()




if __name__ == "__main__":
    ####################################%%%%%%%%%%%%%%%%%%%%%
    parser = argparse.ArgumentParser(description='annotations2num_format.py')
    parser.add_argument('-conll', type=str, help='Folder with the original text documents (.conll).', default='data/train-with-nersuite')  # required=True)
    parser.add_argument('-save', type=str, help='Folder for where to save the enumerated conll file (.conll.num).', default='data/train-with-nersuite-num') #required=True)
    #parser.add_argument('-word_mappings', type=str, help='Filename to save the word-to-num mappings.', default='data/train-with-nersuite-num/word-mappings.txt') #required=True)
    #parser.add_argument('-lemma_mappings', type=str, help='Filename for saving the lemma-to-num mappings', default='data/train-with-nersuite-num/lemma-mappings.txt') #required=True)
    #parser.add_argument('-pos_mappings', type=str, help='Filename for saving the part-of-speech-to-num mappings', default='data/train-with-nersuite-num/pos-mappings.txt')  # required=True)
    #parser.add_argument('-ann_mappings', type=str, help='Filename for saving the annotaions-to-num mappings', default='data/train-with-nersuite-num/ann-mappings.txt')  # required=True)
    parser.add_argument('-simple', type=int, help='Use simple format, i.e. only the word and the annotation per line; default = 0 (False).', default=0)  # required=True)
    ####################################%%%%%%%%%%%%%%%%%%%%%
    args = parser.parse_args(sys.argv[1:])

    print("\nStart ... ")
    map_to_num(conll_folder=args.conll, save_num_folder=args.save, simple=args.simple)
    print("\nDone!")