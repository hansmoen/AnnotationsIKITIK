from __future__ import division

import argparse
import sys
import random



if __name__ == "__main__":
    ####################################%%%%%%%%%%%%%%%%%%%%%
    parser = argparse.ArgumentParser(description='shuffle_conll_sentences.py')
    parser.add_argument('-conll', type=str, help='Conll file to shuffle, one word per line, sentences separated by empty lines.', required=True)
    parser.add_argument('-save', type=str, help='Save file for the shuffled conll file.', required=True)
    ####################################%%%%%%%%%%%%%%%%%%%%%
    args = parser.parse_args(sys.argv[1:])

    print("Start ... ")

    conll_as_list = []
    sentence = []
    with open(args.conll, 'rb') as conll_file:
        for line in conll_file:
            line = line.decode('utf-8').strip()
            if len(line) == 0:
                if len(sentence) > 0:
                    conll_as_list.append(sentence)
                    sentence = []
            else:
                sentence.append(line)

    # Shuffle order
    random.shuffle(conll_as_list)

    # Save the shuffeled conll sentences
    save_file = open(args.save, 'wb')
    for sent in conll_as_list:
        for word_line in sent:
            save_file.write((word_line + '\n').encode('utf-8'))
        save_file.write(('\n').encode('utf-8'))
    save_file.close()

    print("Done!\n")