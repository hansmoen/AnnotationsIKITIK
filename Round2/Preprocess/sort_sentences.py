from __future__ import division

import argparse
import sys


def sort_file(filename):

    sentences = []

    with open(filename, 'rb') as file:
        for line in file:
            line = line.decode('utf-8').replace('\n', '')
            if len(line) > 0:
                sentences.append(line)

    for sent in sorted(sentences):
        print(sent)


if __name__ == "__main__":
    ####################################%%%%%%%%%%%%%%%%%%%%%
    parser = argparse.ArgumentParser(description='sort_sentences.py')
    parser.add_argument('-f', type=str, help='File to sort.', default='data/ANN-classes/sekavuus.txt')
    ####################################%%%%%%%%%%%%%%%%%%%%%
    args = parser.parse_args(sys.argv[1:])

    print("\nStart ... \n")
    sort_file(filename=args.f)
    print("\nDone!")