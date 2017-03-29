import argparse
from os import sys
from os import listdir
from os.path import isfile
import operator

###########################
# Some static parameteres #
###########################
FILE_EXTENSION = "nersuite"
###########################

class GetUniqueAnnotations:
    def __init__(self, conll_folder):
        self._conll_folder = conll_folder
        self._unique_annotations = {}

    def get_unique_multi_annotations_words(self):
        # Go through each CoNLL file line by line
        #print("FOLDER:::: " + self._conll_folder) #-------
        for filename in listdir(self._conll_folder):
            i_conll_and_fullpath = self._conll_folder + "/" + filename
            #print("iFILE:::: " + i_conll_and_fullpath) #-------
            if isfile(i_conll_and_fullpath) and filename.lower().endswith("." + FILE_EXTENSION):

                with open(i_conll_and_fullpath, 'rt') as f_conll:
                    for line in f_conll:
                        #print("iLINE:::: " + line) #-------
                        line_segments = line.split()
                        #print(line_segments) #---------------

                        if len(line_segments) >= 4:
                            #print(line_segments) #---------------
                            i_label = line_segments[0]
                            i_word = line_segments[3]
                            i_lemma = line_segments[4]

                            if i_label != "O" and "_+_" in i_label:
                                i_annotation = i_label.replace("B-", "").replace("I-", "") + "\t" + i_word.lower()
                                #print(i_annotation) #------
                                if i_annotation in self._unique_annotations:
                                    self._unique_annotations[i_annotation] += 1
                                else:
                                    self._unique_annotations[i_annotation] = 1

    def print_stats_sort_by_freq(self):
        # Sort and print
        sorted_unique_annotations = sorted(self._unique_annotations.items(), key=operator.itemgetter(1), reverse=True)
        i = 1
        for i_annotation, i_freq in sorted_unique_annotations:
            print(str(i) + "\t" + i_annotation + "\t" + str(i_freq))
            i += 1

    def print_stats_sort_by_name(self):
        # Sort and print
        sorted_unique_annotations = sorted(self._unique_annotations.items(), key=operator.itemgetter(0))
        i = 1
        for i_annotation, i_freq in sorted_unique_annotations:
            print(str(i) + "\t" + i_annotation + "\t" + str(i_freq))
            i += 1


#################################
    def get_unique_multi_annotations_sentences(self):
        # Go through each CoNLL file line by line
        #print("FOLDER:::: " + self._conll_folder) #-------
        for filename in listdir(self._conll_folder):
            i_conll_and_fullpath = self._conll_folder + "/" + filename
            #print("iFILE:::: " + i_conll_and_fullpath) #-------
            if isfile(i_conll_and_fullpath) and filename.lower().endswith("." + FILE_EXTENSION):

                with open(i_conll_and_fullpath, 'rt') as f_conll:

                    i_sentence = []
                    i_sentence_has_multi_annotation = False
                    for line in f_conll:
                        #print("iLINE:::: " + line) #-------
                        if line == "\n":
                            if i_sentence_has_multi_annotation:
                                ###############################
                                print("\n" + ' '.join(i_sentence))
                                ###############################
                                i_sentence_has_multi_annotation = False
                            i_sentence = []

                        else:
                            line_segments = line.split()
                            #print(line_segments) #---------------

                            if len(line_segments) >= 4:
                                #print(line_segments) #---------------
                                i_label = line_segments[0]
                                i_word = line_segments[3]
                                i_lemma = line_segments[4]
                                if i_label != "O" and "_+_" in i_label:
                                    i_annotation = i_label.replace("B-", "").replace("I-", "")
                                    i_sentence.append(i_word + "[" + i_annotation + "]")
                                    '''
                                    if i_annotation == "Kivunhoito_+_Suunnitelma":
                                        i_sentence_has_multi_annotation = True

                                    if i_annotation == "Kipu_+_Toistuva_tilanne":
                                        i_sentence_has_multi_annotation = True

                                    if i_annotation == "Kipu_+_Sijainti":
                                        i_sentence_has_multi_annotation = True

                                    if i_annotation == "Kivunhoito_+_Suunnitelma(Ehdollinen:True)":
                                        i_sentence_has_multi_annotation = True

                                    if i_annotation == "Suunnitelma_+_Toimenpide":
                                        i_sentence_has_multi_annotation = True

                                    if i_annotation == "Kipu_+_Kivunhoito":
                                        i_sentence_has_multi_annotation = True
                                    '''
                                    if i_annotation == "Implisiittinen_kipu_+_Laatu":
                                        i_sentence_has_multi_annotation = True

                                else:
                                    i_sentence.append(i_word)

############################################################################
#Main ...
if __name__ == "__main__":
    # Argument handling
    parser = argparse.ArgumentParser(description='Generate some statistics')
    parser.add_argument('-f', nargs='+') # CoNLL folder
    args = None
    if len(sys.argv) == 1:
        args = parser.parse_args('-f data/save/'.split()) # For debugging ---------
    else:
        args = parser.parse_args(sys.argv[1:])

    print("Arguments: " + str(args))
    print(" ... processing ... \n")
    stats = GetUniqueAnnotations(''.join(args.f))

    '''
    stats.get_unique_multi_annotations_words()
    print("\n\nSORTED BY FREQUENCY")
    stats.print_stats_sort_by_freq()
    print("\n\nSORTED BY NAME")
    stats.print_stats_sort_by_name()
    '''

    stats.get_unique_multi_annotations_sentences()