import argparse
import mmap
import sys
from collections import OrderedDict
from os import listdir
from os import stat
from os.path import isfile
from os.path import splitext

from CRF import evaluate_common_pain as simplify

# Data at ikitik:
# ANN:          /data/finished-annotation/reference-corpus/episodes-kipu-000-006-008-014-consensus-revised-by-kemheik/consensus/  + SUBFOLDERS!
# CoNLL/fincgs: /data/finished-annotation/reference-corpus/episodes-fincgs

# Code location at ikitik: /data/ikitik-processed-data/annotation/code/
# nersuite files at ikitik: /data/ikitik-processed-data/annotation/nersuite-files (/kipu etc.)

# Combined generated pain data: /data/ikitik-processed-data/annotation/nersuite-files/kipu/

###########################
# Some static parameteres #
###########################
CONLL_FILE_EXTENSION = "fincg"
ANN_FILE_EXTENSION = "ann"
SAVE_FILE_EXTENSION = "nersuite"

CONLL_SPLIT_DELIMITER = None
#ANN_SPLIT_DELIMITER = None
SAVE_SPLIT_DELIMITER = "\t"

MIN_ROW_ELEMENTS = 6 # The named entities/annotations comes in addition
###########################

#-------------------------#
# Statistics
#-------------------------#
discontinued_annotation_count = 0

#-------------------------#


# Check file. 0 = no file found, 1 = file found, 2 = file is not empty, 3 = file contains the given find_string.
def check_file(file_fullpath, find_string):
    return_value = 0
    if isfile(file_fullpath):
        return_value = 1

        if stat(file_fullpath).st_size != 0:
            return_value = 2
            f = open(file_fullpath)
            s = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

            if find_string is not None and find_string != "":
                if s.find(find_string) != -1:
                    return_value = 3
    return return_value


# Class for combining annotations in Brat ANN files with rows in CoNLL style files.
class Combiner:

    def __init__(self, conll_folder, ann_folder, save_folder, process_restrictions):
        self._conll_folder = conll_folder
        self._ann_folder = ann_folder
        self._save_folder = save_folder

        self._process_restrictions = process_restrictions # True or False

        self._conll_filenames_fullpath = {}
        self._ann_filenames_and_fullpath = {}

        self._conll_ann_dictionary = {}

    # Read the CoNLL files and ANN files into dictionaries containing the filenames and their full paths.
    def load_filenames(self):
        # CoNLL files:
        for filename in listdir(self._conll_folder):
            full_file_path = self._conll_folder + "/" + filename
            #print(filename) #-------------
            if isfile(full_file_path) and filename.lower().endswith("." + CONLL_FILE_EXTENSION):
                self._conll_filenames_fullpath[splitext(filename)[0]] = full_file_path
                #print("CoNLL:::: " + splitext(filename)[0] + " -- " + full_file_path) #-----------

        # ANN files:
        for filename in listdir(self._ann_folder):
            full_file_path = self._ann_folder + "/" + filename
            #print(filename) #-------------
            if isfile(full_file_path) and filename.lower().endswith("." + ANN_FILE_EXTENSION):
                self._ann_filenames_and_fullpath[splitext(filename)[0]] = full_file_path
                #print("ANN:::: " + splitext(filename)[0] + " -- " + full_file_path) #-----------

    def combine_file_pairs(self):
        files_processed_count = 0

        # Go through each CoNLL file
        for i_conll_filename, i_conll_and_fullpath in self._conll_filenames_fullpath.items():

            # Find its corresponding ANN file
            i_annotation_file = None
            if (i_conll_filename in self._ann_filenames_and_fullpath) and (check_file(self._ann_filenames_and_fullpath[i_conll_filename], "T1") >= self._process_restrictions):
                # This file has been annotated! Create a new file where the annotations are alligned with the corresponding tokens.
                i_annotation_file = AnnotationFile(self._ann_filenames_and_fullpath[i_conll_filename])
            else:
                if self._process_restrictions: # We only want to process and store the file that has been annotated
                    continue

            f_save = open(self._save_folder + "/" + i_conll_filename + "." + SAVE_FILE_EXTENSION, 'wt')

            # Go through each CoNLL file line by line
            with open(i_conll_and_fullpath, 'rt') as f_conll:
                for line in f_conll:
                    #line = line.decode('utf-8')
                    line_segments = line.split(CONLL_SPLIT_DELIMITER)
                    #print(line_segments) #---------------

                    if len(line_segments) >= 4:
                        i_word_offset_start = int(line_segments[0])
                        i_word_offset_end = int(line_segments[1])
                        i_word_text = line_segments[2]

                        # Make sure the number of elements per row are of a fixed size
                        if len(line_segments) > MIN_ROW_ELEMENTS:
                            ''' # OLD:
                            # Remove any elements above MIN_ROW_ELEMENTS
                            line_segments = line_segments[0:MIN_ROW_ELEMENTS]
                            '''
                            # Combine the remaining elements into one string
                            last_ele = '-'.join(line_segments[MIN_ROW_ELEMENTS-1:len(line_segments)])
                            line_segments = line_segments[0:MIN_ROW_ELEMENTS-1]
                            line_segments.append(last_ele)

                        elif len(line_segments) < MIN_ROW_ELEMENTS:
                            line_segments += ['O'] * (MIN_ROW_ELEMENTS - len(line_segments)) #<--- Padding of O's up to MIN_ROW_ELEMENTS

                        # Create a string out of the list
                        i_line_combined = SAVE_SPLIT_DELIMITER.join(str(s) for s in line_segments)

                        #print(line_segments[2] + "\t" + str(i_word_offset_start) + " - " + str(i_word_offset_end)) #----------------

                        if i_annotation_file is not None:
                            #print("TARGET WORD = " + i_word_text ) #-------------
                            i_annotations = i_annotation_file.get_word_annotations(i_word_offset_start, i_word_offset_end)

                            if len(i_annotations) > 0: # Has one or more annotations
                                i_annotations.sort(key=lambda x: x[2:]) # <-- NOTE: sort to create some consistency in the joint labels! Sorting from third char since the first ones should be "B-" or "I-".
                                i_annotations_texts = '_+_'.join(str(a) for a in i_annotations)
                                '''
                                #### NB! #####################
                                if len(i_annotations) > 1:
                                    print("\n>-> NB! in file '" + i_conll_filename + "', line: '" + i_line_combined + "' has multiple annotations: " + i_annotations_texts)
                                ##############################
                                '''
                                #print("\t" + i_word_text + " should have annotation(s): " + ' and '.join(str(c) for c in i_annotations)) #------------
                                f_save.write(i_annotations_texts + "\t" + i_line_combined + "\n")
                                continue
                        # No annotations for this line (in the CoNLL file)
                        f_save.write("O\t" + i_line_combined + "\n")
                    else:
                        f_save.write(line)

            f_save.close()
            files_processed_count += 1
        return files_processed_count



# Helper class for Combiner
class AnnotationFile:
    def __init__(self, ann_fullpath):
        self._ann_fullpath = ann_fullpath
        self._annotation_entries = {}
        #############
        # Read the different tag types into separate lists
        self._T_types = {}
        self._E_types = {}
        self._A_types = {}
        self._M_types = {}
        self._N_types = {}
        self._Note_types = {}
        self._Equiv_types = {}

        self.load_ann_file(self._ann_fullpath)

    def has_content(self):
        return self.has_content

    def load_ann_file(self, ann_fullpath):
        #if stat(ann_fullpath).st_size != 0:
        #print("\nFILE: " + ann_fullpath) #---------
        with open(ann_fullpath, 'rt') as f_ann:
            for i_line in f_ann:
                #print("\t" + line) #--------
                i_tab_sep = i_line.split("\t")
                i_id = i_tab_sep[0].strip()

                # T: text-bound annotation
                # R: relation
                # E: event
                # A: attribute
                # M: modification (alias for attribute, for backward compatibility)
                # N: normalization [new in v1.3]
                # #: note
                # Equiv_types: has 'Equiv' as second argument

                if i_id.startswith("T"):
                    self._T_types[i_id] = T_AnnotationEntry(i_id, i_tab_sep[1].strip(), i_tab_sep[2].strip())
                elif i_id.startswith("E"):
                    print("E-type ... ")
                    #self._E_types[i_id] = i_arg_string
                elif i_id.startswith("A"):
                    #print("A-type ... ")
                    self._A_types[i_id] = A_AnnotationEntry(i_id, i_tab_sep[1].strip())
                elif i_id.startswith("N"):
                    print("N-type ... ")
                    #self._N_types[i_id] = i_arg_string
                elif i_id.startswith("#"):
                    print("#-type ... ")
                    #self._Note_types[i_id] = i_arg_string
                elif i_tab_sep[1].startswith("Equiv"):
                    print("Equiv-type ... ")
                    #self._Equiv_types[i_id] = i_arg_string

        # Get all modifying annotations that affects the T-annotations
        for i_A_type in self._A_types.values():
            for i_T_type_id in i_A_type.get_target_annotation_ids().intersection(self._T_types):
                self._T_types[i_T_type_id].add_ref_annotation(i_A_type)

    def get_word_annotations(self, word_start_offset, word_end_offset):
        annotations = []
        for i_id, i_annotation_entry in self._T_types.items():
            i_to_annotate = i_annotation_entry.check_get_annotation(word_start_offset, word_end_offset)
            if i_to_annotate is not None:
                annotations.append(i_to_annotate + "-" + i_annotation_entry.get_annotation_string() + i_annotation_entry.get_ref_annotations_text())
                #if i_annotation_entry.get_ref_annotations_text() != "": #---------
                #   print("%%%%" + i_annotation_entry.get_annotation_string() + i_annotation_entry.get_ref_annotations_text()) #---------
        return annotations

    def has_annotations(self):
        return len(self._annotation_entries) > 0



#########################################
# Helper classes for AnnotationFile
# Brat info: http://brat.nlplab.org/standoff.html
# T: text-bound annotation
class T_AnnotationEntry:
    # T1	Implisiittinen_kipu 79 85	Angina
    def __init__(self, id, arg_string, target_string):
        self._id = id
        self._annotation = "None"
        self._offset_pairs = OrderedDict() # <-- !
        self._first_word_annotation = True

        # Add B- and I- tags!
        self._beginning_annotation = "B"
        self._intermediate_annotation = "I"
        self._last_match_i = -1

        split_discontinuous_offsets = arg_string.split(";")
        '''
        #-------------
        if len(split_discontinuous_offsets) > 1:
            global discontinued_annotation_count
            discontinued_annotation_count += 1
        #-------------
        '''
        i = 0
        for i_substring in split_discontinuous_offsets:
            i_ele = i_substring.split(" ")
            if i == 0:
                self._annotation = simplify.simplify_class(i_ele[0]) # <-- Using Juho's simplify function.
                self._offset_pairs[int(i_ele[1])] = int(i_ele[2])
            else:
                self._offset_pairs[int(i_ele[0])] = int(i_ele[1])
            i += 1
        self._annotated_term = target_string
        self._ref_annotations = []

    def check_get_annotation(self, word_start_offset, word_end_offset):
        #print("entry_start_offset=" + str(self._start_offset) + ", entry_end_offset=" + str(self._end_offset) + ";; word_start_offset=" + str(word_start_offset) + ", word_end_offset=" + str(word_end_offset)) #------------
        has_annotation = False

        i = 0
        for i_start_offset, i_end_offset in self._offset_pairs.items():
            if (i_start_offset <= word_start_offset) and (i_end_offset >= word_end_offset):
                #return True
                has_annotation = True
            elif (i_start_offset >= word_start_offset) and (i_start_offset < word_end_offset):
                #return True
                has_annotation = True
            elif (i_end_offset > word_start_offset) and (i_end_offset <= word_end_offset):
                #return True
                has_annotation = True

            if has_annotation:

                # Include the following code to reset anntation span when it has a discontinued offset. -->
                if i > self._last_match_i:
                    self._first_word_annotation = True # <-- Enable if offset spans
                    self._last_match_i = i
                # <--

                if self._first_word_annotation:
                    annotation_prefix = self._beginning_annotation
                    self._first_word_annotation = False
                else:
                    annotation_prefix = self._intermediate_annotation
                return annotation_prefix
            i += 1

        return None

    def get_annotation_string(self):
        return self._annotation

    def get_id(self):
        return self._id

    def get_annotated_term(self):
        return self._annotated_term

    def add_ref_annotation(self, ref_annotation):
        self._ref_annotations.append(ref_annotation)

    def get_ref_annotations_text(self):
        if len(self._ref_annotations) > 0:
            return "(" + ','.join(r.get_text() for r in self._ref_annotations) + ")"
        else:
            return ""

# R: relation
class R_AnnotationEntry:
    # R1	Origin Arg1:T3 Arg2:T4
    def __init__(self, id, arg_string):
        print(id)

# E: event
class E_AnnotationEntry:
    # T2	MERGE-ORG 14 27	joint venture
    # E1	MERGE-ORG:T2 Org1:T1 Org2:T3
    def __init__(self, id, arg_string):
        print(id)

# A: attribute + M: modification (alias for attribute, for backward compatibility)
class A_AnnotationEntry:
    # A1	Negation E1
    # A2	Confidence E2 L1
    def __init__(self, id, arg_string):
        self._id = id

        split_args = arg_string.split(" ")
        self._modifier = split_args[0]

        self._target_annotation_ids = set()
        self._target_annotation_ids.add(split_args[1])

        self._value = None
        if len(split_args) > 2:
            self._value = '&'.join(v for v in split_args[2:])

    def get_modifier(self):
        return self._modifier

    def get_target_annotation_ids(self):
        return self._target_annotation_ids

    def get_modifier_value(self):
        return self._value

    def get_text(self):
        if self._value is not None:
            return self.get_modifier() + ":" + self._value
        else:
            return self.get_modifier() + ":True"

# N: normalization [new in v1.3]
class N_AnnotationEntry:
    # N1	Reference T1 Wikipedia:534366	Barack Obama
    def __init__(self, id, arg_string):
        print(id)

# #: note
class Note_AnnotationEntry:
    # #1	AnnotatorNotes T1	this annotation is suspect
    def __init__(self, id, arg_string):
        print(id)

# Equiv_types: has 'Equiv' as second argument
class Equiv_AnnotationEntry:
    # T1	Organization 0 43	International Business Machines Corporation
    # T2	Organization 45 48	IBM
    # T3	Organization 52 60	Big Blue
    # *	Equiv T1 T2 T3
    def __init__(self, id, arg_string):
        print(id)



############################################################################
#Main ...
if __name__ == "__main__":

    # Argument handling
    parser = argparse.ArgumentParser(description='Combine CoNLL and ANN files arguments')
    parser.add_argument('-conll', nargs='+') # CoNLL folder
    parser.add_argument('-ann', nargs='+') # ANN folder
    parser.add_argument('-s', nargs='+') # Save filename
    parser.add_argument('-incl', nargs='?')
    args = None
    if len(sys.argv) == 1:
        args = parser.parse_args('-conll data/conll_with_linebreaks -ann data/ann -s data/save'.split()) # For debugging ---------
    else:
        args = parser.parse_args(sys.argv[1:])

    # Default inclusion criteria. 0 = ignore if ANN file exists, 1 = ANN file has to exist, 2 = ANN file must not be empty, 3 = ANN file must contain the string 'T1'
    args_incl = 1
    if args.incl is not None:
        args_incl = (''.join(str(incl) for incl in args.incl))
        try:
            args_incl = int(args_incl)
        except Exception:
            print("ERROR: Argument for -incl should be an integer between 0 and 3")

    print("Arguments: " + str(args))
    print(" ... combining ... ")
    comb = Combiner(''.join(str(conll) for conll in args.conll), ''.join(str(ann) for ann in args.ann), ''.join(str(save) for save in args.s), args_incl)
    comb.load_filenames()
    files_processed = comb.combine_file_pairs()

    print("Done!\t" + str(files_processed) + " file(s) processed.")
    #print("Discontinued annotations = " + str(discontinued_annotation_count)) #----------