from __future__ import division

import argparse
import sys
from os import listdir
from collections import OrderedDict
import simplify_pain_annotations as simplify
import os.path
import random
import math



def divide_train_devel_test(ann_folder, train_percentage=60, t_only=False):

    # Get all annotations and what files, i.e. filenames, they occur in
    #annotations_occurring_in_files = {}

    files_and_their_annotations = {}

    #filename_set = set()
    for ann_filename in listdir(ann_folder):
        if ann_filename.endswith('.ann'):

            files_and_their_annotations[ann_filename[:-len('.ann')]] = []

            #filename_set.add(ann_filename[:-len('.ann')])
            ann_filename_fullpath = ann_folder + '/' + ann_filename
            ann_obj = AnnotationFile(ann_fullpath=ann_filename_fullpath, t_annotations_only=t_only)
            all_annotations = ann_obj.get_all_annotations_in_file()
            if len(all_annotations) > 0:
                for annotation_text in all_annotations.keys():

                    files_and_their_annotations[ann_filename[:-len('.ann')]].append(annotation_text)

                    #if annotation_text in annotations_occurring_in_files:
                    #    annotations_occurring_in_files[annotation_text].add(ann_filename[:-len('.ann')]) # Remove the .ann filetype ending
                    #else:
                    #    annotations_occurring_in_files[annotation_text] = set(ann_filename[:-len('.ann')]) # Remove the .ann filetype ending
            else:
                files_and_their_annotations[ann_filename[:-len('.ann')]].append(u'empty')
                #if u'empty' in annotations_occurring_in_files:
                #    annotations_occurring_in_files[u'empty'].add(ann_filename[:-len('.ann')]) # Remove the .ann filetype ending
                #else:
                #    annotations_occurring_in_files[u'empty'] = set(ann_filename[:-len('.ann')])  # Remove the .ann filetype ending


    # Conduct some sort of stratified division into train, devel and test sets, where each document is considered belonging to one annotation class
    annotations_and_their_files = {}
    for filename, annotation_list in files_and_their_annotations.items():
        selected_ann_class = random.choice(annotation_list)
        files_and_their_annotations[filename] = selected_ann_class

        if selected_ann_class in annotations_and_their_files:
            annotations_and_their_files[selected_ann_class].append(filename)
        else:
            annotations_and_their_files[selected_ann_class] = [filename]

    # Calculate the distribution of annotations for train, devel and test
    train_filenames = []
    devel_filenames = []
    test_filenames = []

    for annotation_text, ann_filename_list in annotations_and_their_files.items():
        file_count = len(ann_filename_list)
        if file_count == 1:
            train_size = 1
            test_size = 0
            devel_size = 0
        else:
            train_size = int(math.floor(file_count / 100 * train_percentage))
            test_size = int(math.ceil((file_count - train_size) / 2))
            devel_size = file_count - (train_size + test_size)
        train_filenames.extend(annotations_and_their_files[annotation_text][0:train_size])
        test_filenames.extend(annotations_and_their_files[annotation_text][train_size:(train_size + test_size)])
        devel_filenames.extend(annotations_and_their_files[annotation_text][(train_size + test_size):])
        #print(annotation_text, 'train_size', train_size, 'devel_size', devel_size, 'test_size', test_size) #------

    # Randomize the filenames
    random.shuffle(train_filenames)
    random.shuffle(devel_filenames)
    random.shuffle(test_filenames)
    return (train_filenames, devel_filenames, test_filenames)


def get_conllu_rows_with_offsets(txt_filename, conllu_filename):
    new_conllu_rows = []

    conllu_file = open(conllu_filename, 'rb')
    with open(txt_filename, 'rb') as txt_file:
        conllu_line = conllu_file.next().decode('utf-8').strip()
        conllu_line_parts = conllu_line.split('\t')
        word_num_in_sent = conllu_line_parts[0]
        conllu_word = conllu_line_parts[1]
        lem_word = conllu_line_parts[2]
        pos_tag = conllu_line_parts[3]
        rest_tags = conllu_line_parts[4:] # <-- any use?

        current_doc_offset = 0
        #text_word = ''
        word_start_offset = 0
        for line in txt_file:
            text_word = ''
            line = line.decode('utf-8')
            #print('\nLINE: ' + line)  # --------
            #print('{{conllu_word}} ' + conllu_word) #----
            for char in line:
                if not char.isspace():
                    text_word += char
                    #print('[[TEXT]] ' + text_word) #-----
                    current_doc_offset += 1
                    word_end_offset = current_doc_offset

                    if text_word == conllu_word:
                        #print('>>> ' + str(word_start_offset) + ', ' + str(word_end_offset) + ', ' + conllu_word + ', ' + lem_word + ', ' + pos_tag)  # ---------
                        new_conllu_rows.append([word_start_offset, word_end_offset, conllu_word, lem_word, pos_tag] + rest_tags)
                        text_word = ''
                        word_start_offset = current_doc_offset
                        conllu_line = conllu_file.next().decode('utf-8').strip()

                        if len(conllu_line) == 0:
                            try:
                                conllu_line = conllu_file.next().decode('utf-8').strip()
                            except:
                                conllu_file.close()
                                new_conllu_rows.append('NEWDOC')
                                return new_conllu_rows
                        conllu_line_parts = conllu_line.split('\t')
                        word_num_in_sent = conllu_line_parts[0]
                        conllu_word = conllu_line_parts[1]
                        lem_word = conllu_line_parts[2]
                        pos_tag = conllu_line_parts[3]
                        rest_tags = conllu_line_parts[4:] # <-- any use?
                else:
                    current_doc_offset += 1
                    word_start_offset = current_doc_offset
            #print('NEWLINE')  # --------------
            new_conllu_rows.append('NEWLINE')
        new_conllu_rows.append('NEWDOC')
    conllu_file.close()
    return new_conllu_rows



def add_and_save_annotations(ann_filename, new_conllu_rows, nersuite_save_filename, append_save=False, t_annotations_only=False): # nn_save_filename
    '''
    Simple format:
    --------------
    Transcripts O
    Upregulated O
    in  O
    ST  B-Cell
    -   I-Cell
    HSC I-Cell
    Compared    O
    to  O
    LT  B-Cell
    -   I-Cell
    HSC I-Cell


    Full format:
    --------------
    O	60	62	by	by  IN	B-PP
    B-GGP	63	74	interleukin	interleukin	NN	B-NP
    I-GGP	74	75	-	-	HYPH	B-NP
    I-GGP	75	80	1beta	1beta	NN	I-NP
    O	81	89	requires	require	VBZ	B-VP
    '''

    #nn_save_file = None
    nersuite_save_file = None

    if append_save:
        if not os.path.isfile(nersuite_save_filename):
            #nn_save_file = open(nn_save_filename, 'wb')
            nersuite_save_file = open(nersuite_save_filename, 'wb')
        else:
            #nn_save_file = open(nn_save_filename, 'ab')
            #if not os.stat(nn_save_filename).st_size == 0:
            #    nn_save_file.write('\n')
            nersuite_save_file = open(nersuite_save_filename, 'ab')
            if not os.stat(nersuite_save_filename).st_size == 0:
                nersuite_save_file.write('\n')
    else:
        #nn_save_file = open(nn_save_filename, 'wb')
        nersuite_save_file = open(nersuite_save_filename, 'wb')

    ann_obj = AnnotationFile(ann_fullpath=ann_filename, t_annotations_only=bool(t_annotations_only))

    if ann_obj.file_has_annotations():
        ongoing_annotation_rows = []
        ongoing_annotation = False
        for row in new_conllu_rows:
            if type(row) is list:
                i_annotations = ann_obj.get_word_annotations(row[0], row[1])
                if len(i_annotations) > 0:  # Has one or more annotations

                    #-------------------------
                    #- SECOND VERSION --------
                    #-------------------------
                    ann_has_B = False
                    ann_has_I = False
                    for k_ann in i_annotations:
                        if k_ann.startswith('B-'):
                            ann_has_B = True
                        elif k_ann.startswith('I-'):
                            ann_has_I = True

                    if ann_has_I or (ann_has_I and ann_has_B):
                        i_ann_set = set()
                        for ann in i_annotations:
                            i_ann_set.add(ann[2:])
                        ongoing_annotation_rows.append((i_ann_set, str(row[0]) + '\t' + str(row[1]) + '\t' + row[2] + '\t' + row[3] + '\t' + row[4] + '\t' + 'O'))

                    if ann_has_B and not ann_has_I: # New annotation sequence
                        # ===========================================
                        # Write the previous annotation to file if this have not been done yet
                        if len(ongoing_annotation_rows) > 0:
                            combined_ann_set = set()
                            for (i_ann_set, i_str_row) in ongoing_annotation_rows:
                                combined_ann_set = combined_ann_set.union(i_ann_set)
                            combined_ann_text = '-AND-'.join(sorted(combined_ann_set))
                            i_prefix = 'B-'
                            for (i_ann_set, i_str_row) in ongoing_annotation_rows:
                                nersuite_save_file.write((i_prefix + combined_ann_text + '\t' + i_str_row + "\n").encode('utf-8'))
                                i_prefix = 'I-'
                            ongoing_annotation_rows = []
                        # ===========================================

                        # Start new sequence
                        i_ann_set = set()
                        for ann in i_annotations:
                            i_ann_set.add(ann[2:])
                        ongoing_annotation_rows.append((i_ann_set, str(row[0]) + '\t' + str(row[1]) + '\t' + row[2] + '\t' + row[3] + '\t' + row[4] + '\t' + 'O'))

                    """
                    #-------------------------
                    #- FIRST VERSION ---------
                    #-------------------------
                    i_annotations.sort(key=lambda x: x[2:])  # <-- NOTE: sort to create some consistency in the joint labels! Sorting from third char since the first ones should be "B-" or "I-".
                    i_annotations_texts = '-AND-'.join(str(a) for a in i_annotations)
                    # Keep only the B- or I- at the start, remove the rest
                    #i_annotations_texts = i_annotations_texts[:2] + i_annotations_texts[2:].replace('B-', '').replace('I-', '')
                    prefix = ''
                    if ('B-' in i_annotations_texts) or ('B-' in i_annotations_texts and 'I-' in i_annotations_texts):
                        prefix = 'B-'
                    elif ('I-' in i_annotations_texts):
                        prefix = 'I-'
                    i_annotations_texts = prefix + i_annotations_texts[2:].replace('B-', '').replace('I-', '')
                    nersuite_save_file.write((i_annotations_texts + '\t' + str(row[0]) + '\t' + str(row[1]) + '\t' + row[2] + '\t' + row[3] + '\t' + row[4] + '\t' + 'O' + "\n").encode('utf-8'))
                    #-------------------------
                    """
                else:
                    # ===========================================
                    # Write the previous annotation to file if this have not been done yet
                    if len(ongoing_annotation_rows) > 0:
                        combined_ann_set = set()
                        for (i_ann_set, i_str_row) in ongoing_annotation_rows:
                            combined_ann_set = combined_ann_set.union(i_ann_set)
                        combined_ann_text = '-AND-'.join(sorted(combined_ann_set))
                        i_prefix = 'B-'
                        for (i_ann_set, i_str_row) in ongoing_annotation_rows:
                            nersuite_save_file.write(
                                (i_prefix + combined_ann_text + '\t' + i_str_row + "\n").encode('utf-8'))
                            i_prefix = 'I-'
                        ongoing_annotation_rows = []
                    # ===========================================
                    nersuite_save_file.write(('O' + '\t' + str(row[0]) + '\t' + str(row[1]) + '\t' + row[2] + '\t' + row[3] + '\t' + row[4] + '\t' + 'O' + "\n").encode('utf-8'))
            else:
                # ===========================================
                # Write the previous annotation to file if this have not been done yet
                if len(ongoing_annotation_rows) > 0:
                    combined_ann_set = set()
                    for (i_ann_set, i_str_row) in ongoing_annotation_rows:
                        combined_ann_set = combined_ann_set.union(i_ann_set)
                    combined_ann_text = '-AND-'.join(sorted(combined_ann_set))
                    i_prefix = 'B-'
                    for (i_ann_set, i_str_row) in ongoing_annotation_rows:
                        nersuite_save_file.write(
                            (i_prefix + combined_ann_text + '\t' + i_str_row + "\n").encode('utf-8'))
                        i_prefix = 'I-'
                    ongoing_annotation_rows = []
                # ===========================================
                nersuite_save_file.write(('\n').encode('utf-8'))
    else:
        # No annotations to expect, fast version ...
        for row in new_conllu_rows:
            if type(row) is list:
                #save_file.write((str(row[0]) + '\t' + str(row[1]) + '\t' + row[2] + '\t' + row[3] + '\t' + row[4] + '\n').encode('utf-8'))
                #nn_save_file.write((row[2] + '\t' + 'O' + '\n').encode('utf-8'))
                nersuite_save_file.write(('O' + '\t' + str(row[0]) + '\t' + str(row[1]) + '\t' + row[2] + '\t' + row[3] + '\t' + row[4] + '\t' + 'O' + "\n").encode('utf-8'))
            else:
                #nn_save_file.write(('\n').encode('utf-8'))
                nersuite_save_file.write(('\n').encode('utf-8'))
    #nn_save_file.close()
    nersuite_save_file.close()


#########################################
# Helper class for handling Brat ann files
class AnnotationFile:
    def __init__(self, ann_fullpath, t_annotations_only=False):
        self._ann_fullpath = ann_fullpath
        #self._annotation_entries = {}
        #############
        # Read the different tag types into separate lists
        self._T_types = {}
        self._E_types = {}
        self._A_types = {}
        self._M_types = {}
        self._N_types = {}
        self._Note_types = {}
        self._Equiv_types = {}
        self.t_annotations_only = t_annotations_only
        self.load_ann_file(self._ann_fullpath)

    def load_ann_file(self, ann_fullpath):
        #if stat(ann_fullpath).st_size != 0:
        #print("\nFILE: " + ann_fullpath) #---------
        with open(ann_fullpath, 'rb') as f_ann:
            for i_line in f_ann:
                i_line = i_line.decode('utf-8').strip()
                #print("\t" + line) #--------
                i_tab_sep = i_line.split("\t")
                i_id = i_tab_sep[0]

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
                elif i_id.startswith("E") and not self.t_annotations_only:
                    print("E-type ... ")
                    #self._E_types[i_id] = i_arg_string
                elif i_id.startswith("A") and not self.t_annotations_only:
                    #print("A-type ... ")
                    self._A_types[i_id] = A_AnnotationEntry(i_id, i_tab_sep[1].strip())
                elif i_id.startswith("N") and not self.t_annotations_only:
                    print("N-type ... ")
                    #self._N_types[i_id] = i_arg_string
                elif i_id.startswith("#") and not self.t_annotations_only:
                    print("#-type ... ")
                    #self._Note_types[i_id] = i_arg_string
                elif i_tab_sep[1].startswith("Equiv") and not self.t_annotations_only:
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

    def file_has_annotations(self):
        if (self._T_types): # or self._E_types or self._A_types or self._M_types or self._N_types or self._Note_types or self._Equiv_types):
            return True
        return False

    def get_all_annotations_in_file(self):
        annotaions_dict = {}
        for i_id, i_annotation_entry in self._T_types.items():
            annotation = i_annotation_entry.get_annotation_string()
            if annotation in annotaions_dict:
                annotaions_dict[annotation] += 1
            else:
                annotaions_dict[annotation] = 1
        return annotaions_dict

#========================================
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
#########################################



if __name__ == "__main__":
    ####################################%%%%%%%%%%%%%%%%%%%%%
    parser = argparse.ArgumentParser(description='combine_text_conllu_ann.py')
    parser.add_argument('-text', type=str, help='Folder with the original text documents (.txt).', default='data/text-ann-conllu')  # required=True)
    parser.add_argument('-ann', type=str, help='Folder with the ann files (.ann).', default='data/text-ann-conllu') #required=True)
    parser.add_argument('-conllu', type=str, help='Folder with the conllu files from the parser (.txt.conllu).', default='data/text-ann-conllu') #required=True)
    #parser.add_argument('-nn_save', type=str, help='Folder to save the combined files (.conll) in the Neural Network training format, or give a filename to save as one combined file.', default='data/combined/nn-one-file.conll') #required=True)
    parser.add_argument('-nersuite_save', type=str, help='Folder to save the combined files (.conll) in the NERsuite format, or give a filename to save as one combined file.', default='data/train-with-nersuite/nersuite.conll')  # required=True)
    parser.add_argument('-t_only', type=int, help='Use only the T-type annotations; default = 0 (false).', default=0)
    parser.add_argument('-train_percentage', type=int, help='How much of the data to use for training, the remaining will be divided equally between the devel and test sets; default = 60.', default=60)
    ####################################%%%%%%%%%%%%%%%%%%%%%
    args = parser.parse_args(sys.argv[1:])

    print("\nStart ... ")
    # Divide into train, devel and test
    (train_filenames, devel_filenames, test_filenames) = divide_train_devel_test(ann_folder=args.ann, train_percentage=args.train_percentage, t_only=args.t_only)
    #print('train_filenames:', train_filenames, 'train_filenames:', devel_filenames, 'train_filenames:', test_filenames) #-----
    print('Train files: ' + str(len(train_filenames)) + ', Devel files: ' + str(len(devel_filenames)) + ', Test files: ' + str(len(test_filenames)))

    if os.path.isdir(args.nersuite_save):
        # Save as individual files
        for fn_no_ending in train_filenames:
            fn_text = args.text + '/' + fn_no_ending + '.txt'
            fn_conllu = args.conllu + '/' + fn_no_ending + '.txt.conllu'
            fn_ann = args.ann + '/' + fn_no_ending + '.ann'
            i_new_connlu_rows = get_conllu_rows_with_offsets(txt_filename=fn_text, conllu_filename=fn_conllu)
            add_and_save_annotations(ann_filename=fn_ann, new_conllu_rows=i_new_connlu_rows, nersuite_save_filename=args.nersuite_save + '/train-' + fn_no_ending + '.nersuite.conll', t_annotations_only=args.t_only)

        for fn_no_ending in devel_filenames:
            fn_text = args.text + '/' + fn_no_ending + '.txt'
            fn_conllu = args.conllu + '/' + fn_no_ending + '.txt.conllu'
            fn_ann = args.ann + '/' + fn_no_ending + '.ann'
            i_new_connlu_rows = get_conllu_rows_with_offsets(txt_filename=fn_text, conllu_filename=fn_conllu)
            add_and_save_annotations(ann_filename=fn_ann, new_conllu_rows=i_new_connlu_rows, nersuite_save_filename=args.nersuite_save + '/devel-' + fn_no_ending + '.nersuite.conll', t_annotations_only=args.t_only)

        for fn_no_ending in test_filenames:
            fn_text = args.text + '/' + fn_no_ending + '.txt'
            fn_conllu = args.conllu + '/' + fn_no_ending + '.txt.conllu'
            fn_ann = args.ann + '/' + fn_no_ending + '.ann'
            i_new_connlu_rows = get_conllu_rows_with_offsets(txt_filename=fn_text, conllu_filename=fn_conllu)
            add_and_save_annotations(ann_filename=fn_ann, new_conllu_rows=i_new_connlu_rows, nersuite_save_filename=args.nersuite_save + '/test-' + fn_no_ending + '.nersuite.conll', t_annotations_only=args.t_only)

        """
        # Save as separate files
        for filename in listdir(args.text):
            fn_text = args.text + '/' + filename
            if filename.lower().endswith('.txt'):
                fn_no_ending = filename[:-len('.txt')]
                fn_conllu = args.conllu + '/' + fn_no_ending + '.txt.conllu'
                fn_ann = args.ann + '/' + fn_no_ending + '.ann'
                i_new_connlu_rows = get_conllu_rows_with_offsets(txt_filename=fn_text, conllu_filename=fn_conllu)
                add_and_save_annotations(ann_filename=fn_ann, new_conllu_rows=i_new_connlu_rows, nersuite_save_filename=args.nersuite_save + '/' + fn_no_ending + '.nersuite.conll', t_annotations_only=args.t_only)
        """

    elif not os.path.isdir(args.nersuite_save):
        # Save as one file
        #nn_folder, nn_filename = os.path.split(args.nn_save)
        nersuite_folder, nersuite_filename = os.path.split(args.nersuite_save)

        #nn_train_filepath = (nn_folder + '/' if len(nn_folder) > 0 else '') + 'train-' + nn_filename
        nersuite_train_filepath = (nersuite_folder + '/' if len(nersuite_folder) > 0 else '') + 'train-' + nersuite_filename
        #open(nn_train_filepath, 'w').close() # Clear content of any existing file
        open(nersuite_train_filepath, 'w').close()  # Clear content of any existing file
        for fn_no_ending in train_filenames:
            #print('TRAIN:', fn_no_ending) #--------
            fn_text = args.text + '/' + fn_no_ending + '.txt'
            fn_conllu = args.conllu + '/' + fn_no_ending + '.txt.conllu'
            fn_ann = args.ann + '/' + fn_no_ending + '.ann'
            i_new_connlu_rows = get_conllu_rows_with_offsets(txt_filename=fn_text, conllu_filename=fn_conllu)
            add_and_save_annotations(ann_filename=fn_ann, new_conllu_rows=i_new_connlu_rows, nersuite_save_filename=nersuite_train_filepath, append_save=True, t_annotations_only=args.t_only)

        #nn_devel_filepath = (nn_folder + '/' if len(nn_folder) > 0 else '') + 'devel-' + nn_filename
        nersuite_devel_filepath = (nersuite_folder + '/' if len(nersuite_folder) > 0 else '') + 'devel-' + nersuite_filename
        #open(nn_devel_filepath, 'w').close() # Clear content of any existing file
        open(nersuite_devel_filepath, 'w').close()  # Clear content of any existing file
        for fn_no_ending in devel_filenames:
            #print('DEVEL:', fn_no_ending)  # --------
            fn_text = args.text + '/' + fn_no_ending + '.txt'
            fn_conllu = args.conllu + '/' + fn_no_ending + '.txt.conllu'
            fn_ann = args.ann + '/' + fn_no_ending + '.ann'
            i_new_connlu_rows = get_conllu_rows_with_offsets(txt_filename=fn_text, conllu_filename=fn_conllu)
            add_and_save_annotations(ann_filename=fn_ann, new_conllu_rows=i_new_connlu_rows, nersuite_save_filename=nersuite_devel_filepath, append_save=True, t_annotations_only=args.t_only)

        #nn_test_filepath = (nn_folder + '/' if len(nn_folder) > 0 else '') + 'test-' + nn_filename
        nersuite_test_filepath = (nersuite_folder + '/' if len(nersuite_folder) > 0 else '') + 'test-' + nersuite_filename
        #open(nn_test_filepath, 'w').close() # Clear content of any existing file
        open(nersuite_test_filepath, 'w').close()  # Clear content of any existing file
        for fn_no_ending in test_filenames:
            #print('TEST:', fn_no_ending)  # --------
            fn_text = args.text + '/' + fn_no_ending + '.txt'
            fn_conllu = args.conllu + '/' + fn_no_ending + '.txt.conllu'
            fn_ann = args.ann + '/' + fn_no_ending + '.ann'
            i_new_connlu_rows = get_conllu_rows_with_offsets(txt_filename=fn_text, conllu_filename=fn_conllu)
            add_and_save_annotations(ann_filename=fn_ann, new_conllu_rows=i_new_connlu_rows, nersuite_save_filename=nersuite_test_filepath, append_save=True, t_annotations_only=args.t_only)

        """
        open(args.nn_save, 'w').close() # Clear content of any existing file
        open(args.nersuite_save, 'w').close()  # Clear content of any existing file
        for filename in listdir(args.text):
            fn_text = args.text + '/' + filename
            if filename.lower().endswith('.txt'):
                fn_no_ending = filename[:-len('.txt')]
                fn_conllu = args.conllu + '/' + fn_no_ending + '.txt.conllu'
                fn_ann = args.ann + '/' + fn_no_ending + '.ann'
                i_new_connlu_rows = get_conllu_rows_with_offsets(txt_filename=fn_text, conllu_filename=fn_conllu)
                add_and_save_annotations(ann_filename=fn_ann, new_conllu_rows=i_new_connlu_rows, nersuite_save_filename=args.nersuite_save, append_save=True, t_annotations_only=args.t_only)
        """
    else:
        print('-nn_save and -nersuite_save does both need to be either a filename or both need to be a folder name.')

    print("\nDone!")