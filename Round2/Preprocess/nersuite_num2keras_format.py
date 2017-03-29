"""
Code for converting annotated text in the NERSuite friendly format to a format used for sentence classification with Keras.

"""

from __future__ import division

import sys
import argparse
from os import listdir



def nersuite2keras(read_nersuite_folder, save_keras_sent_folder, save_keras_doc_folder, file_type='conll.num'):
    """
    NERSuite format:
    ----------------
    O	933	934	-	-	PUNCT	O
    O	935	939	ihon	iho	NOUN	O
    O	940	944	vari	vari	NOUN	O
    O	944	945	,	,	PUNCT	O
    O	946	951	lampo	lampo	NOUN	O
    O	952	954	ja	ja	CONJ	O
    O	955	966	turvotukset	turvotus	NOUN	O

    O	967	971	Kipu	kipu	NOUN	O

    O	972	979	Aamulla	aamu	NOUN	O
    O	980	988	yskiessa	yskia	VERB	O
    O	989	993	kova	kova	ADJ	O
    O	994	998	kipu	kipu	NOUN	O
    O	999	1007	haavalle	haapa	NOUN	O
    O	1007	1008	,	,	PUNCT	O
    B-Muu_oire	1009	1015	virtsa	virtsa	NOUN	O
    B-KNIS-AND-Muu_oire-AND-ROFL	1016	1020	meni	menna	VERB	O
    I-KNIS-AND-Muu_oire-AND-ROFL	1021	1025	alle	alle	ADV	O
    O	1026	1027	+	+	SYM	O
    B-Muu_poikkeava_tajunnan_taso	1028	1034	hetken	hetki	NOUN	O
    I-Muu_poikkeava_tajunnan_taso	1035	1041	silmat	silma	NOUN	O
    I-Muu_poikkeava_tajunnan_taso	1042	1053	tuijottavat	tuijottaa	VERB	O
    O	1053	1054	,	,	PUNCT	O
    B-Ei_herateltavissa	1055	1057	ei	ei	VERB	O
    I-Ei_herateltavissa	1058	1064	saanut	saada	VERB	O
    I-Ei_herateltavissa	1065	1074	kontaktia	kontakti	NOUN	O
    O	1074	1075	.	.	PUNCT	O

    Keras format:
    -------------
    word\tlemma\tpos\tword\tlemma\tpos\t|-|\tAnnotation1\tAnnotation2

    kipu\tkipu\tNOUN\thaavalle\thaapa\tNOUN\t,\t,\tPUNCT\tvirtsa\tvirtsa\tNOUN\t|-|\tMuu_oire\tKNIS
    """
    for filename in listdir(read_nersuite_folder):
        if filename.endswith(file_type):
            full_filepath = read_nersuite_folder + '/' + filename

            f_sent_save = open(save_keras_sent_folder + '/' + 'sent-' + filename[:-len('.' + file_type)] + '.txt', 'wb')
            f_doc_save = open(save_keras_doc_folder + '/' + 'doc-' + filename[:-len('.' + file_type)] + '.txt', 'wb')

            empty_ann = None

            with open(full_filepath, 'rb') as file:
                sent_annotation_set = set()
                sent_string = ''
                doc_annotation_set = set()
                doc_string = ''
                for line in file:
                    line = line.decode('utf-8').strip()
                    #print(line) #-------
                    if len(line) > 0:
                        # Normal conllu row for one word
                        line_parts = line.split('\t')

                        word = line_parts[3]
                        lemma = line_parts[4]
                        pos = line_parts[5]
                        sent_string += word + ' ' + lemma + ' ' + pos + '\t'
                        doc_string += word + ' ' + lemma + ' ' + pos + '\t'

                        annotation = line_parts[0]
                        # Find how the empty annotation looks like
                        if empty_ann is None:
                            if len(annotation) == 1:
                                empty_ann = annotation

                        annotation_parts = annotation.split('-AND-')
                        for i_annotation in annotation_parts:
                            sent_annotation_set.add(i_annotation[i_annotation.find('-') + 1:])
                            doc_annotation_set.add(i_annotation[i_annotation.find('-') + 1:])
                    elif len(line) == 0 and len(sent_string) > 0 and len(sent_annotation_set) > 0:
                        # Sentence has ended, add annotaion(s) and prepare for a new one
                        sent_string += '|-|\t'
                        if len(sent_annotation_set) > 1:
                            if empty_ann in sent_annotation_set:
                                sent_annotation_set.remove(empty_ann)
                        sent_string += ' '.join(sorted(map(str, sent_annotation_set)))
                        sent_string = sent_string.strip()
                        f_sent_save.write((sent_string + '\n').encode('utf-8'))

                        sent_annotation_set = set()
                        sent_string = ''
                    elif len(line) == 0 and len(doc_string) > 0 and len(doc_annotation_set) > 0:
                        # Document has ended, add annotaion(s) and prepare for a new one
                        doc_string += '|-|\t'
                        if len(doc_annotation_set) > 1:
                            if empty_ann in doc_annotation_set:
                                doc_annotation_set.remove(empty_ann)
                        doc_string += ' '.join(sorted(map(str, doc_annotation_set)))
                        doc_string = doc_string.strip()
                        f_doc_save.write((doc_string + '\n').encode('utf-8'))

                        doc_annotation_set = set()
                        doc_string = ''

                if len(sent_string) > 0:
                    # End of file, any unstored sentence?
                    sent_string += '|-|\t'
                    if len(sent_annotation_set) > 1:
                        if empty_ann in sent_annotation_set:
                            sent_annotation_set.remove(empty_ann)
                    sent_string += ' '.join(sorted(map(str, sent_annotation_set)))
                    sent_string = sent_string.strip()
                    f_sent_save.write((sent_string + '\n').encode('utf-8'))

                if len(doc_string) > 0:
                    # End of file, store the last document
                    doc_string += '|-|\t'
                    if len(doc_annotation_set) > 1:
                        if empty_ann in doc_annotation_set:
                            doc_annotation_set.remove(empty_ann)
                    doc_string += ' '.join(sorted(map(str, doc_annotation_set)))
                    doc_string = doc_string.strip()
                    f_doc_save.write((doc_string + '\n').encode('utf-8'))

            f_sent_save.close()
            f_doc_save.close()



if __name__ == "__main__":
    ####################################%%%%%%%%%%%%%%%%%%%%%
    parser = argparse.ArgumentParser(description='nersuite_num2keras_format.py')
    parser.add_argument('-read_nersuite', type=str, help='File in NERSuite format to read (.conll).', default='data/train-with-nersuite-num/')  # required=True)
    parser.add_argument('-file_type', type=str, help='Expected file type of the NERsuite files in read_nersuite_folder; default = "conll.num".', default='conll.num')  # required=True)
    parser.add_argument('-sent_save', type=str, help='Folder where to save Keras intended format with one sentence per line.', default='data/train-with-keras') #required=True)
    parser.add_argument('-doc_save', type=str, help='Folder where to save Keras intended format with one document per line.', default='data/train-with-keras')  # required=True)
    ####################################%%%%%%%%%%%%%%%%%%%%%
    args = parser.parse_args(sys.argv[1:])

    print('\nStart ...')
    nersuite2keras(read_nersuite_folder=args.read_nersuite, save_keras_sent_folder=args.sent_save, save_keras_doc_folder=args.doc_save, file_type=args.file_type)
    print('\nDone!')