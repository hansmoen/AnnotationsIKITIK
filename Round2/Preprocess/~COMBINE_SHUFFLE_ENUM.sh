#!/bin/bash
# Script by Hans Moen, hans.moen@utu.fi


#########################################
## Setup ################################
CODE_FOLDER=/data/ikitik-processed-data/annotation/code
TEXT_AND_ANN_FOLDER=text-and-ann
PARSED_CONLLU_FOLDER=parsed-conllu

#SAVE_COMBINED=conll-with-ann/combined.conll
#SAVE_SHUFFLED=shuffled-conll-with-ann/combined.shuffled.conll
#SAVE_ENUM=shuffled-num-conll-with-ann/combined.shuffled.num.conll
#WORD_MAPPINGS=shuffled-num-conll-with-ann/word-mappings.txt
#ANN_MAPPINGS=shuffled-num-conll-with-ann/ann-mappings.txt

SAVE_COMBINED=combined.conll
SAVE_SHUFFLED=combined.shuffled.conll
SAVE_ENUM=combined.shuffled.num.conll
WORD_MAPPINGS=word-mappings.txt
ANN_MAPPINGS=ann-mappings.txt

T_TAGS_ONLY=1
#########################################


echo "Combine conllu files and ann files into one file"
python ${CODE_FOLDER}/combine_text_conllu_ann.py -text ${TEXT_AND_ANN_FOLDER} -ann ${TEXT_AND_ANN_FOLDER} -conllu ${PARSED_CONLLU_FOLDER} -save ${SAVE_COMBINED} -t_only ${T_TAGS_ONLY}

echo "Shuffle sentences in the combined file"
python ${CODE_FOLDER}/shuffle_conll_sentences.py -conll ${SAVE_COMBINED} -save ${SAVE_SHUFFLED}

echo "Map words and annotations to numbers, create a new conll file where these have been replaced by the corresponding numbers"
python ${CODE_FOLDER}/annotations_to_num.py -conll ${SAVE_SHUFFLED} -save ${SAVE_ENUM} -word_mappings ${WORD_MAPPINGS} -ann_mappings ${ANN_MAPPINGS}