#!/bin/bash
# Script by Hans Moen, hans.moen@utu.fi


# KIPU annotated data preparation
#########################################
## Setup ################################
SAVE_ROOT_FOLDER=/data/ikitik-processed-data/annotation/kipu

CODE_FOLDER=/data/ikitik-processed-data/annotation/code
TEXT_AND_ANN_FOLDER=${SAVE_ROOT_FOLDER}/text-and-ann
PARSED_CONLLU_FOLDER=${SAVE_ROOT_FOLDER}/parsed-conllu
SAVE_NERSUITE_FOLDER=${SAVE_ROOT_FOLDER}/train-with-nersuite
SAVE_ENUM_FOLDER=${SAVE_ROOT_FOLDER}/train-with-nersuite-num
SAVE_KERAS_FOLDER=${SAVE_ROOT_FOLDER}/train-with-keras

#WORD_MAPPINGS=${SAVE_ENUM_FOLDER}/word-mappings.txt
#ANN_MAPPINGS=${SAVE_ENUM_FOLDER}/ann-mappings.txt

TRAIN_PERCENTAGE=60
T_TAGS_ONLY=1
#########################################

echo "Create folders that might be missing"
mkdir -p ${SAVE_NERSUITE_FOLDER}
mkdir -p ${SAVE_ENUM_FOLDER}/simple
mkdir -p ${SAVE_ENUM_FOLDER}/full
mkdir -p ${SAVE_KERAS_FOLDER}/doc
mkdir -p ${SAVE_KERAS_FOLDER}/sent

echo "Combine text, ann and conllu files into a ann format and a nersuite format"
python ${CODE_FOLDER}/combine_text_conllu_ann.py -text ${TEXT_AND_ANN_FOLDER} -ann ${TEXT_AND_ANN_FOLDER} -conllu ${PARSED_CONLLU_FOLDER} -nersuite_save ${SAVE_NERSUITE_FOLDER}/annotations.conll -train_percentage ${TRAIN_PERCENTAGE} -t_only ${T_TAGS_ONLY}

echo "Map words and annotations to numbers, create a new conll file where these have been replaced by the corresponding numbers"
python ${CODE_FOLDER}/nersuite2num_format.py -conll ${SAVE_NERSUITE_FOLDER} -save ${SAVE_ENUM_FOLDER}/simple -simple 1
python ${CODE_FOLDER}/nersuite2num_format.py -conll ${SAVE_NERSUITE_FOLDER} -save ${SAVE_ENUM_FOLDER}/full -simple 0

echo "Convert the NERSuite enumerated version to Keras friendly data"
python ${CODE_FOLDER}/nersuite_num2keras_format.py -read_nersuite ${SAVE_ENUM_FOLDER}/full -file_type conll.num -sent_save ${SAVE_KERAS_FOLDER}/sent -doc_save ${SAVE_KERAS_FOLDER}/doc

echo "Show overview of annotation class distribution"
python ${CODE_FOLDER}/check_class_distribution.py -folder ${SAVE_NERSUITE_FOLDER} -save ${SAVE_ROOT_FOLDER}/annotation-counts.txt
