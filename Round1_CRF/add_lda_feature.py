import argparse
import sys

# For adding LDA topic features
sys.path.append("/Users/hansmoe/Dropbox/MyProgramming/PythonProjects/PyCharm/Gensim_LDA")
sys.path.append("/data/ikitik-processed-data/annotation/code/lda")
import lda_test
from CRF import text_filter as filter


def init_text_filter(fstoplist=None, lowercasing=True, min_word_length=1):
    filter.init(fstoplist=fstoplist, lowercasing=lowercasing, min_word_length=min_word_length)


# Add LDA topics as a features. Each word in a sentence gets the same topic ID (Ex: 'TOPIC_2')
def add_lda_topics_1file(rfile, wfile, lda_model, lda_dict, lemma_column=4):
    read_file = open(rfile, mode="r", encoding="utf-8")
    write_file = open(wfile, mode="w", encoding="utf-8")

    # Reset file reader's position to the beginning of the file ... because I don't know
    read_file.seek(0)

    # Load the LDA model and its dictionary
    print("Loading LDA model and dictionary ... ")
    lda = lda_test.LDA_test()
    lda.load_model(lda_model)
    lda.load_dictionary(lda_dict)
    print("Loading LDA model done.")

    print("Adding features ... ")

    # Read all lemmas in the next sentence
    conll_rows = [] # a list of lists
    sent_lemmas = ""
    for line in read_file:
        if len(line.strip()) > 0: # Assuming this is a normal conll row
            line_parts = line.strip().split("\t")
            conll_rows.append(line_parts)
            i_lemma = line_parts[lemma_column] # <--- NB!
            if not sent_lemmas:
                sent_lemmas = str(i_lemma)
            else:
                sent_lemmas += " " + str(i_lemma)

        elif len(line.strip()) == 0: # Assuming that this is a sentence break
            if sent_lemmas:
                # Filter the text ...
                sent_lemmas = filter.filter(sent_lemmas)
                sent_lemmas_list = sent_lemmas.split()
                # Calculate the closest topic ID for this sentence
                topic_id = lda.get_max_topic_nr(sent_lemmas_list)
                #print(sent_lemmas_list) #--------
                if topic_id is not None: # Add this topic ID to the conll rows for the sentence
                    for i in range(0, len(conll_rows)):
                        conll_rows[i].append("TOPIC_" + str(topic_id))
                else: # No topic found, adding -1
                    for i in range(0, len(conll_rows)):
                        conll_rows[i].append("TOPIC_" + str(-1))
                # Write this new rows to the write_file
                for i_row in conll_rows:
                    write_file.write('\t'.join(i_row) + "\n")
                write_file.write("\n") # And add back the empty line ...
            conll_rows = []
            sent_lemmas = ""

    # Last sentence is still missing, add it too ...
    if sent_lemmas:
        if sent_lemmas:
            # Filter the text ...
            sent_lemmas = filter.filter(sent_lemmas)
            sent_lemmas_list = sent_lemmas.split()
            # Calculate the closest topic ID for this sentence
            topic_id = lda.get_max_topic_nr(sent_lemmas_list)
            #print(sent_lemmas_list) #--------
            if topic_id is not None: # Add this topic ID to the conll rows for the sentence
                for i in range(0, len(conll_rows)):
                    conll_rows[i].append("TOPIC_" + str(topic_id))
            else: # No topic found, adding -1
                for i in range(0, len(conll_rows)):
                    conll_rows[i].append("TOPIC_" + str(-1))
            # Write this new rows to the write_file
            for i_row in conll_rows:
                write_file.write('\t'.join(i_row) + "\n")
            write_file.write("\n") # And add back the empty line ...
        #conll_rows = []
        #sent_lemmas = ""

    read_file.close()
    write_file.close()


#Main ...
if __name__ == "__main__":
    ###############################################
    # Argument handling
    parser = argparse.ArgumentParser(description='add_feature_to_nersuite')
    parser.add_argument('-f', help="Input (CoNLL) filename", required=True) # Input (CoNLL) file
    parser.add_argument('-lemma_column', help="Column number where the lemmas are in the CoNLL file, 0 equals the first one; default is 4", type=int, default=4)
    parser.add_argument('-s', help="Save (CoNLL) filename", required=True) # Save file
    parser.add_argument('-lda', help="LDA model file to use. Ex: /data/ikitik-processed-data/ehrdata-prep/doctors-and-nurses/lda/lda-doctors-nurses-doclb-topics100.lda", required=True) # LDA file to use
    parser.add_argument('-lda_dict', help="Dictionary belonging the LDA model file. Ex: /data/ikitik-processed-data/ehrdata-prep/doctors-and-nurses/lda/lda-doctors-nurses-doclb-topics100.dict", required=True) # Dictionary for the LDA file
    parser.add_argument('-stoplist', help="Stoplist for the text filter", default=None) # Stoplist for the text filter
    args = None
    if len(sys.argv) == 1:
        args = parser.parse_args('-f data/add_features/input_file.nersuite -s data/add_features/output_file.nersuite -lda /Users/hansmoe/Dropbox/MyProgramming/PythonProjects/PyCharm/Gensim_LDA/data/models/ldamodel.lda -lda_dict /Users/hansmoe/Dropbox/MyProgramming/PythonProjects/PyCharm/Gensim_LDA/data/models/ldamodel.dict -stoplist stoplist.txt'.split()) # For debugging ---------
    else:
        args = parser.parse_args(sys.argv[1:])


    print("Start ... ")
    init_text_filter(fstoplist=args.stoplist)
    add_lda_topics_1file(rfile=args.f, wfile=args.s, lda_model=args.lda, lda_dict=args.lda_dict, lemma_column=args.lemma_column)
    print("Done.")
