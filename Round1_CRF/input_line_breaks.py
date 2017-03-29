from os import listdir
from os.path import isfile
from os.path import splitext
import argparse
from os import sys
import codecs


# TXT:          /data/finished-annotation/reference-corpus/episodes-kipu-000-006-008-014-consensus-revised-by-kemheik/consensus/  + SUBFOLDERS!
# CoNLL/fincgs: /data/finished-annotation/reference-corpus/episodes-fincgs

# Code@Ikitik:  /data/ikitik-processed-data/annotation/code/

###########################
# Some static parameteres #
###########################
CONLL_FILE_EXTENSION = "fincg"
TXT_FILE_EXTENSION = "txt"
###########################

class InputLineBreaks:
    def __init__(self, conll_folder, txt_folder, save_folder):
        self._conll_folder = conll_folder
        self._txt_folder = txt_folder
        self._save_folder = save_folder

        self._conll_filenames_fullpath = {}
        self._txt_filenames_and_fullpath = {}


    # Read the CoNLL files and TXT files into dictionaries containing the filenames and their full paths.
    def load_filenames(self):

        # CoNLL files:
        for filename in listdir(self._conll_folder):
            full_file_path = self._conll_folder + "/" + filename
            #print(filename) #-------------
            if isfile(full_file_path) and filename.lower().endswith("." + CONLL_FILE_EXTENSION):
                self._conll_filenames_fullpath[splitext(filename)[0]] = full_file_path
                #print("CoNLL:::: " + splitext(filename)[0] + " -- " + full_file_path) #-----------

        # TXT files:
        for filename in listdir(self._txt_folder):
            full_file_path = self._txt_folder + "/" + filename
            #print(filename) #-------------
            if isfile(full_file_path) and filename.lower().endswith("." + TXT_FILE_EXTENSION):
                self._txt_filenames_and_fullpath[splitext(filename)[0]] = full_file_path
                #print("TXT:::: " + splitext(filename)[0] + " -- " + full_file_path) #-----------


    def input_linebreaks_from_txt(self):
        files_processed_count = 0

        # Go through each CoNLL file
        for i_conll_filename, i_conll_and_fullpath in self._conll_filenames_fullpath.items():

            # Find its corresponding TXT file
            i_txt_file = None
            if (i_conll_filename in self._txt_filenames_and_fullpath):


                f_save = open(self._save_folder + "/" + i_conll_filename + "." + CONLL_FILE_EXTENSION, 'wt')


                linebreak_offsets = []

                # file -ib <txt filename>  -->  inode/symlink; charset=binary

                # Go through each TXT file char by char
                #f = codecs.open('in', 'r', 'utf8')
                with codecs.open(self._txt_filenames_and_fullpath[i_conll_filename], 'r', 'utf-8') as f:
                    i = 0
                    last_char_offset = 0

                    while True:
                        c = f.read(1)
                        i += 1

                        if not c:
                            break
                            #print "END OF FILE" #----------


                        if c != " " and c != "\t" and c != "\n":
                            last_char_offset = i
                        #else:
                        #    print("SPACE! i = " + str(i) + ", last offset = " + str(last_char_offset)) # + ", c = '" + c + "'") #---------

                        #print("CHAR: = '" + c + "' \t i = " + str(i) + ", last char = " + str(last_char_offset)) #-------------

                        if c == "\n":
                            #print("\t\t>>line-break>> i = " + str(i) + ", range = [" + str(','.join(str(indx) for indx in range(last_char_offset,i))) + "]") #--------
                            linebreak_offsets.extend(range(last_char_offset,i))
                            #linebreak_offsets.append(i)
                            last_char_offset = i-1
                            #print("LAST CHAR OFFSET = " + str(last_char_offset)) #--------

                # Go through each CoNLL file line by line
                with open(i_conll_and_fullpath, 'rt') as f_conll:
                    for line in f_conll:
                        #print("LENGTH = " + str(len(line)))
                        if line:
                            line_segments = line.split()
                            if len(line_segments) >= 4:
                                #print(line_segments) #---------------
                                #i_word_offset_start = int(line_segments[0])
                                i_word_offset_end = int(line_segments[1])
                                #i_word_text = line_segments[2]
                                f_save.write(line)
                                if i_word_offset_end in linebreak_offsets:
                                    f_save.write("\n")
                            else:
                                f_save.write(line)

                f_save.close()
                files_processed_count += 1
        return files_processed_count


############################################################################
#Main ...
if __name__ == "__main__":
    # Argument handling
    parser = argparse.ArgumentParser(description='Add line breaks in CoNLL from TXT files arguments')
    parser.add_argument('-conll', nargs='+') # CoNLL folder
    parser.add_argument('-txt', nargs='+') # TXT folder
    parser.add_argument('-s', nargs='+') # Save filename
    args = None
    if len(sys.argv) == 1:
        args = parser.parse_args('-conll data/conll/ -txt data/txt/ -s data/conll_with_linebreaks/'.split()) # For debugging ---------
    else:
        args = parser.parse_args(sys.argv[1:])

    print("Arguments: " + str(args))
    print(" ... processing ... ")
    lb = InputLineBreaks(''.join(str(conll) for conll in args.conll), ''.join(str(txt) for txt in args.txt), ''.join(str(save) for save in args.s))
    lb.load_filenames()
    files_processed = lb.input_linebreaks_from_txt()

    print("Done!\t" + str(files_processed) + " file(s) processed.")
    #print("Discontinued annotations = " + str(discontinued_annotation_count)) #----------