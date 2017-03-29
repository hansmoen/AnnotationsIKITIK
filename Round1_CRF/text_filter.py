import re
import sys


stoplist = set()
lower = True
min_word_len = 1
filter1 = re.compile('\W+', re.U) # Remove non-alphanumeric and underscore characters
#filter2 = re.compile("^\d+\s|\s\d+\s[\d+$]*|\s\d+$", re.U) # Remove terms containing only digits
filter2 = re.compile("\s+[\w-]*\d[\w-]*|[\w-]*\d[\w-]*\s*", re.U) # Remove terms containing any digits


def init(fstoplist=None, lowercasing=True, min_word_length=1):
    global min_word_len, stoplist, lower
    if fstoplist:
        load_add_stoplist(fstoplist)
    lower = lowercasing
    min_word_len = min_word_length

def get_stoplist_length():
    global stoplist
    return len(stoplist)

def load_add_stoplist(fstoplist):
    global stoplist
    if fstoplist:
        nstopwords_loaded = 0
        if sys.version_info.major == 2:
            with open(fstoplist, 'r') as slist:
                for line in slist:
                    stoplist.add(line.decode('utf-8').strip())
                    nstopwords_loaded += 1
        elif sys.version_info.major == 3:
            with open(fstoplist, 'r', encoding='utf-8') as slist:
                for line in slist:
                    stoplist.add(line.strip())
                    nstopwords_loaded += 1
        print(str(nstopwords_loaded) + " stop words loaded (tot=" + str(len(stoplist)) + ").")

def clear_stoplist():
    global stoplist
    stoplist = set()

def filter(text_string):
    global min_word_len, stoplist, lower
    if lower:
        text_string = text_string.lower()
    #print("\n[0]\t" + text_string) #--------
    text_string = filter1.sub(" ",text_string)
    #print("[1]\t" + text_string) #--------
    text_string = filter2.sub(" ",text_string)
    #print("[2]\t" + text_string) #--------

    text_parts = text_string.split()

    # Remove short words
    if min_word_len > 1:
        text_parts = text_string.split()
        temp_text_parts = []
        for w in text_parts:
            if len(w) >= min_word_len:
                temp_text_parts.append(w)
        text_parts = temp_text_parts

    # Remove stopwords
    if stoplist:
        temp_text_parts = []
        for w in text_parts:
            if w not in stoplist:
                temp_text_parts.append(w)
        text_parts = temp_text_parts

    if text_parts:
        return ' '.join(text_parts)
    else:
        return text_string



