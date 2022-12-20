import warnings
import logging
from os import mkdir

logging.captureWarnings(True)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import pandas as pd
import re

from nltk.stem.porter import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.wordnet import  WordNetLemmatizer
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

# Import Data From ConsulSUTD excel file -----------------------
def read_consul (fn = 'ConsulSUTD Data.xlsx'):
    df = pd.read_excel(fn, sheet_name='Debates Content', index_col=0, header=0)
    return(df)

# CLEANING DATA ------------------------------------------------
# LOWERCASE ----------------------------------------------------
def make_lower (data):
    clean_text_1 = []
    for i in data.index:
        clean_text_1.append(str(data["Comment Text"][i]).lower())
    return (clean_text_1)

# TOKENIZE SENTENCE --------------------------------------------
def sent_token (data):
    sent_tok = []
    for sent in data:
        sent = sent_tokenize(sent)
        sent_tok+=sent
    return (sent_tok)

def word_token (data):
    return ([word_tokenize(i) for i in data])

# PUNCTUATION
def remove_punc (list_punc):
    clean_biglist = []
    for words in list_punc:
        clean = []
        for w in words:
            res = re.sub(r'[^\w\s]', "", w)
            if res != "":
                clean.append(res)
        clean_biglist.append(clean)
    return clean_biglist

# SPECIAL CHARACTERS -------------------------------------------
def remove_char (list_punc):
    clean_char = []
    for words in list_punc:
        clean = []
        for item in words:
            l1 = item.removesuffix('_x000d_')
            if l1 != "":
                clean.append(l1)
        clean_char.append(clean)
    return clean_char

# STOPWORDS ----------------------------------------------------
def remove_stopword (list_stops):
    clean_biglist = []
    stop_words = nltk.corpus.stopwords.words('english')
    stop_words.extend(['2','3','ti','it','1','due','behind','vice','versa',
                       'nt','also','us','one','could','would','till','le','el',
                       'among','eg','b','na','wan','pas','10','11','much',
                       'c','yes','no','ict','jie','4M','15M','5M','however',
                       'within','else','still','http','probably','may','us',
                       '4','_x000d_','sometimes','nu','15m','79','p','sth','8',
                       'even', 'nan','93','dtype','comment1','h', 'content_id','cpf'])
    for words in list_stops:
        w = []
        for word in words:
            if not word in stop_words:
                w.append(word)
        clean_biglist.append(w)
    return clean_biglist

# STEMMING -----------------------------------------------------
def stem_text (list_stem):
    porter = PorterStemmer()
    stem_biglist = []
    for words in list_stem:
        w = []
        for word in words:
            w.append(porter.stem(word))
        stem_biglist.append(w)
    return stem_biglist

# LEMITIZATION -------------------------------------------------

def lemit_text (list_lemit):
    wnet = WordNetLemmatizer()
    lemit_biglist = []
    for words in list_lemit:
        w = []
        for word in words:
            w.append(wnet.lemmatize(word))
        lemit_biglist.append(w)
    return lemit_biglist

# Call Functions -----------------------------------------------
df = read_consul()
clean_text_1 = make_lower(df)
print(clean_text_1)
clean_text_2 = sent_token(clean_text_1)
print(clean_text_2)
clean_text_3 = word_token(clean_text_2)
print(clean_text_3)
clean_text_4 = remove_punc(clean_text_3)
print(clean_text_4)
clean_text_5 = remove_stopword(clean_text_4)
print(clean_text_5)
clean_text_6 = lemit_text(clean_text_5)
print(clean_text_6)
clean_text_7 = remove_char(clean_text_6)
print(clean_text_7)
clean_text_8 = stem_text(clean_text_7)
print(clean_text_8)
final_data = clean_text_8
print(final_data)

# x = open("Final_Data.py", "w+")
# x.write(str(clean_text_7))
# x.close()
