import warnings
import logging

logging.captureWarnings(True)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import gensim.utils
from gensim import corpora, models, similarities

### Bring the pre-processed data from the previous steps
x  = open("/Users/cem_ataman/PycharmProjects/ConsulSUTD-Data-Analysis/data/intermediate_data/final_data.py", "r")
final_data = eval(x.readlines()[0])
x.close()

### Creating our Corpus
id2word = corpora.Dictionary(final_data) #Initilize a dictionary (list of words) by using our dataset

# Creating our corpus: a list of tuples corresponding the word id and their frequency in text
corpus = []
for text in final_data:
    new = id2word.doc2bow(text) # create a bag of words (bow)
    corpus.append(new)

# Creating LDA model by defining the hyperparameters
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                            id2word=id2word,
                                            num_topics=8,
                                            random_state=100,
                                            update_every=8,
                                            chunksize=100,
                                            passes=10,
                                            alpha="auto",
                                            per_word_topics=True)

# Saving the LDA model
lda_model.save('/Users/cem_ataman/PycharmProjects/ConsulSUTD-Data-Analysis/data/intermediate_data/LDA_model/LDA_model.model')

