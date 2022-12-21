import warnings
import logging

logging.captureWarnings(True)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from gensim import models
import pyLDAvis.gensim_models
from gensim import corpora

### define the corpus and dictionary in the same way from the final data ----------------------------

x = open("/Users/cem_ataman/PycharmProjects/ConsulSUTD-Data-Analysis/data/intermediate_data/final_data.py", "r")
final_data = eval(x.readlines()[0])
x.close()

id2word = corpora.Dictionary(final_data)

corpus = []
for text in final_data:
    new = id2word.doc2bow(text) # create a bag of words (bow)
    corpus.append(new)

# Call the LDA topic model
lda_model =  models.LdaModel.load('/Users/cem_ataman/PycharmProjects/ConsulSUTD-Data-Analysis/data/intermediate_data/LDA_model/LDA_model.model')
lda_model.show_topics()

###--------------------------------------------------------------------------------------------------

# intertopic distance map
def intertopicmap ():
    vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word, mds="mmds", R=30)
    pyLDAvis.save_html(vis, '/Users/cem_ataman/PycharmProjects/ConsulSUTD-Data-Analysis/results/Consul_LDA_visualization.html')

# intertopicmap()