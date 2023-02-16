import warnings
import logging

logging.captureWarnings(True)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from gensim import models
import pyLDAvis.gensim_models
from gensim import corpora
from gensim.corpora import Dictionary

### Call the corpus and dictionary and the final data ----------------------------
x  = open("/Users/cem_ataman/PycharmProjects/ConsulSUTD-Data-Analysis/data/intermediate_data/final_data.py", "r")
final_data = eval(x.readlines()[0])
x.close()

# Calling our dictionary
id2word = Dictionary.load('/Users/cem_ataman/PycharmProjects/ConsulSUTD-Data-Analysis/data/intermediate_data/LDA_model/id2word.dict')

# Calling our corpus
corpus = corpora.MmCorpus('/Users/cem_ataman/PycharmProjects/ConsulSUTD-Data-Analysis/data/intermediate_data/LDA_model/corpus.mm')

# Calling the LDA topic model
lda_model =  models.LdaModel.load('/Users/cem_ataman/PycharmProjects/ConsulSUTD-Data-Analysis/data/intermediate_data/LDA_model/LDA_model.model')

# Printing the LDA model topics
lda_topics = lda_model.show_topics()
for topic in lda_topics:
    print(topic)

### LDA Visualization--------------------------------------------------------------------------------------------------
# intertopic distance map
def intertopicmap ():
    vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word, mds="mmds", R=30)
    pyLDAvis.save_html(vis, '/Users/cem_ataman/PycharmProjects/ConsulSUTD-Data-Analysis/results/final_visualizations/Consul_LDA_visualization.html')

# run for the html
# intertopicmap()