import warnings
import logging

logging.captureWarnings(True)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from gensim import models
from gensim.corpora import Dictionary
from gensim import corpora
import matplotlib.pyplot as plt
import nltk
from wordcloud import WordCloud
from matplotlib import colors as mcolors

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


### WORD CLOUDS ----------------------------------------------------------------------------------------------------

# Define colors to use in the word cloud
cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]

# Define stop words to exclude from the word cloud
stop_words = nltk.corpus.stopwords.words('english')
stop_words.extend(['2', '3', 'ti', 'it', '1', 'due', 'behind', 'vice', 'versa', 'nt', 'also', 'us', 'one', 'could',
                   'would', 'till', 'le', 'el', 'among', 'eg', 'b', 'na', 'wan', 'pas', '10', '11', 'much', 'c', 'yes',
                   'no', 'ict', 'jie', '4M', '15M', '5M', 'however', 'within', 'else', 'still', 'http', 'probably',
                   'may', 'us', '4', '_x000d_', 'sometimes', 'nu', '15m', '79', 'p', 'sth', '8', 'even', 'mil', 'nu',
                   'cof', 'r', 'bu'])

# Define the WordCloud object to use for generating the word cloud
cloud = WordCloud(stopwords=stop_words,
                  background_color='white',
                  width=3000,
                  height=1600,
                  max_words=6,
                  colormap='tab10',
                  color_func=lambda *args, **kwargs: cols[i],
                  prefer_horizontal=1.0)

# Retrieve the topics generated from LDA
topics = lda_model.show_topics(formatted=False)

# Create a 2x2 grid of subplots to display the word clouds for each topic
fig, axes = plt.subplots(4, 2, figsize=(10,20), sharex=True, sharey=True)

# Generate a word cloud for each topic and display it in the corresponding subplot
for i, ax in enumerate(axes.flatten()):
    fig.add_subplot(ax)
    if i < len(topics):
        # Retrieve the top words for the current topic
        topic_words = dict(topics[i][1])
        # Generate the word cloud using the top words for the current topic
        cloud.generate_from_frequencies(topic_words, max_font_size=300)
        # Display the word cloud in the current subplot
        plt.gca().imshow(cloud)
        plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=20))
        plt.gca().axis('off')
        plt.savefig(f'/Users/cem_ataman/PycharmProjects/ConsulSUTD-Data-Analysis/results/final_visualizations/wordcloud.png')

# Adjust the layout and display the plot
plt.subplots_adjust(wspace=0, hspace=0)
plt.axis('off')
plt.margins(x=0, y=0)
plt.tight_layout()
plt.show()