import warnings
import logging

logging.captureWarnings(True)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from gensim import models
from gensim.corpora import Dictionary
from gensim import corpora
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE

from bokeh.plotting import figure
from bokeh.io import output_file, save, show
from bokeh.io import export_png

import matplotlib.colors as mcolors

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



### t-SNE CLUSTERING ----------------------------------------------------------------------------------------------------
# Get topic weights for each document in the corpus using the LDA model
topic_weights = []
for i, row_list in enumerate(lda_model[corpus]):
    topic_weights.append([w for i, w in row_list[0]])

# Convert the list of topic weights to a NumPy array and fill missing values with zeros
arr = pd.DataFrame(topic_weights).fillna(0).values

# Keep only the rows with well-separated points (optional)
arr = arr[np.amax(arr, axis=1) > 0.35]

# Find the dominant topic number for each document by computing the argmax of the array
topic_num = np.argmax(arr, axis=1)

# Perform t-SNE dimensionality reduction on the array to obtain 2D coordinates
tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='pca') #angle: smaller is more accurate but slower. 0.99 = speed over accuracy.
tsne_lda = tsne_model.fit_transform(arr)

# Create a color map for the different topics using the Tableau color scheme
n_topics = 8
mycolors = np.array([color for name, color in mcolors.TABLEAU_COLORS.items()])

# Create a scatter plot of the t-SNE coordinates, with each point colored by its dominant topic
plot = figure(title="t-SNE Clustering of {} LDA Topics".format(n_topics),
              width=800, height=600)
plot.scatter(x=tsne_lda[:,0], y=tsne_lda[:,1], color=mycolors[topic_num])

### FINAL DISPLAY AND SAVE OUTPUTS

# Display the plot
pprint(plot)

# specify the output file and save them
### as HTML
# output_file("/Users/cem_ataman/PycharmProjects/ConsulSUTD-Data-Analysis/results/final_visualizations/tsne.html")
# save(plot)

###as PNG
# export_png(plot, filename="/Users/cem_ataman/PycharmProjects/ConsulSUTD-Data-Analysis/results/final_visualizations/tsne.png")