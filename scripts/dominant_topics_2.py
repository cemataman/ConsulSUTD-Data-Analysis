import pandas as pd

from gensim import corpora, models
from gensim.corpora import Dictionary
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

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

def topics_per_document(model, corpus, start=0, end=1):
    """Helper function to get the dominant topic and percentage for each document in the corpus"""
    corpus_sel = corpus[start:end]
    dominant_topics = []
    topic_percentages = []
    for i, corp in enumerate(corpus_sel):
        topic_percs, wordid_topics, wordid_phivalues = model[corp]
        dominant_topic = sorted(topic_percs, key=lambda x: x[1], reverse=True)[0][0]
        dominant_topics.append((i, dominant_topic))
        topic_percentages.append(topic_percs)
    return (dominant_topics, topic_percentages)

# Get the dominant topics and topic percentages for each document in the corpus using the topics_per_document function
dominant_topics, topic_percentages = topics_per_document(model=lda_model, corpus=corpus, end=-1)

# Distribution of Dominant Topics in Each Document
# Create a Pandas DataFrame to group the documents by their dominant topic and get the count for each topic
df = pd.DataFrame(dominant_topics, columns=['Document_Id', 'Dominant_Topic'])
dominant_topic_in_each_doc = df.groupby('Dominant_Topic').size()
df_dominant_topic_in_each_doc = dominant_topic_in_each_doc.to_frame(name='count').reset_index()

# Total Topic Distribution by actual weight
# Create a Pandas DataFrame to get the total topic distribution across all documents
topic_weightage_by_doc = pd.DataFrame([dict(t) for t in topic_percentages])
df_topic_weightage_by_doc = topic_weightage_by_doc.sum().to_frame(name='count').reset_index()

# Top 3 Keywords for each Topic
# Get the top 3 keywords for each topic in the LDA model
topic_top3words = [(i, topic) for i, topics in lda_model.show_topics(formatted=False)
                                 for j, (topic, wt) in enumerate(topics) if j < 3]
df_top3words_stacked = pd.DataFrame(topic_top3words, columns=['topic_id', 'words'])
df_top3words = df_top3words_stacked.groupby('topic_id').agg(', \n'.join)
df_top3words.reset_index(level=0, inplace=True)

# Plot
fig, ax1 = plt.subplots()

# Topic Distribution by Dominant Topics
# Plot the distribution of dominant topics in each document as a bar chart
ax1.bar(x='Dominant_Topic', height='count', data=df_dominant_topic_in_each_doc, width=.5, color='indianred')
# Format the x-axis labels to include the topic number and the top 3 keywords for that topic
ax1.set_xticks(range(df_dominant_topic_in_each_doc.Dominant_Topic.nunique()))
tick_formatter = FuncFormatter(lambda x, pos: 'Topic ' + str(x) + '\n' + df_top3words.loc[df_top3words.topic_id == x, 'words'].values[0])
ax1.xaxis.set_major_formatter(tick_formatter)
# Set the title and axis labels for the plot
ax1.set_title('Dominant Topics', fontdict=dict(size=10))
ax1.set_ylabel('Number of Arguments')
ax1.set_ylim(0, 35)


### Plot the outputs
plt.subplots_adjust(bottom=0.2)
# Save the plot as a PNG file
plt.savefig('/Users/cem_ataman/PycharmProjects/ConsulSUTD-Data-Analysis/results/final_visualizations/dominant_topics.png')
# Show the plot on the screen
plt.show()
