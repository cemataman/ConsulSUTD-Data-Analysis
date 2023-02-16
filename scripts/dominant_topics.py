import warnings
import logging

logging.captureWarnings(True)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from pprint import pprint
import pandas as pd

from gensim import models
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

### CREATE A FUNCTION TO EXTRACT/FORMAT THE MAIN TOPIC AND KEYWORDS ----------------------------------------------------
def format_topics_sentences(ldamodel=None, corpus=None, texts=None):
    # Initialize output as a DataFrame
    sent_topics_df = pd.DataFrame()

    # For each document in the corpus, get the main topic and its keywords
    for i, row_list in enumerate(ldamodel[corpus]):
        # Extract the topic distribution for the current document
        # Check if per-word topics are enabled
        if ldamodel.per_word_topics:
            # If per-word topics are enabled, take the first element of row_list
            row = row_list[0]
        else:
            # If per-word topics are disabled, use row_list directly
            row = row_list
        # Sort the topics by their contribution to the current document
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # For each topic, get the keywords and the percentage of contribution to the document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # Only consider the dominant topic for each document
                # Get the keywords for the dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                # Add the dominant topic, its percentage of contribution, and its keywords to the output
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic, 4), topic_keywords]), ignore_index=True)
            else:
                break  # Exit the loop if we've considered the dominant topic

    # Rename the columns of the output
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add the original text to the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)

    # Return the output
    return sent_topics_df

### CALL THE FUNCTION TO EXTRACT DOMINANT TOPICS ----------------------------------------------------------------------------------------
# Call function to get topic keywords and their percentage of contribution to each document
df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=final_data)

# Create new DataFrame with renamed columns
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

# Create new DataFrame with top document for each dominant topic
sent_topics_sorteddf_mallet = pd.DataFrame()
sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')

for i, grp in sent_topics_outdf_grpd:
    sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet, grp.sort_values(['Perc_Contribution'], ascending=False).head(1)], axis=0)

# Reset index and rename columns of the new DataFrame
sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)
sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Representative Text"]


# Modify the display settings of the new DataFrame
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

# Display the new DataFrame and print topic keywords
pprint(sent_topics_sorteddf_mallet)

# Save the new dataframe as excel file
sent_topics_sorteddf_mallet.to_excel('/Users/cem_ataman/PycharmProjects/ConsulSUTD-Data-Analysis/results/dominant_topics.xlsx', index=False)


