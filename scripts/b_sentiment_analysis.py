import warnings
import logging
import pandas as pd

logging.captureWarnings(True)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Create an instance of the sentiment analyzer
from nltk.sentiment import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

### Import dataframe from the excel file
df = pd.read_excel('ConsulSUTD Data.xlsx', sheet_name='Debates Content', index_col=0, header=0)

# Define a function that takes a string and returns a dictionary of sentiment scores for that string
def get_sentiment_scores(text):
    scores = sia.polarity_scores(text)
    return scores

# Get sentiment scores for each row in the dataframe
sentiment_scores = []
for i in df['Comment Text']:
    sentiment_scores.append(get_sentiment_scores(i))

# Add the sentiment score results to the dataframe by only using the "compound" values
df_sent = pd.DataFrame(sentiment_scores)
compound_df = df_sent.loc[:, 'compound']

compound_list = compound_df.values.tolist() # put the values as a list to add into the dataframe
df['sentiment scores'] = compound_list # add the results as a new column

# Select the useful columns and save the dataframe as an excel file
df_final = df.loc[:, ['debate_id', 'Comment', 'Reply', 'created_at', 'cached_votes_total', 'Comment Text', 'sentiment scores']]
df_final.to_excel('df_final.xlsx', index=False)
