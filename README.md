
# Consul SUTD Data Analsis

The Consul SUTD is a digital participation experiment conducted in the campus of the Singapore University of Technology and Design. Consul Project, an open-source and web-based application for citizen participation and open government, is modified and implemented for the experiment.

## Data Description
The dataset contains qualitative textual data gathered from the Debates and Proposals modules, where students exchanged ideas and opinions about campus and university policies. The data collection period spanned over 2 months and involved 47 students. The modules for participation had a hierarchical structure, allowing users to respond to the debate descriptions or comments from other users. This structure means that each debate topic includes additional discussions about the main topic description.

## Citation
Please include the following citation if you are interested in using this dataset or the methods:

Ataman, Cem, Bige Tuncer, and Simon Perrault. 2022. ”Asynchronous Digital Participation in Urban Design Processes: Qualitative Data Exploration and Analysis with Natural Language Processing”. In *POST-CARBON - Proceedings of the 27th Conference on Computer-Aided Architectural Design Research in Asia (CAADRIA),* 383–92, Sydney, Australia.

## How to use the code
It is recommended to follow the order of these three steps when executing the code:

Run the **data_cleaning.py** file to preprocess the data.
Next, execute **sentiment_analysis.py** to perform sentiment analysis on the preprocessed data.
Finally, run **topic_modeling.py** to build a topic model based on the preprocessed data and the results of sentiment analysis.

After creating the model, the visualization and analysis scripts can be used as per the analysis requirements. Please ensure to replace the file path with your own dataset.

### Folders & Scripts
- Data
    - ConsulSUTD Data.xlsx = initial dataset
    - intermediate_data = folder for intermediate data
        - LDA_model = the results of the topic model
- Results = all the final visuals and files
- Scripts = forder for all the scripts
    - *data_cleaning.py* : preprocessing the textual data
    - *topic_modeling.py* : formulating LDA topic model based on final data
    - *sentiment_analysis.py* : calculating sentiment scores of each argument
    - *dominant_topics.py* : detecting dominant topics and their weights
    - *dominant_topics_2.py* : detecting dominant topics and 3 keywords in each topic
- Visualizations
    - *tSNE_clustring.py* : creating a cluster map where each dot represents a document
    - *word_clouds.py* : creating word clouds for each topic
    - *LDA_intertopic_distance_map.py* : the most common visualization for LDA topic modeling.
