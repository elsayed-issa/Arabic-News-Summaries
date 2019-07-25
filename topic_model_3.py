# import modules
import os
import re
import json
import gensim
import nltk
from nltk.stem.isri import ISRIStemmer
from gensim.utils import simple_preprocess
from smart_open import smart_open
from gensim import corpora
from pprint import pprint
import pandas as pd
from collections import Counter
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from gensim.models import CoherenceModel

############# (1) Reading the json files #############

# make sure that the json files are in the diectory
path_to_json = '/Users/elsayedissa/Desktop/pycharm/data'
json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
print("Reading the following files: ",json_files) # prints a list of the json files


documents=[] # outputs all the data in all the doucments in a form of list of dictionaries
# reading the json files in the specified directory
for index, js in enumerate(json_files):
    with open(os.path.join(path_to_json, js)) as json_file:
        json_data = json.load(json_file)
        documents.append(json_data)
#print (documents)


############# (2) cleaning the data #############

for docs in documents:
    for doc in docs:
        for k,v in doc.items():
            tokens.append(v)






list_of_cleaned_data = []
for item in tokens:
    for line in item:
        line = re.sub(r'[\d+ a-zA-Z? & , \xd8 \xa0 « » . :"،]', ' ', line)
        list_of_cleaned_data.append(line)



#cleaned = []
#for item in tokens:
    #item = re.sub(r'[\d+ a-zA-Z? & , \xd8 \xa0 « » . :"،]', ' ', line) # remove non-alphabetical characters and non-arabic characters
   # print (item)
#print (doc)
        #for line in doc['content']:
            #print (line)
            #line1 = re.sub(r'[\d+ a-zA-Z? & , \xd8 \xa0 « » . :"،]', ' ', line) # remove non-alphabetical characters and non-arabic characters
            #print (line1)
            #tokens.append(line)
            #tkns = text.split()
            #tokenss = []
            #for token in tkns:
                #tokenss.append(token)
            #tokens.append(tokenss) # produces list of lists of tokens
#print ('The number of lists of tokens including the empty [] is',len(tokens))

# cleaning the data from empty lists
#cleaned_data = [item for item in tokens if item != []]
#print ("The No. of lists of tokens in the files is ",len(cleaned_data)) # there are 17 lists after removing the empty ones.

#print (cleaned_data)

'''
############# (3) getting rid of the stop words #############
# updating the NLT corpus with Arabic stopwords found on the github
# https://github.com/mohataher/arabic-stop-words
stop_words = stopwords.words('arabic')
stop_words.extend(['في','من','على','أن','إلى','عن','الذي','مع','وكذا','وذلك'
                   ,'ما','أيضا','وهي','حتى','أخبارنا','المغربية','مغربية','مغرب','المغرب','محمد'])

# get rid of the stop words
no_stopwords_data = [[word for word in doc if word not in stop_words if len(word)>3] for doc in cleaned_data]

print (no_stopwords_data[0:2])

############# (5)lemmatizing the data #############
# produces a list of lists of the data lemmatized ... the lemmatizer does not work well when lemmatizing suffixes
stemmer = ISRIStemmer()

lemmatized_data = []
for items in no_stopwords_data:
    lemmas = []
    for token in items:
        token = stemmer.pre32(token) # removes the three-letter and two-letter prefixes
        token = stemmer.suf32(token) # removes the three-letter and two-letter suffixes
        token = stemmer.norm(token, num=1) # removes diacritics
        lemmas.append(token)
    lemmatized_data.append(lemmas)
print (lemmatized_data[0:2])

# ############ (4) the Bigrams #################
# create the bigrams given a minimum count
bigrams = gensim.models.Phrases(lemmatized_data, min_count=5)
print (bigrams)

bigrams_model = gensim.models.phrases.Phraser(bigrams)
print (bigrams_model)

bigram_data = [bigrams_model[doc] for doc in lemmatized_data]
print (bigram_data[0:2])



############# (5) Preparing the data using gensim for the Model #############
# the preprocess using gensim involves buidling the dictionary, the corpus and the bigrams
# the data is (data_after_lemmatization) and it is a list of lists

# The Dictionary
dictionary = corpora.Dictionary(lemmatized_data)

# the corpus
corpus = [dictionary.doc2bow(d) for d in lemmatized_data]
print (corpus[0:2])

############# (6) The Model #############
#the model
print ('Please Wait, Printing Topics ... ')
for k in [100]:
    lda = gensim.models.ldamodel.LdaModel (corpus=corpus,
                                                   id2word=dictionary,
                                                   num_topics=k,
                                                   random_state=100,
                                                   update_every=1,
                                                   chunksize=100,
                                                   passes=10,
                                                   alpha='auto',
                                                   per_word_topics=True)
    print ('Number of topics %d' % k)
    print ('perplexity: %d' % lda.log_perplexity(corpus))
    coherence=gensim.models.CoherenceModel(model=lda, corpus=corpus,
                                          coherence='u_mass')
    print ('coherence: %d' % coherence.get_coherence())

############# (7) Dominant Topics #############
print ('Please Wait, Printing the Dominant Topics ... ')
def format_topics_sentences(ldamodel=None, corpus=corpus, texts=cleaned_data):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row_list in enumerate(ldamodel[corpus]):
        row = row_list[0] if ldamodel.per_word_topics else row_list
        #print(row)
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)


df_topic_sents_keywords = format_topics_sentences(ldamodel=lda, corpus=corpus, texts=cleaned_data)

# Format
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
#print (df_dominant_topic)
print ('A sample of the keywords \n', df_dominant_topic['Keywords'][:1])
print ('Asample of the text \n', df_dominant_topic['Text'][:1])

# save to file
df_dominant_topic['Text'].to_csv('dominant_topics_lda22.csv', sep='\t', header=True, index=False, encoding='utf-8')

for item in df_dominant_topic['Text']:
    print (len(item))


############# (8) rep sentences #############
# Most representative sentence for each topic
# Display setting to show more characters in column
pd.options.display.max_colwidth = 0

sent_topics_sorteddf_mallet = pd.DataFrame()
sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')

for i, grp in sent_topics_outdf_grpd:
    sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet,
                                             grp.sort_values(['Perc_Contribution'], ascending=False).head(1)],
                                            axis=0)

# Reset Index
sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)

# Format
sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Representative Text"]

# Show
print (sent_topics_sorteddf_mallet)

# save to file
sent_topics_sorteddf_mallet.reset_index().to_csv('sentences_lda22.csv', sep='\t', header=True, index=False, encoding='utf-8')

################################
# Group top 5 sentences under each topic
sent_topics_sorteddf_mallet = pd.DataFrame()

sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')

for i, grp in sent_topics_outdf_grpd:
    sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet,
                                             grp.sort_values(['Perc_Contribution'], ascending=[0]).head(1)],
                                            axis=0)

# Reset Index
sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)

# Format
sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Text"]

# Show
sent_topics_sorteddf_mallet

sent_topics_sorteddf_mallet['Text'].to_csv('summary100_22.csv', sep='\t', header=True, index=False, encoding='utf-8')






# Number of Documents for Each Topic
topic_counts = df_topic_sents_keywords['Dominant_Topic'].value_counts()

# Percentage of Documents for Each Topic
topic_contribution = round(topic_counts / topic_counts.sum(), 4)

# Topic Number and Keywords
topic_num_keywords = df_topic_sents_keywords[['Dominant_Topic', 'Topic_Keywords']]

# Concatenate Column wise
df_dominant_topics = pd.concat([topic_num_keywords, topic_counts, topic_contribution], axis=1)

# Change Column names
df_dominant_topics.columns = ['Dominant_Topic', 'Topic_Keywords', 'Num_Documents', 'Perc_Documents']

# Show
df_dominant_topics

'''