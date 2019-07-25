import os
import re
import json
import gensim
from nltk.stem.isri import ISRIStemmer
from gensim.utils import simple_preprocess
from smart_open import smart_open
from gensim import corpora
from pprint import pprint
import pandas as pd
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import numpy as np

#documents = [] # a list of dictionaries


def read_files():
	# this function reads all documents and returns a list of dictionaries (content as key and text as value)
	# make sure that the json files are in the diectory
	path_to_json = '/Users/elsayedissa/Desktop/pycharm/data'
	json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
	#print("Reading the following files: ",json_files) # prints a list of the json files
	documents = []
	# reading the json files in the specified directory
	for index, js in enumerate(json_files):
		with open(os.path.join(path_to_json, js)) as json_file:
			json_data = json.load(json_file)
			documents.append(json_data)
	return documents

def clean_data():
    #print ("Files cleaned ...")
    documents = read_files()
    tokens = []
    for docs in documents:
        for doc in docs:
            # print (doc)
            for line in doc['content']:
                text = re.sub(r'[\d+ a-zA-Z? & , \xd8 « » . :"،]', ' ',
                              line)  # remove non-alphabetical characters and non-arabic characters
                tkns = text.split()
                tokenss = []
                for token in tkns:
                    tokenss.append(token)
                tokens.append(tokenss)  # produces list of lists of tokens
    cleaned_data = [item for item in tokens if item != []]
    return cleaned_data


stemmer = ISRIStemmer()
data = clean_data() # this is a list of lists of tokens

def lemmatizer(token):
    #print ("Data lemmatized")
    token = stemmer.pre32(token)  # removes the three-letter and two-letter prefixes
    token = stemmer.suf32(token)  # removes the three-letter and two-letter suffixes
    token = stemmer.norm(token, num=1)  # removes diacritics
    return token

def stop_words():
    stop_words = stopwords.words('arabic')
    stop_words.extend(['في', 'من', 'على', 'أن', 'إلى', 'عن', 'الذي', 'مع', 'وكذا', 'وذلك'
                          , 'ما', 'أيضا', 'وهي', 'حتى', 'أخبارنا', 'المغربية', 'مغربية', 'مغرب', 'المغرب', 'محمد'])
    return stop_words

def make_bigrams(data):
    # create the bigrams given a minimum count
    bigrams = gensim.models.Phrases(data, min_count=5)
    #print(bigrams)
    bigrams_model = gensim.models.phrases.Phraser(bigrams)
    #print(bigrams_model)
    bigram_data = [bigrams_model[doc] for doc in data]
    print(bigram_data[0])
    return bigram_data

def pre_process():
    '''
    input: list of strings
    output:
    call functions in order
    '''
    # remove stop words
    no_stop_words = [[word for word in doc if word not in stop_words()] for doc in data]
    # lemmatizer
    lemmatized_data = [[lemmatizer(token) for token in item] for item in no_stop_words]
    # bigrams
    bigram_data = make_bigrams(lemmatized_data)
    return bigram_data


tokens = pre_process()
dictionary = corpora.Dictionary(tokens)
corpus = [dictionary.doc2bow(text) for text in tokens]


def LDA():
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=dictionary,
                                                num_topics=10,
                                                random_state=100,
                                                update_every=1,
                                                chunksize=100,
                                                passes=10,
                                                alpha='auto',
                                                per_word_topics=True)
    pprint (lda_model.print_topics())
    return lda_model


def format_topics_sentences(ldamodel=None, corpus=corpus, texts=data):
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


df_topic_sents_keywords = format_topics_sentences(ldamodel=LDA(), corpus=corpus, texts=data)

# Format
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
#print (df_dominant_topic)
print ('A sample of the keywords \n', df_dominant_topic['Keywords'][:1])
print ('A sample of the text \n', df_dominant_topic['Text'][:1])

df_dominant_topic['Text'].to_csv('sss.csv',sep='\t',header=True,index=False,encoding='utf-8')


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
#sent_topics_sorteddf_mallet['Representative Text'].to_csv('sentences1_lda.csv', sep='\t', header=True, index=False, encoding='utf-8')
###############################