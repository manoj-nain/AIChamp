#Using nlp to remove stopwords and finding most frequesnt and essential words

#Creating Pandas Data Frame

import pandas as pd
profile_df = pd.read_csv('Profile1.csv',names=['Index','Profile'])
profile_df.head()

#Cleaning text data to remove symbols, numbers, parantheses, converting all text to lower case etc.
import re
def clean_data(text):
    text = re.sub('RT@[\w]*:',"",text)
    text = re.sub('@[\w]*',"",text)
    text = re.sub('https?://[A-Za-z0-9./]*',"",text)
    text = re.sub('\n',"",text)
    text = re.sub("\."," ",text) 
    text = re.sub("/"," ",text)
    text = re.sub("[^a-zA-Z]", " ",text) 
    text = text.lower()
    return text

#using clean_data function
profile_df['Profile'] = profile_df['Profile'].apply(lambda x: clean_data(x))

#removing stopwords

import nltk

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
#break the text into individual sentences
sentences = []
for s in profile_df['Profile']:
    sentences.append(sent_tokenize(s))

sentences = [y for x in sentences for y in x]

nltk.download('stopwords')

#remove stopwords from sentences
stop_words = set(stopwords.words("english"))
##Creating a list of custom stopwords
new_words = ["linkedin", "com", "www", "top", "page", "co", "c", "om","using","various","year","month","months","years"]
stop_words = stop_words.union(new_words)

# function to remove stopwords
def remove_stopwords(sen):
    sen_new = " ".join([i for i in sen if i not in stop_words])
    return sen_new
clean_sentences = [remove_stopwords(r.split()) for r in sentences]

#Find frequent words
freq_words =" ".join(clean_sentences)
freq_words = freq_words.split()
freq = nltk.FreqDist(freq_words)
print(freq.most_common(10))

#Create word cloud

from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
%matplotlib inline
wordcloud = WordCloud(
                          background_color='white',
                          stopwords=stop_words,
                          max_words=100,
                          max_font_size=50, 
                          random_state=42
                         ).generate(str(freq_words))
print(wordcloud)
fig = plt.figure(1)
plt.figure(figsize=(20,10))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
fig.savefig("word1.png", dpi=900)

#Using TF IDF to get Keywords
from sklearn.feature_extraction.text import CountVectorizer

#create a vocabulary of words, 
#ignore words that appear in 85% of documents, 
#eliminate stop words
cv=CountVectorizer(max_df=0.85,stop_words=stop_words)
word_count_vector=cv.fit_transform(freq_words)

from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transformer.fit(word_count_vector)

tfidf_transformer.idf_

def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""
    
    #use only topn items from vector
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []

    for idx, score in sorted_items:
        fname = feature_names[idx]
        
        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    #create a tuples of feature,score
    #results = zip(feature_vals,score_vals)
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    
    return results

feature_names=cv.get_feature_names()

# get the document that we want to extract keywords from
doc=freq_words

#generate tf-idf for the given document
tf_idf_vector=tfidf_transformer.transform(cv.transform(freq_words))

#sort the tf-idf vectors by descending order of scores
sorted_items=sort_coo(tf_idf_vector.tocoo())

#extract only the top n; n here is 500
keywords=extract_topn_from_vector(feature_names,sorted_items,500)


for k in keywords:
    print(k,keywords[k])