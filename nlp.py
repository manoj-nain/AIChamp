#Using nlp to remove stopwords and finding most frequesnt and essential words

#Creating Pandas Data Frame

import pandas as pd
profile_df = pd.read_csv('Profile1.csv',names=['Index','Profile'])
profile_df.head()

#cleaning text data to remove symbols, numbers, parantheses, converting all text to lower case etc.
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
stop_words = stopwords.words('english')

# function to remove stopwords
def remove_stopwords(sen):
    sen_new = " ".join([i for i in sen if i not in stop_words])
    return sen_new
clean_sentences = [remove_stopwords(r.split()) for r in sentences]