#Using nlp to remove stopwords and finding most frequesnt and essential words

#Creating Pandas Data Frame

import pandas as pd
profile_df = pd.read_csv('Profile1.csv',names=['Index','Profile'])
profile_df.head()

##cleaning text data to remove symbols, numbers, parantheses
import re
def clean_data(text):
    text = re.sub('RT@[\w]*:',"",text)
    text = re.sub('@[\w]*',"",text)
    text = re.sub('https?://[A-Za-z0-9./]*',"",text)
    text = re.sub('\n',"",text)
    text = re.sub("\."," ",text) 
    text = re.sub("/"," ",text)
    text = re.sub("[^a-zA-Z]", " ",text) 
    return text

#using clean_data function
profile_df['Profile'] = profile_df['Profile'].apply(lambda x: clean_data(x))