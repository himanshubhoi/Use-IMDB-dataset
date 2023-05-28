# Use-IMDB-dataset


#Load the libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud,STOPWORDS
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize,sent_tokenize
from bs4 import BeautifulSoup
import spacy
import re,string,unicodedata
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import LancasterStemmer,WordNetLemmatizer
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from textblob import TextBlob
from textblob import Word
from sklearn.metrics import␣
↪classification_report,confusion_matrix,accuracy_score
import os
import warnings

imdb_data=pd.read_csv('/content/drive/MyDrive/AI/IMDB Dataset.csv')
print(imdb_data.shape)
imdb_data.head(10)


#Summary of the dataset
imdb_data.describe()


 #sentiment count
imdb_data['sentiment'].value_counts()


#split the dataset
#train dataset
train_reviews=imdb_data.review[:40000]
train_sentiments=imdb_data.sentiment[:40000]
#test dataset
test_reviews=imdb_data.review[40000:]
test_sentiments=imdb_data.sentiment[40000:]
print(train_reviews.shape,train_sentiments.shape)
print(test_reviews.shape,test_sentiments.shape)


nltk.download('stopwords')


 #Tokenization of text
tokenizer=ToktokTokenizer()
#Setting English stopwords
stopword_list=nltk.corpus.stopwords.words('english')


#Removing the html strips
def strip_html(text):
soup = BeautifulSoup(text, "html.parser")
return soup.get_text()
#Removing the square brackets
def remove_between_square_brackets(text):
return re.sub('\[[^]]*\]', '', text)
#Removing the noisy text
def denoise_text(text):
text = strip_html(text)
text = remove_between_square_brackets(text)
return text
#Apply function on review column
imdb_data['review']=imdb_data['review'].apply(denoise_text)

#Define function for removing special characters
def remove_special_characters(text, remove_digits=True):
pattern=r'[^a-zA-z0-9\s]'
text=re.sub(pattern,'',text)
return text
#Apply function on review column
imdb_data['review']=imdb_data['review'].apply(remove_special_characters)


#Define function for removing special characters
def remove_special_characters(text, remove_digits=True):
pattern=r'[^a-zA-z0-9\s]'
text=re.sub(pattern,'',text)
return text
#Apply function on review column
imdb_data['review']=imdb_data['review'].apply(remove_special_characters)

