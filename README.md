# 🎬 IMDB Movie Reviews – Text Preprocessing & NLP Pipeline

## 📘 Overview
This project demonstrates a **complete text preprocessing pipeline** for Natural Language Processing (NLP) using the **IMDB Movie Reviews Dataset**.  
The main objective is to clean and prepare raw textual data for sentiment analysis or any downstream NLP task.

---

## 📂 Dataset
**Dataset used:** [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

**File path in Kaggle:**
import kagglehub
# Download latest version
path = kagglehub.dataset_download("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
print("Path to dataset files:", path)

---

## 🧰 Libraries Used
```python
import numpy as np
import pandas as pd
import os
import re
import string
import time
import nltk
import spacy
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
🧹 Data Cleaning Steps
1️⃣ Load and Inspect Data
df = pd.read_csv('/kaggle/input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv')
df.shape
df.head()


Convert all reviews to lowercase:

df['review'] = df['review'].str.lower()

2️⃣ Remove HTML Tags

Removes unwanted HTML formatting from the reviews.

def remove_html_tags(text):
    pattern = re.compile('<.*?>')
    return pattern.sub('', text)

df['review'] = df['review'].apply(remove_html_tags)

3️⃣ Remove URLs

Cleans out hyperlinks (http, https, or www).

def remove_url(text):
    pattern = re.compile(r'https?://\S+|www\.\S+')
    return pattern.sub('', text)

4️⃣ Remove Punctuation

Two methods were tested for speed and performance:

Method 1: Manual character replacement.

Method 2: Using str.translate() (faster).

exclude = string.punctuation
def remove_punc1(text):
    return text.translate(str.maketrans('', '', exclude))

5️⃣ Chat Word Conversion

Expands common abbreviations (e.g., “FYI” → “For Your Information”).

def chat_conversion(text):
    new_text = []
    for w in text.split():
        if w.upper() in chat_words:
            new_text.append(chat_words[w.upper()])
        else:
            new_text.append(w)
    return " ".join(new_text)

6️⃣ Spelling Correction

Uses TextBlob to automatically fix spelling errors.

from textblob import TextBlob
TextBlob("ceertain conditionas duriing").correct()

7️⃣ Remove Stopwords

Removes common, non-informative words (like is, the, in, of, etc.).

def remove_stopwords(text):
    new_text = []
    for word in text.split():
        if word not in stopwords.words('english'):
            new_text.append(word)
    return " ".join(new_text)

8️⃣ Remove Emojis

Removes emojis using regex or converts them using the emoji library.

def remove_emoji(text):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


Alternate method:

import emoji
emoji.demojize('Loved the movie. It was 😘')

✂️ Tokenization Methods
1️⃣ Split Function

Simple tokenization using Python’s .split().

sent1 = 'I am going to delhi'
sent1.split()

2️⃣ Regular Expressions
re.findall("[\w']+", text)

3️⃣ NLTK Tokenization
from nltk.tokenize import word_tokenize, sent_tokenize
word_tokenize("I am going to delhi!")

4️⃣ SpaCy Tokenization
import spacy
nlp = spacy.load('en_core_web_sm')
doc = nlp("I am going to delhi!")
for token in doc:
    print(token)

🌱 Stemming & Lemmatization
🔹 Porter Stemmer

Reduces words to their root form (e.g., running → run).

ps = PorterStemmer()
def stem_words(text):
    return " ".join([ps.stem(word) for word in text.split()])

🔹 WordNet Lemmatizer

Produces linguistically accurate root words.

wordnet_lemmatizer = WordNetLemmatizer()
wordnet_lemmatizer.lemmatize('running', pos='v')  # → 'run'
