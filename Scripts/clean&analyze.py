#%% IMPORT AND CLEAN DATA

import pandas as pd


data_file = "sen.csv"

base_df = pd.read_csv("../Data/" + data_file, parse_dates=["date"])
congress = pd.read_csv("../Data/legislators-current.csv")

# Assume independents (Bernie and Angus King) effectively align with Democrats
congress.loc[(congress.party == "Independent"), "party"] = "Democrat/Ind"
congress.loc[(congress.party == "Democrat"), "party"] = "Democrat/Ind"

# Add outcome (democrat/republican)
base_df = base_df.merge(
    congress[["twitter", "party"]], how="left", left_on="username", right_on="twitter"
)

# Filter unnecessary columns
keep_cols = ["username", "to", "text", "date", "hashtags", "mentions", "urls", "party"]
base_df = base_df[keep_cols]

#%% CLEAN THE DATA AND MAKE USEFUL VARIABLES
import string
from urllib.parse import urlparse
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Remove noisy urls
def remove_all_urls(text, urls):
    if str(urls) == "":
        return str(text).strip()
    elif pd.isnull(text):
        return ""
    else:
        url_list = str(urls).split(sep=",")
        for i in url_list:
            text = str(text).replace(i, "")
        return str(text).strip()


base_df["no_url_text"] = base_df.apply(
    lambda x: remove_all_urls(x["text"], x["urls"]), axis=1
)

# Define function to strip url to base
def clean_urls(urls):
    if str(urls) == "":
        return ""
    else:
        parse_list = list(map(urlparse, str(urls).split(sep=",")))
        url_list = [i[1] for i in parse_list]
        return " ".join(url_list)


# Strip urls
base_df["clean_urls"] = base_df["urls"].apply(clean_urls)

# Recombine base urls with text
base_df["clean_text"] = (
    base_df["no_url_text"].astype(str) + " " + base_df["clean_urls"].astype(str)
)

custom_punctuation1 = string.punctuation.replace("@", "").replace("#", "")

# Define function to count all caps words
def all_caps(text):
    if str(text) == "":
        return 0
    else:
        caps_list = str(text).upper().split()
        count = 0
        for w in list(set(caps_list)):
            if w.strip(custom_punctuation1).isalpha() & (
                len(w.strip(custom_punctuation1)) > 1
            ):
                count = count + (" " + text + " ").count(" " + w + " ")
        return count


base_df["all_caps"] = base_df["clean_text"].apply(all_caps)

# Define function to count capitalized words
def cap_words(text):
    if str(text) == "":
        return 0
    else:
        caps_list = list(map(str.capitalize, str(text).split()))
        count = 0
        for w in list(set(caps_list)):
            if w.strip(custom_punctuation1).isalpha():
                count = count + (" " + text + " ").count(" " + w + " ")
        return count


base_df["cap_words"] = base_df["clean_text"].apply(cap_words)

# Basic Sentiment Score
analyzer = SentimentIntensityAnalyzer()


def compound(text):
    return analyzer.polarity_scores(text)["compound"]


base_df["sentiment_compound"] = base_df["no_url_text"].apply(compound)


def neg(text):
    return analyzer.polarity_scores(text)["neg"]


base_df["sentiment_neg"] = base_df["no_url_text"].apply(neg)


def pos(text):
    return analyzer.polarity_scores(text)["pos"]


base_df["sentiment_pos"] = base_df["no_url_text"].apply(pos)


def neu(text):
    return analyzer.polarity_scores(text)["neu"]


base_df["sentiment_neu"] = base_df["no_url_text"].apply(neu)

# Drop tweets with no text or url data
ml_df = base_df[base_df["text"].notna() | base_df["urls"].notna()]
dropped = base_df[base_df["text"].isna() & base_df["urls"].isna()]

# df for base model
simple_df = base_df[base_df["text"].notna()]

#%% SPACY SETUP

# Let's try spacy
import en_core_web_sm
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
import string
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

nlp = en_core_web_sm.load()

# Punctuation marks
punctuations = string.punctuation
custom_punctuation2 = punctuations.replace("@", "").replace("!", "").replace("#", "")

# Add nan to STOP_WORDS
# STOP_WORDS

# Load English tokenizer, tagger, parser, NER and word vectors
parser = English()

# Creating our tokenizer function
def spacy_tokenizer(sentence):
    # Creating our token object, which is used to create documents with linguistic annotations.
    mytokens = parser(sentence)

    # Lemmatizing each token and converting each token into lowercase
    mytokens = [
        word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_
        for word in mytokens
    ]

    # Removing stop words
    mytokens = [
        word for word in mytokens if word not in STOP_WORDS and word not in punctuations
    ]

    # return preprocessed list of tokens
    return mytokens


# Custom transformer using spaCy
class predictors(TransformerMixin):
    def transform(self, X, **transform_params):
        # Cleaning Text
        return [clean_text(text) for text in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}


# Basic function to clean the text
def clean_text(text):
    return text.strip().lower()


# Create spacy vectorizers
spacy_cv1 = CountVectorizer(tokenizer=spacy_tokenizer, ngram_range=(1, 1))
spacy_tf1 = TfidfVectorizer(tokenizer=spacy_tokenizer, ngram_range=(1, 1))
spacy_cv2 = CountVectorizer(tokenizer=spacy_tokenizer, ngram_range=(1, 2))
spacy_tf2 = TfidfVectorizer(tokenizer=spacy_tokenizer, ngram_range=(1, 2))

#%% LETS FIT SOME MODELS
classifier = RandomForestClassifier()

# Make train test splits
x = ml_df["clean_text"]
y = ml_df["party"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

# Create pipeline
pipe = Pipeline(
    [("cleaner", predictors()), ("vectorizer", spacy_cv1), ("classifier", classifier)]
)

# model generation
pipe.fit(x_train, y_train)

# Predicting with a test dataset
predicted = pipe.predict(x_test)

# Model Accuracy
print("Accuracy:", metrics.accuracy_score(y_test, predicted))

#%% MODELS WITHOUT PIPELINE

text_counts = spacy_cv1.fit_transform(ml_df["clean_text"])
feature_names = spacy_cv1.get_feature_names()


def spacy_tokenizer2(sentence):
    # Creating our token object, which is used to create documents with linguistic annotations.
    mytokens = parser(sentence)

    # Lemmatizing each token and converting each token into lowercase
    mytokens = [
        word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_
        for word in mytokens
    ]

    # Removing stop words
    mytokens = [
        word for word in mytokens if word not in STOP_WORDS and word not in punctuations
    ]

    # return preprocessed list of tokens
    return mytokens


sentence = "Donald Trump visited Apple headquarters THIS IS a smaple as;odfi n soid lkloiw fj;aklsjfd -sdiof @cnn #jfs"

# %%
