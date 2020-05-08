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
import numpy as np

# Numeric Date
base_df["num_days"] = (base_df["date"] - base_df["date"].min()) / np.timedelta64(1, "D")

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

# Replace missing null values for dirty vs clean demonstration
ml_df["text"] = ml_df["text"].fillna(value=".")

#%% SPACY SETUP

# Let's try spacy
import en_core_web_sm
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English

nlp = en_core_web_sm.load()

# Punctuation marks
punctuations = string.punctuation

# Load English tokenizer, tagger, parser, NER and word vectors
parser = English()

# Creating our tokenizer function
def spacy_tokenizer1(sentence):
    # Creating our token object
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


# Basic function to clean the text
def clean_text(text):
    return text.strip().lower()


# Create spacy vectorizers
spacy_cv1 = CountVectorizer(tokenizer=spacy_tokenizer1, ngram_range=(1, 1))

#%% LETS FIT SOME MODELS
test_classifier = MultinomialNB()

# Make train test splits
x = ml_df["clean_text"]
y = ml_df["party"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

# Create pipeline
pipe = Pipeline(
    [
        ("cleaner", predictors()),
        ("vectorizer", spacy_cv1),
        ("classifier", test_classifier),
    ]
)

# model generation
pipe.fit(x_train, y_train)

# Predicting with a test dataset
predicted = pipe.predict(x_test)

# Model Accuracy
print("Accuracy:", metrics.accuracy_score(y_test, predicted))

#%% DIFFERENT TOKENIZERS
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import RegexpTokenizer

nltk_tk_tweet = TweetTokenizer().tokenize
nltk_tk_regexp = RegexpTokenizer(r"[a-zA-Z0-9]+").tokenize

custom_punctuation2 = punctuations.replace("@", "").replace("!", "").replace("#", "")


# text_counts = spacy_cv1.fit_transform(ml_df["clean_text"])
# feature_names = spacy_cv1.get_feature_names()

custom_nlp = en_core_web_sm.load()


def spacy_tokenizer2(sentence):
    # Creating our token object, which is used to create documents with linguistic annotations.
    spacy_nlp = custom_nlp(sentence)

    # Lemmatizing each token and converting each token into lowercase
    mytokens = [
        word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_
        for word in spacy_nlp
    ]

    # Removing stop words
    mytokens = [
        word for word in mytokens if word not in STOP_WORDS and word not in punctuations
    ]

    # return preprocessed list of tokens
    return mytokens


sentence = "Donald Trump visited Apple headquarters THIS IS a smaple as;odfi n soid lkloiw fj;aklsjfd -sdiof @cnn #jfs apple.com www.foxnews.com"
sentence2 = "THis is a test sentance to see how SPACY works White Collar espn.com"

# %% MODEL TESTING
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier

from sklearn.model_selection import train_test_split
from sklearn import metrics

import time

ngrams = [(1, 1), (1, 2)]
tokenizer_list = [
    ["NLTK", nltk_tk_regexp],
    ["NLTK_tweet", nltk_tk_tweet],
    ["SPACY", spacy_tokenizer1],
]
model_list = [LogisticRegression()]
rstates = [i for i in range(16, 21)]
x_data = [["Clean", ml_df["clean_text"]], ["Dirty", ml_df["text"]]]
vector_methods = ["Count", "TFIDF"]

results_name = "logdetail.csv"
results_list = []

total_len = (
    len(ngrams)
    * len(tokenizer_list)
    * len(model_list)
    * len(rstates)
    * len(x_data)
    * len(vector_methods)
)
y = ml_df["party"]
tsize = 0.3
counter = 0

t0 = time.time()

for dataset in x_data:
    x = dataset[1]
    for t in tokenizer_list:
        tokenizer = t[1]
        for v in vector_methods:
            vector_type = v  # Specify count, if not count defaults to TFIDF
            for ng in ngrams:
                ngram = ng

                # Specify vectorizer
                if vector_type == "Count":
                    vect = CountVectorizer(
                        tokenizer=tokenizer, lowercase=True, ngram_range=ngram
                    )
                else:
                    vect = TfidfVectorizer(tokenizer=tokenizer, ngram_range=ngram)

                text_vect = vect.fit_transform(x)

                for r in rstates:
                    rstate = r

                    # Make new train test splits
                    x_train, x_test, y_train, y_test = train_test_split(
                        text_vect, y, test_size=tsize, random_state=rstate
                    )
                    for c in model_list:
                        classifier = c

                        model = classifier.fit(x_train, y_train)
                        prediction = model.predict(x_test)

                        null_prediction = y_test.replace(
                            to_replace=ml_df["party"]
                            .value_counts()
                            .sort_values()
                            .index[0],
                            value=ml_df["party"].value_counts().sort_values().index[1],
                        )

                        accuracy = metrics.accuracy_score(y_test, prediction)
                        dem_precision = metrics.precision_score(
                            y_test, prediction, pos_label="Democrat/Ind"
                        )
                        rep_precision = metrics.precision_score(
                            y_test, prediction, pos_label="Republican"
                        )
                        dem_recall = metrics.recall_score(
                            y_test, prediction, pos_label="Democrat/Ind"
                        )
                        rep_recall = metrics.recall_score(
                            y_test, prediction, pos_label="Republican"
                        )
                        null_accuracy = metrics.accuracy_score(y_test, null_prediction)
                        lift = accuracy / null_accuracy

                        results_list.append(
                            [
                                str(classifier).split("(")[0],
                                t[0],
                                vector_type,
                                str(ngram),
                                rstate,
                                dataset[0],
                                accuracy,
                                dem_precision,
                                rep_precision,
                                dem_recall,
                                rep_recall,
                                null_accuracy,
                                lift,
                            ]
                        )
                        counter = counter + 1
                        print(
                            "Completed "
                            + str(counter)
                            + "/"
                            + str(total_len)
                            + " models."
                        )
t2 = time.time()
print("Completed all models in " + str(round((t2 - t0), 2)) + " seconds.")

results_df = pd.DataFrame(
    results_list,
    columns=[
        "model",
        "tokenizer",
        "vectorizer",
        "ngram",
        "random_state",
        "data_type",
        "accuracy",
        "dem_precision",
        "rep_precision",
        "dem_recall",
        "rep_recall",
        "null_accuracy",
        "lift",
    ],
)

results_df.to_csv("../Results/" + results_name)

#%% SVD and Random Forest
from sklearn.decomposition import TruncatedSVD

n_comps = 50

svd = TruncatedSVD(n_components=n_comps)
components = svd.fit_transform(text_vect)
print("Explained variance from SVD:" + str(sum(svd.explained_variance_ratio_)))

components_df = pd.DataFrame(components).reset_index(drop=True)
components_df.columns = ["SVD" + str(i + 1) for i in range(n_comps)]

# Meta-data from initial analysis
md_df = ml_df[
    [
        "num_days",
        "all_caps",
        "cap_words",
        "sentiment_compound",
        "sentiment_pos",
        "sentiment_neu",
        "sentiment_neg",
    ]
].reset_index(drop=True)
for n in list(md_df.columns):
    components_df[n] = md_df[n]

# Specify post-SVD parameters
svd_classifier = RandomForestClassifier()
svd_tsize = 0.3
svd_rstate = 1
svd_x = components_df
svd_y = ml_df["party"]


svd_t0 = time.time()

# Make new train test splits
svd_x_train, svd_x_test, svd_y_train, svd_y_test = train_test_split(
    svd_x, svd_y, test_size=svd_tsize, random_state=svd_rstate
)

svd_model = svd_classifier.fit(svd_x_train, svd_y_train)
svd_prediction = svd_model.predict(svd_x_test)

svd_null_prediction = svd_y_test.replace(
    to_replace=svd_y.value_counts().sort_values().index[0],
    value=svd_y.value_counts().sort_values().index[1],
)

print(
    str(svd_classifier).split("(")[0] + " SVD Accuracy:",
    metrics.accuracy_score(svd_y_test, svd_prediction),
)
print(
    "Null Accuracy:", metrics.accuracy_score(svd_y_test, svd_null_prediction),
)
svd_t1 = time.time()
print("Model Time:" + str(t1 - t0))


# %% VISUALIZATION TIME

class_comp_df = pd.read_csv("../Results/classifier_comp.csv")

class_sum_df = (
    class_comp_df.groupby(["model", "tokenizer", "vectorizer", "ngram", "data_type"])
    .agg(
        {
            "accuracy": "mean",
            "null_accuracy": "mean",
            "lift": "mean",
            "dem_precision": "mean",
            "rep_precision": "mean",
            "dem_recall": "mean",
            "rep_recall": "mean",
        }
    )
    .reset_index()
)

nbvlog_df = pd.read_csv("../Results/nb_v_logistic.csv")

nbvlog_sum_df = (
    nbvlog_df.groupby(["model", "tokenizer", "vectorizer", "ngram", "data_type"])
    .agg(
        {
            "accuracy": "mean",
            "null_accuracy": "mean",
            "lift": "mean",
            "dem_precision": "mean",
            "rep_precision": "mean",
            "dem_recall": "mean",
            "rep_recall": "mean",
        }
    )
    .reset_index()
)

logdetail_df = pd.read_csv("../Results/logdetail.csv")

logdetail_sum_df = (
    logdetail_df.groupby(["model", "tokenizer", "vectorizer", "ngram", "data_type"])
    .agg(
        {
            "accuracy": "mean",
            "null_accuracy": "mean",
            "lift": "mean",
            "dem_precision": "mean",
            "rep_precision": "mean",
            "dem_recall": "mean",
            "rep_recall": "mean",
        }
    )
    .reset_index()
)

logdetail_sum_df["model_num"] = range(1, len(logdetail_sum_df) + 1)
logdetail_sum_df["model_name"] = (
    logdetail_sum_df["model"].astype(str)
    + "_"
    + logdetail_sum_df["model_num"].astype(str)
)

# %%
import matplotlib.pyplot as plt
import numpy as np

# Classifier Comparison
classax = class_sum_df.sort_values(by="lift").plot.barh(
    x="model",
    y="lift",
    title="Basic Classifier Comparison",
    xlim=(1, 1.6),
    legend=False,
)
classax.set_ylabel("Classifier Type")
classax.set_xlabel("Lift")

plt.savefig("../Figures/class_comp.png", bbox_inches="tight")
# All Logistic Models
logdetailax = logdetail_sum_df.sort_values(by="lift").plot.barh(
    x="model_num",
    y="lift",
    title="Logistic Model Parameter Tuning",
    xlim=(1, 1.6),
    legend=False,
)
logdetailax.set_ylabel("Model Number")
logdetailax.set_xlabel("Lift")

plt.savefig("../Figures/logdetail_comp.png", bbox_inches="tight")


#%% BEST MODEL

tokenizer_list = [["NLTK_tweet", nltk_tk_tweet]]
rstates = [i for i in range(1, 20)]
x_data = [["Clean", ml_df["clean_text"]]]
y = ml_df["party"]
tsize = 0.3
best_results = []
x = dataset[1]
tokenizer = t[1]
vector_type = "Count"  # Specify count, if not count defaults to TFIDF
ngram = (1, 2)
classifier = LogisticRegression()

t0 = time.time()

# Specify vectorizer
if vector_type == "Count":
    vect = CountVectorizer(tokenizer=tokenizer, lowercase=True, ngram_range=ngram)
else:
    vect = TfidfVectorizer(tokenizer=tokenizer, ngram_range=ngram)

text_vect = vect.fit_transform(x)

for r in rstates:
    rstate = r

    # Make new train test splits
    x_train, x_test, y_train, y_test = train_test_split(
        text_vect, y, test_size=tsize, random_state=rstate
    )

    model = classifier.fit(x_train, y_train)
    prediction = model.predict(x_test)

    null_prediction = y_test.replace(
        to_replace=ml_df["party"].value_counts().sort_values().index[0],
        value=ml_df["party"].value_counts().sort_values().index[1],
    )

    accuracy = metrics.accuracy_score(y_test, prediction)
    dem_precision = metrics.precision_score(
        y_test, prediction, pos_label="Democrat/Ind"
    )
    rep_precision = metrics.precision_score(y_test, prediction, pos_label="Republican")
    dem_recall = metrics.recall_score(y_test, prediction, pos_label="Democrat/Ind")
    rep_recall = metrics.recall_score(y_test, prediction, pos_label="Republican")
    null_accuracy = metrics.accuracy_score(y_test, null_prediction)
    lift = accuracy / null_accuracy

    best_results.append(
        [
            str(classifier).split("(")[0],
            t[0],
            vector_type,
            str(ngram),
            rstate,
            dataset[0],
            accuracy,
            dem_precision,
            rep_precision,
            dem_recall,
            rep_recall,
            null_accuracy,
            lift,
        ]
    )

t2 = time.time()
print("Completed all models in " + str(round((t2 - t0), 2)) + " seconds.")

best_results_df = pd.DataFrame(
    results_list,
    columns=[
        "model",
        "tokenizer",
        "vectorizer",
        "ngram",
        "random_state",
        "data_type",
        "accuracy",
        "dem_precision",
        "rep_precision",
        "dem_recall",
        "rep_recall",
        "null_accuracy",
        "lift",
    ],
)

# ELI5
import eli5

# Train best model on all data
best_vect = CountVectorizer(tokenizer=nltk_tk_tweet, lowercase=True, ngram_range=(1, 2))
best_text_vect = best_vect.fit_transform(ml_df["clean_text"])
best_model = LogisticRegression().fit(best_text_vect, ml_df["party"])

eli5.explain_weights(best_model, vec=best_vect, top=50)
