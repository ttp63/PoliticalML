#%%
# Alright, let's try a super simple model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

from nltk.tokenize import RegexpTokenizer


# Remove na values from analysis
simple_df = base_df[base_df["text"].notna()]

# tokenizer to remove unwanted elements from out data like symbols and numbers
token = RegexpTokenizer(r"[a-zA-Z0-9]+")
cv = CountVectorizer(
    lowercase=True, stop_words="english", ngram_range=(1, 1), tokenizer=token.tokenize
)
text_counts = cv.fit_transform(simple_df["text"])

# train test split time
x_train, x_test, y_train, y_test = train_test_split(
    text_counts, simple_df["party"], test_size=0.3, random_state=1
)

# Model Generation Using Multinomial Naive Bayes
nbcl_count = MultinomialNB().fit(x_train, y_train)
nbcl_count_predicted = nbcl_count.predict(x_test)
print(
    "MultinomialNB Count Accuracy:",
    metrics.accuracy_score(y_test, nbcl_count_predicted),
)


#%%
# Not bad, but let's try applying tf-idf
from sklearn.feature_extraction.text import TfidfVectorizer

tf = TfidfVectorizer()
text_tf = tf.fit_transform(simple_df["text"])

x_train, x_test, y_train, y_test = train_test_split(
    text_tf, simple_df["party"], test_size=0.3, random_state=1
)

nbcl_tf = MultinomialNB().fit(x_train, y_train)
nbcl_tf_predicted = nbcl_tf.predict(x_test)
print(
    "MultinomialNB TF-IDF Accuracy:", metrics.accuracy_score(y_test, nbcl_tf_predicted)
)
metrics.confusion_matrix(y_test, nbcl_tf_predicted)
