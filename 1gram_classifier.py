# Let's build a simple AF 1-gram classifier to get a performance baseline!
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, KFold

df = pd.read_csv('AITA_clean.csv')

# Make a composite text field of the title and body of each post
df['text'] = df["title"] + df["body"].fillna("")

# Prepare k-splits
cv = 5
kf = KFold(n_splits = cv, shuffle = True, random_state = 2)
split_gen = kf.split(df)

# Set up some sklearn objects that are going to be in the pipeline
# SMOTE for class balancing via oversampling the minority class
smt = SMOTE(random_state = 12) 
# TF-IDF Vectorizer
tfidf = TfidfVectorizer(sublinear_tf=True, 
            max_features = 10000,
            min_df=5, norm='l2', 
            encoding='latin-1', 
            ngram_range=(1,1), 
            stop_words='english')
# Multinomial naive bayes classifier
nb = MultinomialNB()

results = []
for i in range(0,cv):
    print("Now processing fold " + str(i+1))
    result = next(split_gen)
    train = df.iloc[result[0]]
    test = df.iloc[result[1]]
    # TF-IDF on training set
    features = tfidf.fit_transform(train.text).toarray()
    labels = train.is_asshole
    # TF-IDF on test set
    X_test = tfidf.transform(test.text).toarray()
    y_test = test.is_asshole
    # Resample with smote
    X_train, y_train = smt.fit_resample(features, labels)
    # Fit the model
    nb.fit(X_train, y_train)
    yhat_score = nb.score(X_test, y_test)
    results.append(yhat_score)
    
print(results)
#tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1,1), stop_words='english')
#features = tfidf.fit_transform(df.text).toarray()
#labels = df.is_asshole
#features.shape


# First split test and training dataset
#smt = SMOTE(random_state=12)
#nb = MultinomialNB()

#pipeline = Pipeline([('smt',smt), ('nb',nb)])

#X_train, X_test, y_train, y_test = train_test_split(features, labels, random_state = 12, stratify=labels)
#pipeline.fit(X_train, y_train)

#pipeline = make_pipeline(SMOTE(random_state=12),
#                            MultinomialNB())
#cross_val_score(pipeline, features, labels, scoring="accuracy", cv=5)
