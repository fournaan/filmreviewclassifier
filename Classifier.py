import pandas as pd
from bs4 import BeautifulSoup
import re
import nltk
#nltk.download()
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import pickle

train = pd.read_csv("labeledTrainData.tsv", header = 0, delimiter = '\t', quoting = 3)
train = pd.DataFrame(train)

first = train["review"][0]

def review_to_words(raw):
    # Remove HTML, and return string
    review = BeautifulSoup(raw, features="html.parser").get_text()
    # Remove Punctuation and numbers,
    review = re.sub("[^a-zA-Z]"," ", review)
    # Make Lowercase
    review = review.lower()
    # Split words into seperate stirngs
    review = review.split()
    # Make set of stopwords
    stops = set(stopwords.words("English"))
    # Remove insignificant words (NLTK stopwords)
    review = [i for i in review if i not in stops]
    review = " ".join(review)
    return review

clean_reviews = []
progress = 0

for i in train["review"]:
    if progress % 1000 == 0:
        print(str(int(progress/24000*100)) + "%, " + str(progress) + " reviews generated.")
    clean_reviews.append(review_to_words(i))
    progress += 1

vectorizer = CountVectorizer(analyzer = "word",   
                             tokenizer = None,    
                             preprocessor = None, 
                             stop_words = None,   
                             max_features = 5000) 

train_data_features = vectorizer.fit_transform(clean_reviews).toarray()

vocab = vectorizer.get_feature_names_out()

model = RandomForestClassifier(n_estimators=150)
model.fit(train_data_features, train["sentiment"])

test_data = pd.read_csv("testData.tsv", header = 0, delimiter = '\t', quoting = 3)
test_data = pd.DataFrame(test_data)

clean_test_data = []
predictions = []

for i in test_data["review"]:
    clean_test_data.append(review_to_words(i))

clean_test_data = vectorizer.transform(clean_test_data).toarray()
predictions = model.predict(clean_test_data)

print(predictions)
result = pd.DataFrame( data= {"id": test_data["id"], "sentiment": predictions})
print(result)
result.to_csv("result.csv", index = False, quoting=3)

review_classifier = 'finalized_model.sav'
review_vectorizer = 'finalized_vectorizer.sav'
pickle.dump(model, open('finalized_model.sav', 'wb'))
pickle.dump(vectorizer, open('finalized_vectorizer.sav', 'wb'))
