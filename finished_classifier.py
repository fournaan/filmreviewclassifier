from bs4 import BeautifulSoup
import re
import pickle
from nltk.corpus import stopwords


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

model = pickle.load(open('finalized_model.sav', 'rb'))
vectorizer = pickle.load(open('finalized_vectorizer.sav', 'rb'))

#review = "(Spoiler-Free Review!) I recently had the pleasure to watch this movie in a theater with my friends and we had a blast. While we're suckers for horror movies, I'll admit that I'm not big with watching every single film in The Conjuring Universe; I've only seen The Conjuring 2, The Nun 1, Annabelle Comes Home, and of course, The Nun 2. So here's what I think: it's totally worth the watch!This sequel is a huge upgrade to its predecessor, and enlightens me to watch the rest of the movies in the trilogy! The Nun 2 exceeded all the expectations I had for this new entry, as I previously thought it would end up either being worse or comparable to the first Nun movie, which I thought was borderline bad and okay. For a horror film, The Nun 2 meets all my/the requirements. It really has it all; despite it's silly storyline and somewhat sloppy build-up, it's action-packed with sick jump scares occurring every 5-10 minutes or so, it's got amazing details, as well as some awesome CGI. I also think Bonnie Aarons plays Valak really well, I couldn't imagine any other actress playing a more terrifying version of her. Some great performances from Reid and Farmiga also sugarcoats the movie's silliness, at least in my opinion.All in all, I think The Nun 2 is an awesome film, and serves its predecessor justice. It's probably the best of The Conjuring films I've seen (can't wait to watch the rest!), as well as the scariest. Again, I am a big sucker for the horror genre, and this movie immediately rose to my #1 fav when the credits began to roll. Totally recommend, 4.5 out of 5 stars."

review = input("Input Review: ")

review = review_to_words(review)
review = vectorizer.transform([review])

prediction = model.predict(review)

sentiment = ""

if prediction == [1]:
    sentiment = "Positive"
else:
    sentiment = "Negative"

print("This review is: " + sentiment)
