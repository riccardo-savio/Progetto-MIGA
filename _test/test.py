from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
import nltk

# nltk.download('sentiwordnet')

# # Funzione per ottenere il punteggio di sentiment di una parola
# def get_sentiment_score(word):
#     synsets = wn.synsets(word)
#     if not synsets:
#         return 0
#     synset = synsets[0]  # Prendi il primo synset
#     swn_synset = swn.senti_synset(synset.name())
#     return swn_synset.pos_score() - swn_synset.neg_score()

# # Funzione per filtrare parole con sentiment in una recensione
# def filter_sentiment_words_swn(review):
#     words = word_tokenize(review)
#     sentiment_words = [word for word in words if get_sentiment_score(word) != 0]
#     return ' '.join(sentiment_words)

# # Lista di recensioni (esempio)
# reviews = [
#     "This product is amazing, I love it!",
#     "Terrible experience, very disappointed.",
#     "It's okay, not great but not bad either."
# ]

# # Applicare la funzione alle recensioni
# filtered_reviews_swn = [filter_sentiment_words_swn(review) for review in reviews]

# # Stampa delle recensioni filtrate
# for review in filtered_reviews_swn:
#     print("NLTK approach: ", review)

from afinn import Afinn

# # Inizializza AFINN
# afinn = Afinn()

# # Lista di recensioni (esempio)
# reviews = [
#     "This product is amazing, I love it!",
#     "Terrible experience, very disappointed.",
#     "It's okay, not great but not bad either."
# ]

# # Funzione per filtrare parole con sentiment
# def filter_sentiment_words_afinn(review, afinn):
#     words = review.split()
#     sentiment_words = [word for word in words if afinn.score(word) != 0]
#     return ' '.join(sentiment_words)

# # Applicare la funzione alle recensioni
# filtered_reviews = [filter_sentiment_words_afinn(review, afinn) for review in reviews]

# # Stampa delle recensioni filtrate
# for review in filtered_reviews:
#     print("afinn approach: ", review)

afinn = Afinn()
def filter_sentiment_words_afinn(review):
    words = review.split()
    sentiment_words = [word for word in words if afinn.score(word) != 0]
    return ' '.join(sentiment_words)

import pandas as pd
reviews_df = pd.read_csv('data/final/reviews_advanced.csv')
#keep the first 10 rows
reviews_df = reviews_df.head(10)

from bs4 import BeautifulSoup
reviews_df['text'] = reviews_df['text'].apply(
    lambda x: BeautifulSoup(x, 'html.parser').get_text())

reviews_df['text'] = reviews_df['text'].apply(filter_sentiment_words_afinn)
print(reviews_df['text'])