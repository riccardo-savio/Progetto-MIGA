import nltk, gensim

def intermediate_project(reviews_df, items_df):
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer, WordNetLemmatizer

    processed_items_df = items_df.copy()

    # -------------------------- 1. Preprocess text attributes of the items ------------------------------------------
    nltk.download('punkt')

    # Tokenize the 'title' and 'description' columns with the word_tokenize function
    processed_items_df['title'] = processed_items_df['title'].apply(word_tokenize)
    processed_items_df['description'] = processed_items_df['description'].apply(word_tokenize)

    nltk.download('stopwords')

    stop_words = set(stopwords.words('english'))

    # Remove the stopwords from the 'title' and 'description' columns
    processed_items_df['title'] = processed_items_df['title'].apply(lambda x: [word for word in x if word.lower() not in stop_words])
    processed_items_df['description'] = processed_items_df['description'].apply(lambda x: [word for word in x if word.lower() not in stop_words])

    # Stem the words in the 'title' and 'description' columns #TODO: Stemming or Lemmatization?
    # stemmer = PorterStemmer()
    # processed_items_df['title'] = processed_items_df['title'].apply(lambda x: [stemmer.stem(word) for word in x])
    # processed_items_df['description'] = processed_items_df['description'].apply(lambda x: [stemmer.stem(word) for word in x])
    lemmatizer = WordNetLemmatizer()
    processed_items_df['title'] = processed_items_df['title'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])
    processed_items_df['description'] = processed_items_df['description'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])

    # -------------------------- 2. BoW representation of the items ---------------------------------------------------
    vocab = set()

    bow_model = []

    #XXX: Maybe we should join the title and description



if __name__ == '__main__':
    import pandas as pd
    reviews_df = pd.read_csv('data/final/reviews.csv')
    items_df = pd.read_csv('data/final/metadata.csv')
    intermediate_project(reviews_df, items_df)