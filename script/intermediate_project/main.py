import os
from sklearn.feature_extraction.text import TfidfVectorizer

from intermediate_project.pre_processing import intermediate_pre_process

def intermediate_project(reviews_df, items_df):
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer, WordNetLemmatizer

    processed_items_df = items_df.copy()
    processed_reviews_df = reviews_df.copy()

    processed_items_df, processed_reviews_df = intermediate_pre_process(processed_items_df, processed_reviews_df)

    # # -------------------------- 2. TF-IDF representation of the items ---------------------------------------------------
    
    # Create a TfidfVectorizer object
    vectorizer = TfidfVectorizer()

    # Fit and transform the 'title_description' column
    tfidf_matrix = vectorizer.fit_transform(processed_items_df['title_description'])

    tfidf_data = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

    tfidf_data['parent_asin'] = processed_items_df['parent_asin']

    # save on csv create folder
    os.makedirs('data/_tfidf', exist_ok=True)
    tfidf_data.to_csv('data/_tfidf/tfidf_data.csv', index=False)


if __name__ == '__main__':
    import pandas as pd
    reviews_df = pd.read_csv('data/final/reviews.csv')
    items_df = pd.read_csv('data/final/metadata.csv')
    intermediate_project(reviews_df, items_df)