import os
from sklearn.feature_extraction.text import TfidfVectorizer

from pre_processing import intermediate_pre_process

def intermediate_project(reviews_df, items_df):
    import gensim
    from nltk.tokenize import word_tokenize

    processed_items_df = items_df.copy()
    processed_reviews_df = reviews_df.copy()

    processed_items_df, processed_reviews_df = intermediate_pre_process(processed_items_df, processed_reviews_df)

    # # -------------------------- 2. TF-IDF representation of the items ---------------------------------------------------
    
    """ # Create a TfidfVectorizer object
    vectorizer = TfidfVectorizer()

    # Fit and transform the 'title_description' column
    tfidf_matrix = vectorizer.fit_transform(processed_items_df['title_description'])

    tfidf_data = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

    tfidf_data['parent_asin'] = processed_items_df['parent_asin']

    # save on csv create folder
    os.makedirs('data/_tfidf', exist_ok=True)
    tfidf_data.to_csv('data/_tfidf/tfidf_data.csv', index=False) """

    # -------------------------- 3. Word2Vec representation of the items -------------------------------------------------

    print()

    data = []

    for sent in processed_items_df['title_description'].to_list():
        data.append([])
        # tokenize the sentence into words
        for word in word_tokenize(sent):
            data[-1].append(word.lower())
            
    skipgram = gensim.models.Word2Vec(data, min_count=1, vector_size=20, window=5, sg=1)

    

    ## -------------------------- 4. Perform dimensionality reduction  -----------------------------------------------------  
    #             
    from sklearn.manifold import TSNE
    import numpy as np

    tokens = []
    labels = []
    print(skipgram.wv.index_to_key)
    for word in skipgram.wv.index_to_key:
        tokens.append(skipgram.wv[word])
        labels.append(word)

    tsne_model = TSNE(n_components=2)
    tsne_embeddings = tsne_model.fit_transform(np.array(tokens))


    import matplotlib.pyplot as plt

    for point, label in zip(tsne_embeddings, labels):
        plt.scatter(point[0], point[1])
        plt.annotate(label,
                    xy=point,
                    xytext=(5, 2),
                    textcoords="offset points",
                    ha="right",
                    va="bottom")
        
    plt.show()



if __name__ == '__main__':
    import pandas as pd
    reviews_df = pd.read_csv('data/final/reviews.csv')
    items_df = pd.read_csv('data/final/metadata.csv')
    intermediate_project(reviews_df, items_df)