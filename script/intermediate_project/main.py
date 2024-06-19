import os
from sklearn.feature_extraction.text import TfidfVectorizer

from pre_processing import intermediate_pre_process

def intermediate_project(reviews_df, items_df):
    import gensim
    from nltk.tokenize import word_tokenize

    processed_reviews_df = reviews_df.copy()

    processed_items_df = intermediate_pre_process(items_df.copy())

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

    # -------------------------- 3. Word2Vec representation of the items -------------------------------------------------

    """ from sentence_transformers import SentenceTransformer

    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    data = []

    for sent in processed_items_df['title_description'].to_list():
        data.append(sent)
    
    embeddings = model.encode(data)

    #create a dataframe with the embeddings and parent_asin
    embeddings_df = pd.DataFrame(embeddings)
    embeddings_df['parent_asin'] = processed_items_df['parent_asin']

    #write the embeddings to a csv file
    os.makedirs('data/_transformers', exist_ok=True)
    embeddings_df.to_csv('data/_transformers/transformers_embeddings.csv', index=False) """


    # print()



    # skipgram = gensim.models.Word2Vec(data, min_count=1, vector_size=20, window=5, sg=1)

    

    # ## -------------------------- 4. Perform dimensionality reduction  -----------------------------------------------------  
    # #             
    # from sklearn.manifold import TSNE
    # import numpy as np

    # tokens = []
    # labels = []
    # print(skipgram.wv.index_to_key)
    # for word in skipgram.wv.index_to_key:
    #     tokens.append(skipgram.wv[word])
    #     labels.append(word)

    # tsne_model = TSNE(n_components=3)
    # tsne_embeddings = tsne_model.fit_transform(np.array(tokens))



if __name__ == '__main__':
    import pandas as pd
    reviews_df = pd.read_csv('data/final/reviews.csv')
    items_df = pd.read_csv('data/final/metadata.csv')
    intermediate_project(reviews_df, items_df)