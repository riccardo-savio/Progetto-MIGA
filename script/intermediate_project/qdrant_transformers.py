from qdrant_client import QdrantClient, models
import os, pandas as pd, numpy as np, time
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
import qdrant as qd

COLLECTION_NAME = "amazon_products_transformers"

if __name__ == '__main__':
    # read from csv file if exists
    if os.path.exists('data/_transformers/transformers_embeddings.csv'):
        reviews_df = pd.read_csv('data/final/reviews.csv')
        embeddings_df = pd.read_csv('data/_transformers/transformers_embeddings.csv')
        ids = embeddings_df['parent_asin']

    print(len(embeddings_df.columns)-1)
    qd.create_collection(len(embeddings_df.columns)-1, embeddings_df.drop(columns='parent_asin'), ids, collection_name=COLLECTION_NAME)
    print("Start testing the qdrant model...")
    t = time.time()

    user_id = 'AH4JBZTYR4BHBX4AX5HX4VNJSLIA'
    user_reviews = reviews_df[reviews_df['user_id'] == user_id]

    rated_items = embeddings_df[embeddings_df['parent_asin'].isin(user_reviews['parent_asin'])]
    dataset = pd.merge(rated_items, user_reviews, on='parent_asin')
    parent_asin_user = dataset['parent_asin']
    dataset = dataset.drop(columns=['user_id'])

    # X_train, X_test, y_train, y_test = train_test_split(dataset.drop(columns='rating_y'), dataset['rating_y'], test_size=0.2, random_state=0)
    
    try:
        X, y = dataset.drop(columns='rating'), dataset['rating']

        predictions = []
        for index, row in X.iterrows():
            embedding = row.drop(index='parent_asin').to_list()
            users_item_ids = parent_asin_user.to_list()
            users_item_ids.remove(row['parent_asin'])
            response = qd.search_similar_products(embedding, ids=users_item_ids, top_k=10, collection_name=COLLECTION_NAME)
            ratings = []
            for x in response:
                if x is not None:
                    ratings.append(dataset[dataset['parent_asin'] == x.payload['product_id']]['rating'].values[0])
            predictions.append(np.mean(ratings))

        print("MSE: ", mean_squared_error(y, predictions))
        print("RMSE: ", np.sqrt(mean_squared_error(y, predictions)))
    except:
        print(dataset[dataset['parent_asin'] == x.payload['product_id']]['rating'])
        print("Error qdrant")
    print(f"Time elapsed (qdrant): {time.time()-t} seconds")

    t = time.time()
    mse_knn = []
    for user_id in reviews_df['user_id'].unique():
        user_reviews = reviews_df[reviews_df['user_id'] == user_id]
        rated_items = embeddings_df[embeddings_df['parent_asin'].isin(user_reviews['parent_asin'])]
        dataset = pd.merge(rated_items, user_reviews, on='parent_asin')
        parent_asin_user = dataset['parent_asin']
        dataset = dataset.drop(columns=['user_id'])

        try:
            X_train, X_test, y_train, y_test = train_test_split(dataset.drop(columns=['rating', 'parent_asin']), dataset['rating'], 
                                                                test_size=0.2, random_state=0)

            # Train the regressor using the trainset
            neigh_reg = KNeighborsRegressor(n_neighbors=7, metric="cosine")
            neigh_reg.fit(X_train, y_train)
            # Test the regressor using the testset
            y_pred = neigh_reg.predict(X_test)
            mse_knn.append(mean_squared_error(y_test, y_pred))
        except Exception as e:
            print("Error KNN", e)

    mse = np.mean(mse_knn)
    print(f"Mean Squared Error (KNN): {mse}")
    print(f"Root Mean Squared Error (KNN): {np.sqrt(mse)}")
    print(f"Time elapsed (KNN): {time.time()-t} seconds")