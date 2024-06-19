from qdrant_client import QdrantClient
import os, pandas as pd, numpy as np, time
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
import qdrant as qd

COLLECTION_NAME = "amazon_products_tfidf"

if os.path.exists('data/_tfidf/tfidf_data.csv'):
    reviews_df = pd.read_csv('data/final/reviews.csv')
    tfidf_data = pd.read_csv('data/_tfidf/tfidf_data.csv')
    ids = tfidf_data['parent_asin']

client = QdrantClient(host="localhost", port=6333)

qd.create_collection(len(tfidf_data.columns)-1,COLLECTION_NAME, tfidf_data.drop(columns='parent_asin'), ids)

print("Start testing the qdrant model...")
t = time.time()

mse = []
for user_id in reviews_df['user_id'].unique():
    user_reviews = reviews_df[reviews_df['user_id'] == user_id]

    rated_items = tfidf_data[tfidf_data['parent_asin'].isin(user_reviews['parent_asin'])]
    dataset = pd.merge(rated_items, user_reviews, on='parent_asin')
    parent_asin_user = dataset['parent_asin']
    dataset = dataset.drop(columns=['user_id'])

    # X_train, X_test, y_train, y_test = train_test_split(dataset.drop(columns='rating_y'), dataset['rating_y'], test_size=0.2, random_state=0)
    
    try:
        X, y = dataset.drop(columns='rating_y'), dataset['rating_y']

        predictions = []
        for index, row in X.iterrows():
            embedding = row.drop(index='parent_asin').to_list()
            users_item_ids = parent_asin_user.to_list()
            users_item_ids.remove(row['parent_asin'])
            response = qd.search_similar_products(embedding, COLLECTION_NAME, ids=users_item_ids, top_k=10)
            ratings = []
            for x in response:
                if x is not None:
                    ratings.append(dataset[dataset['parent_asin'] == x.payload['product_id']]['rating_y'].values[0])
            predictions.append(np.mean(ratings))

        mse.append(mean_squared_error(y, predictions))
    except:
        print(dataset[dataset['parent_asin'] == x.payload['product_id']]['rating_y']) #TODO: fix Series([], Name: rating_y, dtype: float64)
        print("Error qdrant")
        continue
    
print(f"MSE: {np.mean(mse)}")
print(f"RMSE: {np.sqrt(np.mean(mse))}")
print(f"Time elapsed (qdrant): {time.time()-t} seconds")

t = time.time()

mse_knn = []
for user_id in reviews_df['user_id'].unique():
    user_reviews = reviews_df[reviews_df['user_id'] == user_id]

    rated_items = tfidf_data[tfidf_data['parent_asin'].isin(user_reviews['parent_asin'])]
    dataset = pd.merge(rated_items, user_reviews, on='parent_asin')
    parent_asin_user = dataset['parent_asin']
    dataset = dataset.drop(columns=['user_id'])

    try:
        X_train, X_test, y_train, y_test = train_test_split(dataset.drop(columns=['rating_y', 'parent_asin']), dataset['rating_y'], test_size=0.2, random_state=0)

        # Train the regressor using the trainset
        neigh_reg = KNeighborsRegressor(n_neighbors=10, metric="cosine")
        neigh_reg.fit(X_train, y_train)
        # Test the regressor using the testset
        y_pred = neigh_reg.predict(X_test)
        mse_knn.append(mean_squared_error(y_test, y_pred))
    except:
        print("Error KNN")
        continue

print(f"MSE (KNN): {np.mean(mse_knn)}")
print(f"RMSE (KNN): {np.sqrt(np.mean(mse_knn))}")
print(f"Time elapsed (KNN): {time.time()-t} seconds")