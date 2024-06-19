from qdrant_client import QdrantClient, models
import os, pandas as pd, numpy as np, time
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor

COLLECTION_NAME = "amazon_products_tfidf"

# read from csv file if exists
if os.path.exists('data/_tfidf/tfidf_data.csv'):
    reviews_df = pd.read_csv('data/final/reviews.csv')
    tfidf_data = pd.read_csv('data/_tfidf/tfidf_data.csv')
    ids = tfidf_data['parent_asin']

client = QdrantClient(host="localhost", port=6333)

def create_collection(vector_size, collection_name=COLLECTION_NAME):
    # if client.collection_exists(collection_name):
    #     client.delete_collection(collection_name)
    if not client.collection_exists(collection_name):
        print("Creating collection...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE)
        )
        tfidf_data_copy = tfidf_data.copy().drop(columns=['parent_asin'])
        insert_embeddings(tfidf_data_copy.to_numpy(), ids)
    
def insert_embeddings(embeddings, product_ids, collection_name=COLLECTION_NAME):
    ids = [i for i in range(len(embeddings))]
    print(f"Inserting {len(embeddings)} embeddings...")
    for i in range(0, len(embeddings), 200):
        chunk_embeddings = embeddings[i:i+200]
        chunk_parent_asin = product_ids[i:i+200]
        chunk_ids = ids[i:i+200]
        print(f"Inserting embeddings {i} to {i+len(chunk_embeddings)}...")
        operation_status = client.upsert(
            collection_name=collection_name,
            wait=True,
            points=[
                models.PointStruct(
                    id = id,
                    vector = embedding,
                    payload={"product_id": str(product_id)}
                )
                for id, product_id, embedding in zip(chunk_ids, chunk_parent_asin, chunk_embeddings)
            ]
        )
    return operation_status.status == models.UpdateStatus.COMPLETED

def search_similar_products(query_embedding, top_k=5, ids=[], collection_name=COLLECTION_NAME):
    if ids != []:
        response = client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=top_k,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="product_id",
                        match=models.MatchAny(any=ids),
                    )
                ]
            )
        )
    else:
        response = client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=top_k
        )
    return response

if __name__ == '__main__':
    if 'tfidf_data' in locals():
        create_collection(len(tfidf_data.columns)-1)
        
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
                    response = search_similar_products(embedding, ids=users_item_ids, top_k=10)
                    ratings = []
                    for x in response:
                        if x is not None:
                            ratings.append(dataset[dataset['parent_asin'] == x.payload['product_id']]['rating_y'].values[0])
                    predictions.append(np.mean(ratings))

                mse.append(mean_squared_error(y, predictions))
            except:
                print(dataset[dataset['parent_asin'] == x.payload['product_id']]['rating_y'])
                print("Error qdrant")
                continue
            
        print(f"Mean Squared Error: {np.mean(mse)}")
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

        print(f"Mean Squared Error (KNN): {np.mean(mse_knn)}")
        print(f"Time elapsed (KNN): {time.time()-t} seconds")

    else:
        print("TF-IDF data not found. Please run the preprocessing script first.")
