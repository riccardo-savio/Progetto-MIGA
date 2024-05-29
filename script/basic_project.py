def basic_project(rating_matrix):
    from surprise import Dataset, Reader, SVD, KNNBasic, model_selection
    from surprise.model_selection import cross_validate
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.cluster import KMeans

    # pandas to surprise dataset
    reader = Reader(rating_scale=(1, 5))
    dataset = Dataset.load_from_df(rating_matrix[['user_id', 'parent_asin', 'rating']], reader)

    # 2. Find the best configuration for KNN ---------------------------------

    param_grid = {
        'k': list(range(10, 45, 5)),
        'sim_options': {
            'name': ['cosine', 'msd', 'pearson'],
            'user_based': [True, False],
        },
    }
    # Initialize and train the Grid Search
    gs = model_selection.GridSearchCV(KNNBasic, param_grid,
                                    measures=["rmse", "mse"],
                                    cv=5,
                                    n_jobs=-1)
    gs.fit(dataset)

    print(f'Best RMSE = {gs.best_score["rmse"]:.4f}')
    print(f'Best configuration = {gs.best_params["rmse"]}')

    # 3. Fill the rating matrix with the best configuration ------------------

    users_id = rating_matrix["user_id"].unique()
    items_id = rating_matrix["parent_asin"].unique()

    try:
        raise FileNotFoundError
        filled_rating_matrix = pd.read_csv('data/filled_rating_matrix.csv', index_col=0)
        print('Filled rating matrix loaded from file')
    except FileNotFoundError:
        print('Filled rating matrix not found. Filling it now...')
        # Build the full trainset and fit the model

        trainset = dataset.build_full_trainset()
        
        algo = gs.best_estimator['rmse']
        algo.fit(trainset)

        filled_rating_matrix = []
        for uid in users_id:
            filled_rating_matrix.append([])
            for iid in items_id:
                res = algo.predict(uid=uid, iid=iid)
                if res.r_ui is not None:
                    # If the user rated the item, the score is 0.
                    # I don't want to recommend an item already seen.
                    filled_rating_matrix[-1].append(0)
                else:
                    filled_rating_matrix[-1].append(res.est)

        filled_rating_matrix = np.array(filled_rating_matrix)

        # save the filled rating matrix
        filled_rating_matrix = pd.DataFrame(filled_rating_matrix, index=users_id, columns=items_id)
        filled_rating_matrix.to_csv('data/filled_rating_matrix.csv')

    # 4. User segmentation based on preferences ------------------------------ 
    # TODO: change the number of clusters with the number of items categories
    
    # Perform user segmentation based on preferences
    kmeans = KMeans(n_clusters=3, random_state=0)
    user_clusters = kmeans.fit_predict(filled_rating_matrix)

    # Add user_id to the cluster
    user_clusters = pd.DataFrame(user_clusters, index=users_id, columns=['cluster'])
    
    # Print the number of users in each cluster
    print(user_clusters['cluster'].value_counts())

    # 5. Create the recommendation list for each user -------------------------

    res_df = pd.DataFrame(filled_rating_matrix)
    res_df.columns = items_id
    res_df = res_df.set_index(users_id)

    # Sort each row by the score and take the top 5
    def sort_columns(row):
        sorted_columns = sorted(row.items(), key=lambda x: x[1], reverse=True)
        return [col[0] for col in sorted_columns[:5]]

    rec_lists = pd.DataFrame(list(res_df.apply(sort_columns, axis=1)),
                            index=res_df.index)
    
    print(rec_lists.head()) # TODO: Volendo si può associare ciascun parent_asin al suo titolo

    # 6.1 Find the best configuration for Matrix Factorization ----------------
    
    param_grid = {
        'n_factors': list(range(20, 160, 20)),
        'n_epochs': list(range(10, 50, 10)),
        'biased': [True, False]
    }

    # Initialize and train the Grid Search
    gs = model_selection.GridSearchCV(SVD, param_grid,
                                    measures=["rmse", "mse"],
                                    cv=5,
                                    n_jobs=-1)
    gs.fit(dataset)

    print(f'Best RMSE = {gs.best_score["rmse"]:.4f}')
    print(f'Best configuration = {gs.best_params["rmse"]}')

    # 6.2 Fill the rating matrix with Matrix Factorization --------------------

    try:
        raise FileNotFoundError
        filled_rating_matrix_SVD = pd.read_csv('data/filled_rating_matrix_SVD.csv', index_col=0)
        print('Filled SVD rating matrix loaded from file')
    except FileNotFoundError:
        print('Filled SVD rating matrix not found. Filling it now...')
        # Build the full trainset and fit the model
        trainset = dataset.build_full_trainset()
        
        algo = gs.best_estimator['rmse']
        algo.fit(trainset)

        filled_rating_matrix_SVD = []
        for uid in users_id:
            filled_rating_matrix_SVD.append([])
            for iid in items_id:
                res = algo.predict(uid=uid, iid=iid)
                if res.r_ui is not None:
                    # If the user rated the item, the score is 0.
                    # I don't want to recommend an item already seen.
                    filled_rating_matrix_SVD[-1].append(0)
                else:
                    filled_rating_matrix_SVD[-1].append(res.est)

        filled_rating_matrix_SVD = np.array(filled_rating_matrix_SVD)

        # save the filled rating matrix
        filled_rating_matrix_SVD = pd.DataFrame(filled_rating_matrix_SVD, index=users_id, columns=items_id)
        filled_rating_matrix_SVD.to_csv('data/filled_rating_matrix_SVD.csv')

    # 6.3 Create the recommendation list for each user -------------------------

    res_df = pd.DataFrame(filled_rating_matrix_SVD)
    res_df.columns = items_id
    res_df = res_df.set_index(users_id)

    # Sort each row by the score and take the top 5
    rec_lists_SVD = pd.DataFrame(list(res_df.apply(sort_columns, axis=1)),
                            index=res_df.index)
    
    # 6.4 Compare the two recommendation accuracy -----------------------------



if __name__ == '__main__':
    import pandas as pd
    reviews_df = pd.read_csv('data/final/reviews.csv') # TODO: change the path
    basic_project(reviews_df)