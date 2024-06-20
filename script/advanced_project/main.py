import pandas as pd


def bow_dataset(review_df: pd.DataFrame):
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import StandardScaler
    from sklearn.feature_extraction.text import CountVectorizer

    vectorizer = CountVectorizer()
    bow_model = vectorizer.fit_transform(review_df["title_text"])
    bow_dataset = pd.DataFrame(bow_model.toarray(), columns=vectorizer.get_feature_names_out())
    bow_dataset["parent_asin"] = review_df["parent_asin"]
    
    print(bow_dataset.shape)

    sentiment = {1: -1, 2: -1, 3: 0, 4: 1, 5: 1}
    review_df["sentiment"] = review_df["rating"].map(sentiment)
    df = review_df[["parent_asin", "sentiment"]]

    
    print("Scaling the data...")
    scalar = StandardScaler() 
    bow_dataset = pd.DataFrame(scalar.fit_transform(bow_dataset)) #scaling the data

    print("Applying PCA...")
    pca = PCA(n_components = 400)
    pca.fit(bow_dataset)
    bow_dataset = pca.transform(bow_dataset)
    bow_dataset = pd.DataFrame(bow_dataset,columns=['PC1','PC2','PC3'])
    bow_dataset['sentiment'] = df['sentiment']

    print("Saving the dataset...")
    bow_dataset.to_csv("data/_processed/bow_dataset.csv", index=False)
    return bow_dataset


    



def main():
    from pre_processing import advanced_pre_process
    review_df = advanced_pre_process()
    exit()
    review_df = review_df.fillna('')
    bow_dataset(review_df)
if __name__ == '__main__':
    main()