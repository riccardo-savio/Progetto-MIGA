
def main():

    from pre_processing import advanced_pre_process
    from sklearn.feature_extraction.text import CountVectorizer
    import pandas as pd

    review_df = advanced_pre_process()

    vectorizer = CountVectorizer()
    bow_model = vectorizer.fit_transform(review_df["title_text"])
    bow_dataset = pd.DataFrame(bow_model.toarray(), columns=vectorizer.get_feature_names_out())
    bow_dataset["parent_asin"] = review_df["parent_asin"]








if __name__ == '__main__':
    main()