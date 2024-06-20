import pandas as pd

def get_raw_reviews(
    columns: list = [],
    split: str = "full",
    cache_dir: str = "data/_raw/",
    toDF: bool = True,
) -> pd.DataFrame:
    """
    Fetches and returns a dataset from the Amazon Reviews 2023 dataset.

    Args:
        `columns` (list, optional): List of column names to include in the dataset.
            Defaults to [].
        `split` (str, optional): Split of the dataset to use.
            Defaults to "full".
        `cache_dir` (str, optional): Directory to cache the dataset.
            Defaults to "data/_raw/".
        `toDF` (bool, optional): Flag indicating whether to return the dataset as a pandas DataFrame.
            Defaults to True.

    Returns:
        pd.DataFrame or Dataset: The fetched dataset. If `toDF` is True, returns a pandas DataFrame, otherwise returns a Dataset object.
    """
    import os
    from datasets import load_dataset

    os.makedirs(cache_dir) if not os.path.exists(cache_dir) else None
    dataset = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        "raw_review_Video_Games",
        trust_remote_code=True,
        cache_dir=cache_dir,
        split=split,
    )
    if len(columns) > 0:
        dataset = dataset.select_columns(columns)
    return dataset.to_pandas() if toDF else dataset


def advanced_pre_process():
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    import nltk, contractions, re, os
    from unidecode import unidecode
    from transformers import pipeline
    import time

    reviews_df = pd.read_csv('data/final/reviews_advanced.csv')

    reviews_df.drop_duplicates(inplace=True)
    reviews_df.dropna(inplace=True)

    reviews_df['text'] = reviews_df['text'].astype(str)
    reviews_df['title'] = reviews_df['title'].astype(str)
    reviews_df['rating'] = reviews_df['rating'].astype(int)
    reviews_df['parent_asin'] = reviews_df['parent_asin'].astype(str)
    reviews_df['user_id'] = reviews_df['user_id'].astype(str)

    reviews_df = reviews_df[reviews_df['text'] != '']
    reviews_df = reviews_df[reviews_df['title'] != '']

    # -------------------------- 1. Preprocess text attributes of the items ------------------------------------------
    # convert all the text to lowercase

    # append description to title remove description column and rename title column
    reviews_df['title'] = reviews_df['title'] + reviews_df['text']
    reviews_df = reviews_df.drop(columns=['text'])
    reviews_df = reviews_df.rename(columns={'title': 'title_text'})
    
    reviews_df['title_text'] = reviews_df['title_text'].apply(lambda x: x.lower())

    # substitute all ’ with '
    reviews_df['title_text'] = reviews_df['title_text'].apply(lambda x: x.replace('’', "'"))
 
    # remove all ‘
    reviews_df['title_text'] = reviews_df['title_text'].apply(lambda x: x.replace('‘', ''))

    # expand contractions
    reviews_df['title_text'] = reviews_df['title_text'].apply(lambda x: " ".join([contractions.fix(expanded_word) for expanded_word in x.split()]))

    # replace all not letters or space characters with a space
    reviews_df['title_text'] = reviews_df['title_text'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', ' ', x))
  
    # remove extra spaces
    reviews_df['title_text'] = reviews_df['title_text'].apply(lambda x: re.sub(' +', ' ', x))

    # remove diacritics
    reviews_df['title_text'] = reviews_df['title_text'].apply(lambda x: unidecode(x, errors='preserve'))


    #summarize the text

    print('Summarizing text...', time.time())

    summarizer = pipeline("summarization", model="google-t5/t5-small", tokenizer="google-t5/t5-small")

    def summarize_title_text(text):
        if len(text.split()) < 100:
            return text
        return summarizer(text[:511], max_length=100, min_length=5)[0]['summary_text']

    reviews_df['title_text'] = reviews_df['title_text'].apply(lambda x: summarize_title_text(x))

    print('Summarized text ended', time.time())

    reviews_df['title_text'] = reviews_df['title_text'].apply(lambda x: x.lower())

    # substitute all ’ with '
    reviews_df['title_text'] = reviews_df['title_text'].apply(lambda x: x.replace('’', "'"))
 
    # remove all ‘
    reviews_df['title_text'] = reviews_df['title_text'].apply(lambda x: x.replace('‘', ''))

    # expand contractions
    reviews_df['title_text'] = reviews_df['title_text'].apply(lambda x: " ".join([contractions.fix(expanded_word) for expanded_word in x.split()]))

    # replace all not letters or space characters with a space
    reviews_df['title_text'] = reviews_df['title_text'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', ' ', x))
  
    # remove extra spaces
    reviews_df['title_text'] = reviews_df['title_text'].apply(lambda x: re.sub(' +', ' ', x))

    # remove diacritics
    reviews_df['title_text'] = reviews_df['title_text'].apply(lambda x: unidecode(x, errors='preserve'))

    # -------------------------- 1. Preprocess text attributes of the items ------------------------------------------

    nltk.download('punkt')

    # Tokenize the 'title_text' and 'description' columns with the word_tokenize function
    reviews_df['title_text'] = reviews_df['title_text'].apply(word_tokenize)

    nltk.download('stopwords')

    stop_words = set(stopwords.words('english'))

    # Remove the stopwords from the 'title_text' and 'description' columns
    reviews_df['title_text'] = reviews_df['title_text'].apply(lambda x: [word for word in x if word.lower() not in stop_words])

    # Lemmatizing the words in the 'title_text' and 'description' columns
    lemmatizer = WordNetLemmatizer()
    reviews_df['title_text'] = reviews_df['title_text'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])

    # unify the words in the 'title_text' column in a single string
    reviews_df['title_text'] = reviews_df['title_text'].apply(lambda x: ' '.join(x))

    print(reviews_df.shape)
    #drop all rows where text_tilte column is longer than 1000 characters
    reviews_df = reviews_df[reviews_df['title_text'].apply(lambda x: len(x) < 200)]

    #drop all single or double characters in the text_title column
    def remove_small_words(text):
        return ' '.join([word for word in text.split() if len(word) > 2])
    
    reviews_df['title_text'] = reviews_df['title_text'].apply(remove_small_words)

    print(reviews_df.shape)
    # drop all rows where text_title column is empty
    reviews_df = reviews_df.dropna()
    print(reviews_df.shape)
    
    # save on csv create folder
    os.makedirs('data/_lemmataized', exist_ok=True)
    reviews_df.to_csv('data/_lemmataized/lemmataized_reviews.csv', index=False)
    return reviews_df