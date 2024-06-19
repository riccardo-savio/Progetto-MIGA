import hashlib
import uuid
import pandas as pd

def create_uuid_from_string(val: str):
    import hashlib
    import uuid
    hex_string = hashlib.md5(val.encode("UTF-8")).hexdigest()
    return uuid.UUID(hex=hex_string)

df = pd.read_csv('data\\_tfidf\\tfidf_data.csv')
df = df[df['parent_asin'].notna()]
df.to_csv('data\\_tfidf\\parent_asin.csv', index=False)