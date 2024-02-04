import pandas as pd

df = pd.read_pickle('D:/PROJECTS/Poetry-recommendation/poetry_data_prepared.pkl')

def search_poems():
    return df.loc[:10, ['name', 'text', 'author', 'date']].to_dict(orient='records')

