import pandas as pd

df = pd.read_csv('../poetry_data_prepared.csv')

def search(request):
	return df.to_dict(orient='records')[0]

