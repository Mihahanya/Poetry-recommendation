import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import numpy as np
import torch
from pymorphy2 import MorphAnalyzer
import re
import math


torch.cuda.empty_cache()

df = pd.read_pickle('D:/PROJECTS/Poetry-recommendation/poetry_data_prepared_distiluse.pkl')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#

model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2', device=device)
#model = SentenceTransformer('sentence-transformers/LaBSE', device=device)

def to_embeddings(input_text):
    text_chunks = input_text
    embeddings = model.encode(text_chunks)
    return embeddings

#

qa_model = pipeline("question-answering", "timpal0l/mdeberta-v3-base-squad2")

def get_normal_sentence(text):
	morph = MorphAnalyzer()
	words = re.findall(r'\b\w+\b', text.lower())
	lemmatized_words = [morph.parse(word)[0].normal_form for word in words]
	return ' '.join(lemmatized_words)

def extract_metadata(request):
	r = qa_model(question=[
		# 'упомянутое количество строк или четверостиший или слов', 
		# 'упомянутое авторство', 
		# 'упомянутая эпоха или упомянутый век или год'
		'Какой объем?', 
		'Кто авторы или автор или упомянут?', 
		'Какой век или год или эпоха?'
		], context=get_normal_sentence(request)) 
	
	rs = ['length', 'author', 'date']

	return {rs[i]: r[i] for i in range(len(rs))}


def to_standard(data):
	r = ''
	if data['length']['score'] > 0.01: r += f'объем: {get_normal_sentence(data['length']['answer'])}; '
	if data['author']['score'] > 0.01: r += f'автор: {get_normal_sentence(data['author']['answer'])}; '
	if data['date']['score'] > 0.01: r += f'временной период: {get_normal_sentence(data['date']['answer'])}'
	return r

#

def cosine_similarity(a, b):
	return float(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b))

def best_similarity(req_emb, text_embs):
	return max(cosine_similarity(req_emb, text_emb) for text_emb in text_embs)


def search_poems(request, top_n=10, search_priority=0.5):
	mentioned_parameters = extract_metadata(request)
	standard_params_text = to_standard(mentioned_parameters) + ' '
	req_params_emb = to_embeddings(standard_params_text)
	if len(standard_params_text) < 14: search_priority = 1
	
	req_emb = to_embeddings(request)
	texts_datas = []

	for index, row in df.iterrows():
		text_sim = best_similarity(req_emb, row['text_embedding'])
		req_sim = cosine_similarity(req_params_emb, row['metadata_embedding'])
		
		score = text_sim * search_priority + req_sim * (1-search_priority)
		
		texts_datas.append({'score': score, 'text_sim': text_sim, 'req_sim': req_sim, 
			**{k: v for k, v in row.items() if k != 'metadata_embedding' and k != 'text_embedding'}})
	
	texts_datas.sort(key=lambda x: x['score'])

	return texts_datas[-top_n:][::-1]

#def search_poems():
#    return df.loc[:10, ['name', 'text', 'author', 'date']].to_dict(orient='records')

