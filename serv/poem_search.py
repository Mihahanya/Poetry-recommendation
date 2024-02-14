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
    tokens_count = model.tokenize(input_text)['input_ids'].shape[0]
    chunks_n = math.ceil(tokens_count / model.max_seq_length)

    step = math.ceil(len(input_text) / chunks_n)

    text_chunks = [input_text[i:i+step] for i in range(0, len(input_text), step)]
    
    embeddings = model.encode(text_chunks)
    return embeddings

#

qa_model = pipeline("question-answering", "timpal0l/mdeberta-v3-base-squad2")

def get_normal_sentence(text, norm=True):
	morph = MorphAnalyzer()
	words = re.findall(r'\b\w+\b', text.lower())

	if not norm: return ' '.join(words)
	
	lemmatized_words = [morph.parse(word)[0].normal_form for word in words]
	
	return ' '.join(lemmatized_words)

def extract_metadata(request):
	r = qa_model(question=[
		'Какой объем?', 
		'Кто авторы или автор упомянут?', 
		'Какой век или год или эпоха?'
		], context=request) 
	
	rs = ['length', 'author', 'date']

	return {rs[i]: r[i] for i in range(len(rs))}


def to_standard(data):
	r = ''
	if data['length']['score'] > 0.02: r += f'объем: {get_normal_sentence(data['length']['answer'], norm=False)}; '
	if data['author']['score'] > 0.02: r += f'автор: {get_normal_sentence(data['author']['answer'])}; '
	if data['date']['score'] > 0.02: r += f'временной период: {get_normal_sentence(data['date']['answer'])}'
	return r

#

def cosine_similarity(a, b):
	return float(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b))

def best_similarity(req_emb, text_embs):
	return max(cosine_similarity(req_emb, text_emb) for text_emb in text_embs)


def search_poems(request, top_n=10, search_priority=0.5):
	mentioned_parameters = extract_metadata(request)
	standard_params_text = to_standard(mentioned_parameters) + ' '
	req_params_emb = to_embeddings(standard_params_text)[0]
	if len(standard_params_text) < 14: search_priority = 1
	
	req_emb = to_embeddings(request)[0]
	texts_datas = []

	for index, row in df.iterrows():
		text_sim = best_similarity(req_emb, row['text_embedding'])
		req_sim = cosine_similarity(req_params_emb, row['metadata_embedding'])
		
		score = text_sim * search_priority + req_sim * (1-search_priority)
		if row['profanity']: score = -1
		
		texts_datas.append({'score': score, 'text_sim': text_sim, 'req_sim': req_sim, 
			**{k: v for k, v in row.items() if k != 'metadata_embedding' and k != 'text_embedding'}})

	texts_datas.sort(key=lambda x: x['score'])

	return texts_datas[-top_n:][::-1]
