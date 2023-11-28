import networkx as nx
import torch
import pandas as pd
from node2vec import Node2Vec
from sentence_transformers import SentenceTransformer

df = pd.read_csv('IMDB-Movie-Data.csv')  

df['Revenue (Millions)'].fillna(0, inplace=True)
df['Metascore'].fillna(df['Metascore'].mean(), inplace=True) 

df['Actors'] = df['Actors'].str.split(',')
df['Genre'] = df['Genre'].str.split(',')
df['Director'] = df['Director'].str.lower()
G = nx.Graph()

for index, row in df.iterrows():
    G.add_node(str(row['Title']),type="title")
    G.add_node(str(row['Year']),type="year")
    G.add_edge(str(row['Title']), str(row['Year']), relationship='year')
    if not G.has_node(row['Director']):
        G.add_node(row['Director'], type="director")
    G.add_edge(str(row['Title']), row['Director'], relationship='director')
    for actor in row['Actors']:
        if not G.has_node(actor):
            G.add_node(actor, type="actor")
        G.add_edge(str(row['Title']), actor, relationship='actor')
    for genre in row['Genre']:
        if not G.has_node(genre):
            G.add_node(genre, type="genre")
        G.add_edge(str(row['Title']), genre, relationship='genres')

# Node2Vec embeddings
node2vec = Node2Vec(G, dimensions=64, walk_length=40, num_walks=300, workers=4)
model = node2vec.fit(window=10, min_count=1, batch_words=4)

# BERT embeddings
bert_model = SentenceTransformer('bert-large-nli-stsb-mean-tokens')
node_list = list(G.nodes)

node_embeddings = [model.wv[str(node)] for node in node_list]
node_embeddings_tensor = torch.tensor(node_embeddings, dtype=torch.float32)

bert_embeddings = bert_model.encode([str(node) for node in node_list])

# Save Node Embeddings to Dictionary
node_to_node_embedding_dict = {node: emb.tolist() for node, emb in zip(node_list, node_embeddings)}
torch.save(node_to_node_embedding_dict, 'node_to_node_embedding_dict.pth')

# Save BERT Embeddings to Dictionary
node_to_bert_embedding_dict = {node: emb.tolist() for node, emb in zip(node_list, bert_embeddings)}
torch.save(node_to_bert_embedding_dict, 'node_to_bert_embedding_dict.pth')
