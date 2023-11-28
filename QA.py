import networkx as nx
import torch
import pandas as pd
from node2vec import Node2Vec
from sentence_transformers import SentenceTransformer
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from collections import OrderedDict
import re
import spacy

# Load and preprocess data
df = pd.read_csv('IMDB-Movie-Data.csv')  
df['Revenue (Millions)'].fillna(0, inplace=True)
df['Metascore'].fillna(df['Metascore'].mean(), inplace=True) 
df['Actors'] = df['Actors'].str.split(',')
df['Genre'] = df['Genre'].str.split(',')
df['Director'] = df['Director'].str.lower()

# Build the graph
G = nx.Graph()
NERobjects = []

for index, row in df.iterrows():
    G.add_node(str(row['Title']), type="title")
    if str(row['Title']) not in NERobjects:
        NERobjects.append(str(row['Title']))

    G.add_node(str(row['Year']), type="year")
    if str(row['Year']) not in NERobjects:
        NERobjects.append(str(row['Year']))

    G.add_edge(str(row['Title']), str(row['Year']), relationship='year')

    if not G.has_node(row['Director']):
        G.add_node(str(row['Director']), type="director")
        if str(row['Director']) not in NERobjects:
            NERobjects.append(str(row['Director']))

    G.add_edge(str(row['Title']), str(row['Director']), relationship='director')

    for actor in row['Actors']:
        actor = str(actor)
        if not G.has_node(actor):
            G.add_node(actor, type="actor")
            if actor not in NERobjects:
                NERobjects.append(actor)
        G.add_edge(str(row['Title']), actor, relationship='actor')

    for genre in row['Genre']:
        genre = str(genre)
        if not G.has_node(genre):
            G.add_node(genre, type="genre")
            if genre not in NERobjects:
                NERobjects.append(genre)


# Node2Vec embeddings
node2vec = Node2Vec(G, dimensions=64, walk_length=40, num_walks=300, workers=4)
model = node2vec.fit(window=10, min_count=1, batch_words=4)

# BERT embeddings
bert_model = SentenceTransformer('bert-large-nli-stsb-mean-tokens')
node_list = list(G.nodes)

node_embeddings = [model.wv[str(node)] for node in node_list]
node_embeddings_tensor = torch.tensor(node_embeddings, dtype=torch.float32)

bert_embeddings = bert_model.encode([str(node) for node in node_list])

# Save embeddings to dictionaries
node_to_node_embedding_dict = {node: emb.tolist() for node, emb in zip(node_list, node_embeddings)}
torch.save(node_to_node_embedding_dict, 'node_to_node_embedding_dict.pth')

node_to_bert_embedding_dict = {node: emb.tolist() for node, emb in zip(node_list, bert_embeddings)}
torch.save(node_to_bert_embedding_dict, 'node_to_bert_embedding_dict.pth')

# Downstream Task
class DownstreamTask(nn.Module):
    def __init__(self):
        super(DownstreamTask, self).__init__()
        self.fc1 = nn.Linear(1024, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

downstream_model = DownstreamTask()
criterion = nn.MSELoss()
optimizer = optim.Adam(downstream_model.parameters(), lr=0.001)

node_embeddings_tensor = torch.tensor(list(node_to_node_embedding_dict.values()), dtype=torch.float32)
bert_embeddings = torch.tensor(list(node_to_bert_embedding_dict.values()), dtype=torch.float32)
# Train the Downstream Task
X_train_tensor = bert_embeddings
y_train_tensor = node_embeddings_tensor

num_epochs = 100
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = downstream_model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

torch.save({
    'epoch': num_epochs,
    'model_state_dict': downstream_model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
    }, 'trained_model.pth')



def get_neighboring_nodes(word_list, kg):
    allneighbours = []
    for node in kg.nodes():
        if node.lower() in word_list:
            allneighbours.append(node)
            neighbors = list(kg.neighbors(node))
            for n in neighbors:
                allneighbours.append(n)
    return allneighbours

# Load the trained downstream model
downstream_model = DownstreamTask()
checkpoint = torch.load('trained_model.pth')
downstream_model.load_state_dict(checkpoint['model_state_dict'])
downstream_model.eval()

# Load the BERT model
bert_model = SentenceTransformer('bert-large-nli-stsb-mean-tokens')

# Load node embeddings dictionaries
node_to_node_embedding_dict = torch.load('node_to_node_embedding_dict.pth')

# Function to get BERT embedding for a given question
def get_bert_embedding(question):
    return torch.tensor(bert_model.encode(question)).unsqueeze(0)

# Function to get predicted node embedding using the downstream model
def get_predicted_node_embedding(bert_embedding):
    with torch.no_grad():
        return downstream_model(bert_embedding)

# Function to find top k matching node embeddings
def find_top_k_matching_nodes(predicted_embedding, node_to_embedding_dict, nnodes, k=15):
    valid_nodes = [node for node in nnodes if node in node_to_embedding_dict]
    distances = {node: torch.norm(predicted_embedding - torch.tensor(embedding), dim=1).item() for node, embedding in node_to_embedding_dict.items() if node in valid_nodes}
    sorted_distances = OrderedDict(sorted(distances.items(), key=lambda x: x[1]))
    top_k_nodes = list(sorted_distances.keys())[:k]
    return top_k_nodes

# Example usage:
inputSentence, outputSentence = [], []
with open('dataset.txt', 'r') as file:
    lines = file.readlines()
    for line in lines:
        line = line.replace('\n', '')
        answer_pattern = r'Answer: (.*)'
        if re.search(answer_pattern, line):
            outputSentence.append(line[8:])
        elif line != '' and line != '\n':
            inputSentence.append(line)

# Evaluation
correctPredict = 0
total = 0
for i in range(len(inputSentence)):
    total += 1
    question = inputSentence[i]
    output = outputSentence[i]
    bert_embedding = get_bert_embedding(question)
    predicted_embedding = get_predicted_node_embedding(bert_embedding)
    entities = [ner.lower() for ner in NERobjects if ner.lower() in question.lower()]
    neighboring_nodes = get_neighboring_nodes(entities, G)
    nnodes = set(neighboring_nodes)
    top_k_nodes = find_top_k_matching_nodes(predicted_embedding, node_to_node_embedding_dict, nnodes)

    for node in top_k_nodes:
        if str(node).lower() == str(output).lower():
            correctPredict += 1

print("P@15 is",correctPredict/ total)
