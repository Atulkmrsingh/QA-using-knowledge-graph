import torch
import networkx as nx
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from collections import OrderedDict
import re
import spacy
import pandas as pd
#get the kg
df = pd.read_csv('IMDB-Movie-Data.csv')  
# nlp = spacy.load("en_core_web_sm")
df['Revenue (Millions)'].fillna(0, inplace=True)
df['Metascore'].fillna(df['Metascore'].mean(), inplace=True) 

df['Actors'] = df['Actors'].str.split(',')
df['Genre'] = df['Genre'].str.split(',')
df['Director'] = df['Director'].str.lower()
G = nx.Graph()

stopfile="stop_words.txt"
with open(stopfile, 'r') as file:
    stop_word_list = [line.strip() for line in file]


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
        G.add_edge(str(row['Title']), genre, relationship='genres')
# print(G)


def get_neighboring_nodes(word_list, kg):
    allneighbours=[]
    # Iterate over all nodes in the KG
    for node in kg.nodes():
        if node.lower() in word_list:
            allneighbours.append(node)
            neighbors = list(kg.neighbors(node))
            for n in neighbors:
                allneighbours.append(n)

    return allneighbours
# Load the trained downstream model
class DownstreamTask(nn.Module):
    def __init__(self):
        super(DownstreamTask, self).__init__()
        self.fc1 = nn.Linear(1024, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)  # Assuming you want 64-dimensional node embeddings

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x= torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

downstream_model = DownstreamTask()

# Load the state dictionary
checkpoint = torch.load('trained_model.pth')
downstream_model.load_state_dict(checkpoint['model_state_dict'])

# Set the model to evaluation mode
downstream_model.eval()

# Load the BERT model
bert_model = SentenceTransformer('bert-large-nli-stsb-mean-tokens')

# Load node embeddings dictionaries
node_to_node_embedding_dict = torch.load('node_to_node_embedding_dict.pth')
# node_to_bert_embedding_dict = torch.load('node_to_bert_embedding_dict.pth')

# Function to get BERT embedding for a given question
def get_bert_embedding(question):
    return torch.tensor(bert_model.encode(question)).unsqueeze(0)

# Function to get predicted node embedding using the downstream model
def get_predicted_node_embedding(bert_embedding):
    with torch.no_grad():
        return downstream_model(bert_embedding)

# Function to find top k matching node embeddings
def find_top_k_matching_nodes(predicted_embedding, node_to_embedding_dict, nnodes, k=25):
    valid_nodes = [node for node in nnodes if node in node_to_embedding_dict]

    distances = {node: torch.norm(predicted_embedding - torch.tensor(embedding), dim=1).item() for node, embedding in node_to_embedding_dict.items() if node in valid_nodes}
    
    sorted_distances = OrderedDict(sorted(distances.items(), key=lambda x: x[1]))
    top_k_nodes = list(sorted_distances.keys())[:k]
    
    return top_k_nodes

# Example usage:
inputSentence,outputSentence=[],[]
with open('dataset.txt', 'r') as file:
    lines = file.readlines()
    for line in lines:
        line=line.replace('\n','')
        answer_pattern = r'Answer: (.*)'
        if(re.search(answer_pattern, line)):
            outputSentence.append(line[8:])
        elif(line!='' and line!='\n'):
            inputSentence.append(line)

# question = "In which year was 'Jurassic World' released, featuring Chris Pratt and Bryce Dallas Howard in a dinosaur theme park adventure?"
# entities=[]
# for ner in NERobjects:
#     if str(ner).lower() in question.lower():
#         entities.append(ner.lower())
# print(entities)
# neighboring_nodes = get_neighboring_nodes(entities, G)
# nnodes=set(neighboring_nodes)
# # Print the resulting neighboring nodes
# print(len(nnodes))
# bert_embedding = get_bert_embedding(question)
# predicted_embedding = get_predicted_node_embedding(bert_embedding)

# top_k_nodes = find_top_k_matching_nodes(predicted_embedding, node_to_node_embedding_dict, nnodes)

# print(top_k_nodes)

correctPredict=0
total=0
for i in range(len(inputSentence)):
    total+=1
    question=inputSentence[i]
    output=outputSentence[i]
    # question = "Who portrayed the character of Lisbeth Salander in the 2018 film 'The Girl in the Spider's Web'?"
    bert_embedding = get_bert_embedding(question)
    predicted_embedding = get_predicted_node_embedding(bert_embedding)
    entities=[]
    for ner in NERobjects:
        if str(ner).lower() in question.lower():
            entities.append(ner.lower())
    neighboring_nodes = get_neighboring_nodes(entities, G)
    nnodes=set(neighboring_nodes)
    # Find top k matching nodes
    top_k_nodes = find_top_k_matching_nodes(predicted_embedding, node_to_node_embedding_dict, nnodes)

    # Print top k matching nodes
    # print(f'Top {len(top_k_nodes)} matching nodes:')
    for node in top_k_nodes:
        # print(node)
        if(str(node).lower()==str(output).lower()):
            correctPredict+=1
print(correctPredict,total)