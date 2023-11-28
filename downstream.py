import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sentence_transformers import SentenceTransformer
import torch.optim as optim

# Load saved node embeddings and BERT embeddings dictionaries
node_to_node_embedding_dict = torch.load('node_to_node_embedding_dict.pth')
node_to_t5_embedding_dict = torch.load('node_to_bert_embedding_dict.pth')

# Assuming you have a list of nodes for which you want to get BERT embeddings
node_list = list(node_to_node_embedding_dict.keys())

# Convert dictionaries to tensors
node_embeddings_tensor = torch.tensor(list(node_to_node_embedding_dict.values()), dtype=torch.float32)
bert_embeddings = torch.tensor(list(node_to_t5_embedding_dict.values()), dtype=torch.float32)

print(node_embeddings_tensor.shape)
print(bert_embeddings.shape)

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
criterion = nn.MSELoss()
optimizer = optim.Adam(downstream_model.parameters(), lr=0.001)

# Change the input and output logic for the downstream model
X_train_tensor = bert_embeddings  # Use BERT embeddings as input
y_train_tensor = node_embeddings_tensor  # Use node embeddings as target

num_epochs = 1000
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
