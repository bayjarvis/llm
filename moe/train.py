import torch
import torch.optim as optim
import torch.nn as nn
from moe_model import SparseMoE

# Parameters
input_dim = 10
output_dim = 10
num_experts = 8
top_k = 2
hidden_dim = 32
batch_size = 64
num_samples = 1000

# Generate synthetic data
X = torch.randn(num_samples, input_dim)
y = torch.randn(num_samples, output_dim)

# Create a DataLoader
dataset = torch.utils.data.TensorDataset(X, y)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Instantiate the model, loss function, and optimizer
model = SparseMoE(input_dim, output_dim, num_experts, top_k, hidden_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    for batch_X, batch_y in dataloader:
        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print("Training finished.")

# Example of inference
with torch.no_grad():
    sample_X = torch.randn(1, input_dim)
    prediction = model(sample_X)
    print("Example prediction:", prediction)
