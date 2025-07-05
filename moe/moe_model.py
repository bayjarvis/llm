import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    """An expert network."""
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(Expert, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class SparseMoE(nn.Module):
    """A sparse mixture of experts layer."""
    def __init__(self, input_dim, output_dim, num_experts, top_k, hidden_dim):
        super(SparseMoE, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([Expert(input_dim, output_dim, hidden_dim) for _ in range(num_experts)])
        self.gating_network = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        # Get gating scores
        gating_scores = self.gating_network(x)
        
        # Get top-k experts
        top_k_scores, top_k_indices = torch.topk(gating_scores, self.top_k, dim=1)
        
        # Normalize gating scores
        top_k_scores = F.softmax(top_k_scores, dim=1)
        
        # Initialize output tensor
        output = torch.zeros_like(x.new_empty(x.size(0), self.experts[0].fc2.out_features))
        
        # Process each sample in the batch
        for i in range(x.size(0)):
            # Process each expert for the current sample
            for j in range(self.top_k):
                expert_index = top_k_indices[i, j]
                expert_output = self.experts[expert_index](x[i])
                output[i] += top_k_scores[i, j] * expert_output
                
        return output
