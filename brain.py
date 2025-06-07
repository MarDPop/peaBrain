import torch
import torch.nn as nn
import numpy as np

class Brain(nn.Module):
    """A simple neural network model representing the brain of an animal."""

    def __init__(self, input_size, hidden_size, output_size):
        super(Brain, self).__init__()
        # Initialize the neural network layers
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()  # For output normalization
        
        # Store the last input and output for learning
        self.last_input = None
        self.last_output = None
        
        # Initialize weights with small values
        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Convert numpy array to torch tensor if needed
        if isinstance(inputs, np.ndarray):
            inputs = torch.FloatTensor(inputs)
        
        # Store input for learning
        self.last_input = inputs
        
        # Forward pass
        x = self.relu(self.layer1(inputs))
        x = self.tanh(self.layer2(x))
        
        # Store output for learning
        self.last_output = x
        
        return x
    
    def learn(self, reward, learning_rate=0.01):
        """Update the brain's weights based on the reward using policy gradient."""
        if self.last_input is None or self.last_output is None:
            return
            
        # Convert reward to tensor
        reward = torch.FloatTensor([reward])
        
        # Calculate loss (negative because we want to maximize reward)
        loss = -reward * torch.log(torch.abs(self.last_output) + 1e-6).mean()
        
        # Zero gradients
        self.zero_grad()
        
        # Backward pass
        loss.backward()
        
        # Update weights
        with torch.no_grad():
            for param in self.parameters():
                param.data -= learning_rate * param.grad