import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

hidden_layers = [64, 32]

class QNetwork(nn.Module):

    def __init__(self, state_size, action_size, seed):
        super().__init__()
        self.seed = torch.manual_seed(seed)
        
        self.hidden_layers = nn.ModuleList([nn.Linear(state_size, hidden_layers[0])])    
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        
        self.output = nn.Linear(hidden_layers[-1], action_size)

    def forward(self, state):
        x = state
        for linear in self.hidden_layers:
            x = F.relu(linear(x))
        return self.output(x)


class DDQNetwork(nn.Module):

    def __init__(self, state_size, action_size, seed):
        super().__init__()
        self.action_size = action_size
        self.seed = torch.manual_seed(seed)

        self.hidden_layers = nn.ModuleList([nn.Linear(state_size, hidden_layers[0])])    
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])

        self.value = nn.Linear(hidden_layers[-1], 1)
        self.advantage = nn.Linear(hidden_layers[-1], action_size)

    def forward(self, state):
        x = state
        for linear in self.hidden_layers:
            x = F.relu(linear(x))  # last x.size() = (batch_size, hidden_layers[-1])
        adv = F.relu(self.advantage(x))
        val = F.relu(self.value(x))

        mean = adv.mean(1).unsqueeze(1).expand(x.size(0), self.action_size)
        # calculate the mean by rows and unsqueeze to become size = (batch, 1), then expand to (batch, action_size)

        # if don't expand val size, although the calculation is the same, but the model seems won't learn anything
        val = val.expand(x.size(0), self.action_size)  # val.size() = (batch, self.action_size)
        result = val + self.advantage(x) - mean

        return result
