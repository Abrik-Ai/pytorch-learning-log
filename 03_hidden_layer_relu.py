# PATTERN: Adding a hidden layer with ReLU activation (real neural network,
# not just linear regression anymore)
# Learned: hidden layers, neurons, why ReLU is needed (without it, stacking
# linear layers mathematically collapses into just one linear layer),
# nn.Sequential-style forward pass through multiple layers
# Goal: same y = 2x problem, but using a small neural network architecture
# (1 -> 8 -> 1) instead of a single linear layer

import torch
import torch.nn as nn

# Training data: y = 2x
x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y = torch.tensor([[2.0], [4.0], [6.0], [8.0]])

# Model with one hidden layer (8 neurons) + ReLU activation
class HiddenLayerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1, 8)     # input -> hidden (8 neurons)
        self.activation = nn.ReLU()        # nonlinearity — without this,
                                            # stacking layer1+layer2 would be
                                            # mathematically identical to 1 layer
        self.layer2 = nn.Linear(8, 1)      # hidden -> output

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x

model = HiddenLayerModel()
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(20):
    y_pred = model(x)
    loss = loss_fn(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"epoch {epoch}: loss = {loss.item():.3f}")
