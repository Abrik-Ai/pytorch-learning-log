# PATTERN: Saving a trained model's weights to disk, then loading them back
# into a fresh model instance
# Learned: model.state_dict() collects all weights/biases into a dictionary;
# torch.save() writes it to a file; torch.load() reads it back;
# model.load_state_dict() copies those weights into a new model instance;
# model.eval() switches to evaluation mode (important for layers like
# Dropout/BatchNorm, though not used in this simple model yet)
#
# KEY RESULT: after saving and reloading, the fresh model correctly predicted
# ~10.049 for input x=5 (true answer: 10, since y=2x) — confirming no
# information was lost in the save/load cycle.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

x = torch.tensor([[1.0],[2.0],[3.0],[4.0],[5.0],[6.0],[7.0],[8.0],[9.0],[10.0]])
y = torch.tensor([[2.0],[4.0],[6.0],[8.0],[10.0],[12.0],[14.0],[16.0],[18.0],[20.0]])

dataset = TensorDataset(x, y)
dataloader = DataLoader(dataset, batch_size=3, shuffle=True)

class HiddenLayer(nn.Module):  #
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1, 8)
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(8, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x

model = HiddenLayer()
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

for epoch in range(100):
    epoch_loss = []
    for batch_x, batch_y in dataloader:
        y_pred = model(batch_x)
        loss = loss_fn(y_pred, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss.append(loss.item())

    avg_loss = sum(epoch_loss) / len(epoch_loss)
    print(f"epoch {epoch}: avg loss = {avg_loss:.3f}")

# --- Save the trained model ---
torch.save(model.state_dict(), 'my_model.pth')
print("Model saved!")

# --- Load it back into a fresh model ---
model2 = HiddenLayer()
model2.load_state_dict(torch.load('my_model.pth'))
model2.eval()

# --- Test the loaded model ---
test_input = torch.tensor([[5.0]])
prediction = model2(test_input)
print(f"Loaded model prediction for x=5: {prediction.item():.3f}")
