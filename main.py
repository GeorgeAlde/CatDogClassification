import torch
import torch.nn as nn
import torch.optim as optim
import data

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
loader = data.get_loader()
print(len(loader))

class NeuralNetwork(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(inputs, 160),
            nn.ReLU(),

            nn.Linear(160, 80),
            nn.ReLU(),

            nn.Linear(80, 40),
            nn.ReLU(),

            nn.Linear(40, outputs)
        )
    def forward(self, x):
        return  self.network(x)
model = NeuralNetwork(30000, 1).to(device)
learning_rate = 0.001
optimiser = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.BCEWithLogitsLoss()
for epoch in range(1000):
    model.train()

    epoch_loss = 0
    for images, labels in loader:
        images = torch.flatten(images, start_dim=1, end_dim=3).to(device)
        labels = labels.view(-1, 1).float().to(device)
        y_hat = model(images)
        loss = loss_fn(y_hat, labels)

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        epoch_loss += loss.item()
    epoch_loss = epoch_loss/len(loader)

    if epoch%1==0:
        print(epoch_loss)
