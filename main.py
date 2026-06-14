import torch
import torch.nn as nn
import torch.optim as optim
import data

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
loader = data.get_loader()
print(len(loader))

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),      # 128 -> 64

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),      # 64 -> 32

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),      # 32 -> 16
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
model = CNN().to(device)
learning_rate = 0.001
optimiser = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.BCEWithLogitsLoss()
for epoch in range(1000):
    model.train()

    epoch_loss = 0
    for images, labels in loader:
        images = images.to(device)
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
