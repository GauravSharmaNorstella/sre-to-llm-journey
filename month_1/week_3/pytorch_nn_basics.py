
import torch
from torch import nn

X = torch.randn(64, 10)
y = (torch.randn(64) > 0).long()

model = nn.Sequential(
    nn.Linear(10, 32), nn.ReLU(),
    nn.Linear(32, 2)
)

opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(10):
    opt.zero_grad()
    logits = model(X)
    loss = loss_fn(logits, y)
    loss.backward()
    opt.step()
    print(f'Epoch {epoch+1}: loss={loss.item():.4f}')
