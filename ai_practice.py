import torch
import math

from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.transforms import v2

# Hyperparameters

# ----------------- #

epochs = 30
batch_size = 64
lr = 0.001
lmb = 2.0

# ----------------- #

training_data = datasets.FashionMNIST (
    root = "data",
    train = True,
    download = True,
    transform = ToTensor()
)

augmented_training_data = datasets.FashionMNIST (
    root = "data",
    train = True,
    download = True,
    transform = v2.Compose([v2.RandomHorizontalFlip(1.0), ToTensor()])
)

testing_data = datasets.FashionMNIST (
    root = "data",
    train = False,
    download = True,
    transform = ToTensor()
)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

PATH = "C:/Users/aaaaa/Desktop/could this be love/model.pt"

class neural_network(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential (
            nn.Conv2d(1, 20, kernel_size = (5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2)),

            nn.Conv2d(20, 20, kernel_size = (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2)),

            nn.Flatten(),

            nn.Linear(500, 200),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(200, 10)
        )
    
    def forward(self, x):
        output = self.layers(x)
        return output

def train(dataloader, net, cost_function, optimizer):
    size = len(dataloader.dataset)
    net.train()

    for batch, (x, y) in enumerate(dataloader):
        x = x.to(device)
        y = y.to(device)

        ans = net(x)
        cost = cost_function(ans, y)

        cost.backward()
        optimizer.step()
        optimizer.zero_grad()

def test(mode, dataloader, net, cost_function, optimizer):
    size = len(dataloader.dataset)

    net.eval()

    correct_tests = 0
    cost = 0

    with torch.no_grad():
        for batch, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)

            ans = net(x)
            cost += cost_function(ans, y).item()
            correct_tests += (ans.argmax(1) == y).type(torch.float).sum().item()
    
    cost /= math.ceil(size / batch_size)

    if mode == 0:
        print(f"Accuracy for Tests: {100.0 * correct_tests / size:>0.1f} \n Average loss: {cost}")
    else:
        print(f"Accuracy for Training: {100.0 * correct_tests / size:>0.1f} \n Average loss: {cost}")

net = neural_network.to(device)

w = []

w.append(training_data)
w.append(augmented_training_data)

extended_training_data = ConcatDataset(w)

training_dataloader = DataLoader(extended_training_data, batch_size = batch_size, shuffle = True)
testing_dataloader = DataLoader(testing_data, batch_size = batch_size, shuffle = True)

cost_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr, betas = (0.9, 0.999), eps = 1e-8, weight_decay = lmb / (2 * len(training_dataloader)))

for ep in range(epochs):
    train(training_dataloader, net, cost_function, optimizer)

    print(f"Epoch {ep}:")

    test(0, testing_dataloader, net, cost_function, optimizer)
    test(1, training_dataloader, net, cost_function, optimizer)

    torch.save(net, PATH)

    print("\n")

    epochs -= 1
