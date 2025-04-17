# python/pytorch/train_vgg16.py

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from codecarbon import EmissionsTracker


def get_loaders(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761))
    ])
    train_set = datasets.CIFAR100(root='../data/cifar100', train=True, download=True, transform=transform)
    test_set = datasets.CIFAR100(root='../data/cifar100', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader


def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(train_loader.dataset)


def evaluate(model, test_loader, criterion, device):
    model.eval()
    total, correct = 0, 0
    loss_sum = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss_sum += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    acc = 100. * correct / total
    return loss_sum / total, acc


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    train_loader, test_loader = get_loaders()

    model = models.vgg16(pretrained=False)
    model.classifier[6] = nn.Linear(4096, 100)  # CIFAR-100
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(30):
        print(f"Epoch {epoch + 1}/30")
        tracker.start()
        train_loss = train(model, train_loader, criterion, optimizer, device)
        tracker.stop()
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(f"Train Loss={train_loss:.4f}, Test Loss={test_loss:.4f}, Test Acc={test_acc:.2f}%")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/vgg16_cifar100.pth")


if __name__ == "__main__":
    tracker = EmissionsTracker(output_dir="emissions/", output_file="vgg16.csv")
    main()
