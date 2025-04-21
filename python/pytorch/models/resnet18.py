import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from codecarbon import EmissionsTracker


def build_resnet18(num_classes: int = 100, pretrained: bool = False) -> nn.Module:
    model = models.resnet18(pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    if not pretrained:
        def init_weights(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        model.apply(init_weights)

    return model


def get_loaders(dataset_path, batch_size=128, img_size=(32, 32), grayscale=False, test_split="test"):
    transform_list = [transforms.Resize(img_size)]
    if grayscale:
        transform_list.append(transforms.Grayscale(num_output_channels=3))
    transform_list.append(transforms.ToTensor())

    transform = transforms.Compose(transform_list)
    train_set = ImageFolder(root=os.path.join(dataset_path, "train"), transform=transform)
    test_set = ImageFolder(root=os.path.join(dataset_path, test_split), transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader, len(train_set.classes)


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


def run_experiment(dataset_path, output_file, checkpoint_path, img_size=(32, 32), grayscale=False, test_split="test",
                   epochs=30, batch_size=128):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    train_loader, test_loader, num_classes = get_loaders(dataset_path, batch_size, img_size, grayscale, test_split)

    model = build_resnet18(num_classes=num_classes, pretrained=False)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    tracker = EmissionsTracker(output_dir="python/pytorch/emissions/", output_file=output_file)

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        tracker.start()
        train_loss = train(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        tracker.stop()

        print(f"Train Loss={train_loss:.4f}, Test Loss={test_loss:.4f}, Test Acc={test_acc:.2f}%")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
