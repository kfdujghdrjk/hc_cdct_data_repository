from __future__ import annotations

import torch


def train_one_epoch(model, trainloader, criterion, optimizer, device: str, epoch: int = 0):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in trainloader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return {
        "epoch": epoch,
        "loss": train_loss / max(len(trainloader), 1),
        "accuracy": 100.0 * correct / max(total, 1),
    }


@torch.no_grad()
def evaluate(model, testloader, criterion, device: str):
    model.eval()

    test_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in testloader:
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return {
        "loss": test_loss / max(len(testloader), 1),
        "accuracy": 100.0 * correct / max(total, 1),
    }
