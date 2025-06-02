import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def train(model, dataloader, device, epochs, lr):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            progress_bar.set_postfix(loss=loss.item(), acc=100 * correct / total)

        avg_loss = total_loss / len(dataloader)
        accuracy = 100 * correct / total
        print(f"[Epoch {epoch+1}] Avg Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return model

