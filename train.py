import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torchvision.models import MobileNet_V2_Weights

def main():
    # CONFIG
    TRAIN_DIR = "data/train"
    TEST_DIR = "data/test"

    BATCH_SIZE = 64
    EPOCHS = 5
    LR = 0.0005

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # TRANSFORMS
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    train_data = datasets.ImageFolder(TRAIN_DIR, transform=transform)
    test_data = datasets.ImageFolder(TEST_DIR, transform=transform)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, num_workers=2)

    # CLASS IMBALANCE
    class_counts = [0]*len(train_data.classes)
    for _, label in train_data:
        class_counts[label] += 1

    weights = torch.tensor([1.0/c for c in class_counts]).to(device)

    # MODEL
    model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

    # Freeze backbone
    for param in model.features.parameters():
        param.requires_grad = False

    model.classifier[1] = nn.Linear(model.last_channel, len(train_data.classes))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.classifier.parameters(), lr=LR)

    # TRAIN
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # VALIDATION
        model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.2f}, Accuracy: {acc:.2f}%")

    torch.save(model.state_dict(), "model.pth")
    print("✅ model.pth saved")


if __name__ == "__main__":
    main()