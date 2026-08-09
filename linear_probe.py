import torch
import torch.nn as nn
import numpy as np
from torchvision.datasets import STL10
from torchvision import transforms
from torch.utils.data import DataLoader

from model import BarlowTwinsModel


# No augmentation here -- for evaluation we want a clean, consistent view
# of each image, not randomized ones.
eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4467, 0.4398, 0.4066],
                          std=[0.2603, 0.2566, 0.2713]),
])


def load_encoder(checkpoint_path, device):
    model = BarlowTwinsModel()
    model.encoder.load_state_dict(torch.load(checkpoint_path, map_location=device))
    encoder = model.encoder.to(device)
    return encoder, model.feature_dim


def get_fold_loader(fold, root='./data', batch_size=64):
    dataset = STL10(root=root, split='train', folds=fold,
                     download=True, transform=eval_transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)


def get_test_loader(root='./data', batch_size=64):
    dataset = STL10(root=root, split='test', download=True, transform=eval_transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)


def linear_probe(encoder, feature_dim, train_loader, test_loader, device,
                  num_classes=10, epochs=20, lr=1e-3):
    encoder.eval()  # frozen -- we never update the pretrained encoder
    for p in encoder.parameters():
        p.requires_grad = False

    classifier = nn.Linear(feature_dim, num_classes).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        classifier.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                features = encoder(images)
            preds = classifier(features)
            loss = criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluate on the full test set
    classifier.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            features = encoder(images)
            preds = classifier(features).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    return accuracy


def run_harness(checkpoint_path, num_folds=10, epochs=20):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    encoder, feature_dim = load_encoder(checkpoint_path, device)
    test_loader = get_test_loader()

    accuracies = []
    for fold in range(num_folds):
        train_loader = get_fold_loader(fold)
        acc = linear_probe(encoder, feature_dim, train_loader, test_loader,
                            device, epochs=epochs)
        print(f"Fold {fold}: accuracy = {acc:.4f}")
        accuracies.append(acc)

    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    print(f"\nMean accuracy across {num_folds} folds: {mean_acc:.4f} +/- {std_acc:.4f}")
    return accuracies, mean_acc, std_acc


if __name__ == "__main__":
    run_harness('encoder_checkpoint_final.pth')