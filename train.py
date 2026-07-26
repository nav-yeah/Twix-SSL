import torch
from torch.utils.data import Subset

from dataset import get_dataloader
from model import BarlowTwinsModel
from loss import barlow_twins_loss


def train(subset_size=2000, batch_size=256, epochs=10, lr=1e-3, lambda_param=0.005):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # Build the full loader, then trim to a small subset for this proof run.
    # (Remove this Subset step next week, once you're ready to scale to the
    # full 100k unlabeled images.)
    full_loader = get_dataloader(batch_size=batch_size)
    full_dataset = full_loader.dataset
    subset = Subset(full_dataset, range(subset_size))

    loader = torch.utils.data.DataLoader(
        subset, batch_size=batch_size, shuffle=True,
        num_workers=4, drop_last=True, pin_memory=True,
    )

    model = BarlowTwinsModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0

        for view1, view2 in loader:
            view1, view2 = view1.to(device), view2.to(device)

            z1 = model(view1)
            z2 = model(view2)
            loss = barlow_twins_loss(z1, z2, lambda_param=lambda_param)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        print(f"Epoch {epoch+1}/{epochs} - avg loss: {avg_loss:.4f}")

    torch.save(model.encoder.state_dict(), 'encoder_checkpoint.pth')
    print("Saved encoder weights to encoder_checkpoint.pth")


if __name__ == "__main__":
    train()