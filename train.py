import torch
from torch.utils.data import Subset
 
from dataset import get_dataloader
from model import BarlowTwinsModel
from loss import barlow_twins_loss
 
 
def train(subset_size=None, batch_size=256, epochs=50, lr=1e-3, lambda_param=0.005,
          checkpoint_every=5, resume_from=None):
    """
    subset_size=None  -> train on the full unlabeled split (100k images)
    subset_size=2000  -> train on a small subset, e.g. for a quick proof run
    resume_from       -> path to a saved encoder checkpoint to continue from
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
 
    full_loader = get_dataloader(batch_size=batch_size)
 
    if subset_size is not None:
        full_dataset = full_loader.dataset
        dataset = Subset(full_dataset, range(subset_size))
    else:
        dataset = full_loader.dataset
 
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, drop_last=True, pin_memory=True,
    )
    print(f"Training on {len(dataset)} images, {len(loader)} batches per epoch")
 
    model = BarlowTwinsModel().to(device)
 
    if resume_from is not None:
        model.encoder.load_state_dict(torch.load(resume_from, map_location=device))
        print(f"Resumed encoder weights from {resume_from}")
 
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
 
        # Save periodically so a Kaggle session timeout doesn't lose everything
        if (epoch + 1) % checkpoint_every == 0 or (epoch + 1) == epochs:
            ckpt_path = f'encoder_checkpoint_epoch{epoch+1}.pth'
            torch.save(model.encoder.state_dict(), ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")
 
    torch.save(model.encoder.state_dict(), 'encoder_checkpoint_final.pth')
    print("Saved final encoder weights to encoder_checkpoint_final.pth")
 
 
if __name__ == "__main__":
    train()
 