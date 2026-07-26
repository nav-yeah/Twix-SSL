import torch
from torchvision.datasets import STL10
from torchvision import transforms
from torch.utils.data import DataLoader


class BarlowTwinsSTL10(STL10):
    """
    Wraps torchvision's STL10 dataset to return two independently
    augmented views of the same image, instead of (image, label).
    Used for Barlow Twins self-supervised pretraining.
    """

    def __init__(self, root, split='unlabeled', download=True):
        super().__init__(root=root, split=split, download=download)

        self.pair_transform = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4467, 0.4398, 0.4066],
                                  std=[0.2603, 0.2566, 0.2713]),
        ])

    def __getitem__(self, index):
        img, _ = super().__getitem__(index)  # ignore label; unlabeled split has none anyway
        view1 = self.pair_transform(img)
        view2 = self.pair_transform(img)
        return view1, view2


def get_dataloader(batch_size=32, root='./data', split='unlabeled', num_workers=4):
    dataset = BarlowTwinsSTL10(root=root, split=split, download=True)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,     # important: Barlow Twins loss needs a full, consistent batch
        pin_memory=True,
    )
    return loader


if __name__ == "__main__":
    # Quick sanity check: confirm shapes are correct before using this
    # in the training loop.
    loader = get_dataloader(batch_size=32)
    view1, view2 = next(iter(loader))
    print("view1 shape:", view1.shape)  # expect: torch.Size([32, 3, 96, 96])
    print("view2 shape:", view2.shape)  # expect: torch.Size([32, 3, 96, 96])
