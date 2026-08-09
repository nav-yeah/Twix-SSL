import torch


def normalize(z):
    return (z - z.mean(dim=0)) / z.std(dim=0)


def cross_correlation(z1_norm, z2_norm, batch_size):
    c = z1_norm.T @ z2_norm
    c = c / batch_size
    return c


def barlow_twins_loss(z1, z2, lambda_param=0.005):
    batch_size, feature_dim = z1.shape

    z1_norm = normalize(z1)
    z2_norm = normalize(z2)

    c = cross_correlation(z1_norm, z2_norm, batch_size)

    on_diag = torch.diagonal(c)
    invariance_loss = ((on_diag - 1) ** 2).sum()

    #instead of using a loop where i!=j a single line to extract off diag.elements
    off_diag = c.flatten()[:-1].view(c.size(0) - 1, c.size(1) + 1)[:, 1:].flatten()
    redundancy_loss = (off_diag ** 2).sum()

    loss = invariance_loss + lambda_param * redundancy_loss
    return loss


if __name__ == "__main__":

    z1 = torch.randn(32, 512)
    z2 = z1.clone()

    loss = barlow_twins_loss(z1, z2)
    print(f"Loss for identical embeddings: {loss.item():.6f}")

    z1_random = torch.randn(32, 512)
    z2_random = torch.randn(32, 512)
    loss_random = barlow_twins_loss(z1_random, z2_random)
    print(f"Loss for random embeddings: {loss_random.item():.6f}")