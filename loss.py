import torch


def normalize(z):
    """
    Normalize embeddings so each feature (column) has mean 0, std 1
    across the batch. This is required before computing the
    cross-correlation matrix.

    z: tensor of shape (batch_size, feature_dim)
    """
    return (z - z.mean(dim=0)) / z.std(dim=0)


def cross_correlation(z1_norm, z2_norm, batch_size):
    """
    Compute the cross-correlation matrix between two sets of
    normalized embeddings.

    z1_norm, z2_norm: tensors of shape (batch_size, feature_dim)
    Returns: matrix of shape (feature_dim, feature_dim), where
    c[i][j] represents how much feature i (from view1) correlates
    with feature j (from view2), across the batch.
    """
    c = z1_norm.T @ z2_norm
    c = c / batch_size
    return c


def barlow_twins_loss(z1, z2, lambda_param=0.005):
    """
    Compute the Barlow Twins loss between two embeddings of the
    same batch of images (each passed through a different random
    augmentation).

    z1, z2: tensors of shape (batch_size, feature_dim)
    lambda_param: weight on the redundancy-reduction term
                  (0.005 is the value used in the original paper)

    Returns: a single scalar loss value
    """
    batch_size, feature_dim = z1.shape

    z1_norm = normalize(z1)
    z2_norm = normalize(z2)

    c = cross_correlation(z1_norm, z2_norm, batch_size)

    # Diagonal terms: same feature across both views should agree.
    # We want these values close to 1 (invariance).
    on_diag = torch.diagonal(c)
    invariance_loss = ((on_diag - 1) ** 2).sum()

    # Off-diagonal terms: different features shouldn't be redundant
    # with each other. We want these values close to 0.
    off_diag = c.flatten()[:-1].view(c.size(0) - 1, c.size(1) + 1)[:, 1:].flatten()
    redundancy_loss = (off_diag ** 2).sum()

    loss = invariance_loss + lambda_param * redundancy_loss
    return loss


if __name__ == "__main__":
    # Quick sanity test: identical embeddings should give a
    # near-zero loss, since a feature perfectly agrees with itself
    # and there's no redundancy issue when z1 == z2 exactly.
    z1 = torch.randn(32, 512)
    z2 = z1.clone()

    loss = barlow_twins_loss(z1, z2)
    print(f"Loss for identical embeddings: {loss.item():.6f}")

    # Compare against random, unrelated embeddings — loss should
    # be noticeably higher.
    z1_random = torch.randn(32, 512)
    z2_random = torch.randn(32, 512)
    loss_random = barlow_twins_loss(z1_random, z2_random)
    print(f"Loss for random embeddings: {loss_random.item():.6f}")