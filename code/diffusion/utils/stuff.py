import torch
import torch.nn.functional as f


def unsqueeze_n(tensor, n, dims=None):
    """Unsqueeze over multiple dimensions in one go."""
    if dims is None:
        dims = [-1] * n
    for _ in range(n):
        tensor = torch.unsqueeze(tensor, dim=dims.pop())
    return tensor


def cum_matmul(tensor, dim=0):
    """Cumulative matrix multiplication along a given dimension."""

    # pick the first element of the tensor along the given dimension
    cum_tensor = torch.index_select(tensor, dim, torch.tensor([0], device=tensor.device)).squeeze(0)
    # print(f"cum_tensor: {cum_tensor.shape}")
    for e_tensor in torch.split(tensor, 1, dim=dim)[1:]:  # torch.split returns views - faster
        # print(f"e_tensor: {e_tensor.shape}")
        cum_tensor = torch.matmul(cum_tensor, e_tensor.squeeze(0))

    return cum_tensor


def cat_dist(p, num_classes=-1):
    """Sample from a categorical distribution."""
    if len(p.size()) == 3:
        c = torch.stack(
            [
                torch.multinomial(
                    p[i], 1, replacement=True
                ).squeeze(-1) for i in range(p.size(0))
             ], dim=0
        )
    else:
        c = torch.multinomial(p, 1, replacement=True).squeeze(-1)
    o = f.one_hot(c, num_classes)
    return o
