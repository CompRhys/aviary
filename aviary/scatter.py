import torch


def scatter_reduce(src, index, dim=-1, dim_size=None, reduce="sum"):
    """Performs a scatter-reduce operation on the input tensor.

    This function scatters the elements from the source tensor (src) into a new tensor
    of shape determined by dim_size along the specified dimension (dim), using the
    given reduction method. It's compatible with autograd for gradient computation.

    NOTE this function was written by Claude 3.5 Sonnet.

    Args:
        src (torch.Tensor): The source tensor.
        index (torch.Tensor): The indices of elements to scatter. Must be 1D or have
            the same number of dimensions as src.
        dim (int, optional): The axis along which to index. Defaults to -1.
        dim_size (int, optional): The size of the output tensor's dimension `dim`.
            If None, it's inferred as index.max().item() + 1. Defaults to None.
        reduce (str, optional): The reduction operation to perform.
            Options: "sum", "mean", "amax", "max", "amin", "min", "prod".
            Defaults to "sum".

    Returns:
        torch.Tensor: The output tensor after the scatter-reduce operation.

    Raises:
        ValueError: If an unsupported reduction method is specified.
        RuntimeError: If index and src tensors are incompatible.

    Example:
        >>> src = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> index = torch.tensor([0, 1, 0, 1, 2])
        >>> scatter_reduce(src, index, dim=0, reduce="sum")
        tensor([4., 6., 5.])
    """
    if dim_size is None:
        dim_size = index.max().item() + 1

    # Prepare the output tensor shape
    shape = list(src.shape)
    shape[dim] = dim_size

    # Ensure index has the same number of dimensions as src
    if index.dim() != src.dim():
        if index.dim() != 1:
            raise RuntimeError(
                "Index tensor must be 1D or have the same number of dimensions "
                f"as src tensor. {index.shape=} != {src.shape=}"
            )
        # Expand index to match src dimensions
        repeat_shape = [1] * src.dim()
        repeat_shape[dim] = src.size(dim)
        index = index.view(-1, *[1] * (src.dim() - 1)).expand_as(src)

    # Perform scatter_reduce operation
    if reduce in ["sum", "mean"]:
        out = torch.zeros(shape, dtype=src.dtype, device=src.device)
        out = out.scatter_add(dim, index, src)
        if reduce == "mean":
            count = torch.zeros(shape, dtype=src.dtype, device=src.device)
            count = count.scatter_add(dim, index, torch.ones_like(src))
            out = out / (count + (count == 0).float())  # avoid division by zero
    elif reduce in ["amax", "max"]:
        out = torch.full(shape, float("-inf"), dtype=src.dtype, device=src.device)
        out = torch.max(out, out.scatter(dim, index, src))
    elif reduce in ["amin", "min"]:
        out = torch.full(shape, float("inf"), dtype=src.dtype, device=src.device)
        out = torch.min(out, out.scatter(dim, index, src))
    elif reduce == "prod":
        out = torch.ones(shape, dtype=src.dtype, device=src.device)
        out = out.scatter(dim, index, src, reduce="multiply")
    else:
        raise ValueError(f"Unsupported reduction method: {reduce}")

    return out
