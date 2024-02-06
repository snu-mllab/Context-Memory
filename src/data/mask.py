"""Utilities for attention mask generation."""

from typing import Optional, Tuple
import torch


def reverse_cumsum(x: torch.Tensor) -> torch.Tensor:
    """Cumulative sum from right to left.

    See https://github.com/pytorch/pytorch/issues/33520.

    Args:
        x: a tensor of shape (batch_size, seq_len)
    Returns:
        A tensor of shape (batch_size, seq_len) where each element is the sum of
        all elements to the right of it.
    """
    return x + torch.sum(x, dim=-1, keepdims=True) - torch.cumsum(x, dim=-1)


def get_comp_mask(
    inputs: torch.Tensor,
    comp_token: int,
    dtype=torch.float32,
) -> torch.Tensor:
    """Returns a mask where all tokens not COMP token are masked out.

      a C b C d
      0 1 0 1 0

    Args:
        inputs: an array of input tokens where the last dimension is the
            sequence length.
        comp_token: the integer id of the COMP token.
        dtype: the dtype of the mask, default int64.
    Returns:
        The requested mask.
    """
    if type(comp_token) == list:
        first_token = comp_token[0]
        last_token = comp_token[-1]
    else:
        first_token = last_token = comp_token

    mask1 = (inputs == first_token).cumsum(-1) - 1
    mask2 = (inputs == last_token).cumsum(-1) - (inputs == last_token).float()

    mask = (mask1 == mask2)

    return mask.to(dtype)


def get_last_comp_mask(
    inputs: torch.Tensor,
    comp_token: int,
    dtype=torch.float32,
) -> torch.Tensor:
    """Returns a mask where all tokens not the last COMP token are masked out.

      a C b C d
      0 0 0 1 0

    Args:
        inputs: an array of input tokens where the last dimension is the
            sequence length.
        comp_token: the integer id of the COMP token.
        dtype: the dtype of the mask, default int64.
    Returns:
        The requested mask.
    """
    if type(comp_token) == list:
        first_token = comp_token[0]
        last_token = comp_token[-1]
    else:
        first_token = last_token = comp_token

    mask1 = reverse_cumsum(inputs == last_token) - 1
    mask2 = reverse_cumsum(inputs == first_token) - (inputs == first_token).float()

    mask = (mask1 == 0) & (mask2 == 0)

    return mask.to(dtype)


def get_pos_attn_mask(
    inputs: torch.Tensor,
    comp_token: Optional[int] = None,
    pad_token: Optional[int] = None,
    sink_token: Optional[int] = None,
    dtype=torch.int64,
):
    """Creates a 4D pos control mask.
    Returns all ones (unaffected mask).

    Args:
        inputs: an array of shape (batch_size, seq_len) input tokens.
        comp_token: the integer id of the COMP token.
        pad_token: if supplied, mask out where inputs == pad_token.
        dtype: the dtype of the mask, default int64.
    Returns:
        The requested mask of shape (batch_size, 1, seq_len, seq_len)
    """
    del comp_token
    batch_size, seq_len = inputs.shape
    mask = torch.ones((batch_size, 1, seq_len, seq_len), dtype=torch.bool)

    if pad_token is not None:
        mask = mask & (inputs != pad_token)[:, None, None]
    return mask.type(dtype)


def get_comp_attn_mask_recur(
    inputs: torch.Tensor,
    comp_token: Optional[int],
    pad_token: Optional[int] = None,
    sink_token: Optional[int] = None,
    dtype=torch.int64,
) -> torch.Tensor:
    """Creates a 4D recurrent comp mask (last).
    Example, where C is the <COMP> token:

      a C b C d
    a 1 1 0 0 0
    C 1 1 0 0 0
    b 0 1 1 1 0
    C 0 1 1 1 0
    d 0 0 0 1 1

    Args:
        inputs: an array of shape (batch_size, seq_len) input tokens.
        comp_token: the integer id of the COMP token.
        pad_token: if supplied, mask out where inputs == pad_token.
        dtype: the dtype of the mask, default int64.
    Returns:
        The requested mask of shape (batch_size, 1, seq_len, seq_len)
    """
    if type(comp_token) == int:
        first_token = last_token = comp_token
    elif type(comp_token) == list:
        first_token = comp_token[0]
        last_token = comp_token[-1]

    # Make block-diagonal matrix
    post_mask = reverse_cumsum(inputs == last_token)[:, None, None]
    mask = post_mask.permute((0, 1, 3, 2)) == post_mask

    x = inputs == first_token
    pre_mask = torch.sum(x, dim=-1, keepdims=True) - torch.cumsum(x, dim=-1)
    mask2 = post_mask.permute((0, 1, 3, 2)) == pre_mask[:, None, None]

    mask = mask | mask2

    assert mask.dtype == torch.bool

    if pad_token is not None:
        mask = mask & (inputs != pad_token)[:, None, None]
    if sink_token is not None:
        mask = mask | (inputs == sink_token)[:, None, None]

    return mask.type(dtype)


def get_comp_attn_mask_concat_recur(
    inputs: torch.Tensor,
    comp_token: Optional[int],
    pad_token: Optional[int] = None,
    sink_token: Optional[int] = None,
    dtype=torch.int64,
) -> torch.Tensor:
    """Creates a 4D concat recurrent comp mask. (block-wise attention + <COMP> token attention)
    Example, where C is the <COMP> token: 

      a C b C d
    a 1 1 0 1 0
    C 1 1 0 1 0
    b 0 1 1 1 0
    C 0 1 1 1 0
    d 0 1 0 1 1

    Args:
        inputs: an array of shape (batch_size, seq_len) input tokens.
        comp_token: the integer id of the COMP token.
        pad_token: if supplied, mask out where inputs == pad_token.
        dtype: the dtype of the mask, default int64.
    Returns:
        The requested mask of shape (batch_size, 1, seq_len, seq_len)
    """
    if type(comp_token) == int:
        last_token = comp_token
    elif type(comp_token) == list:
        last_token = comp_token[-1]

    # Make block-diagonal matrix
    post_mask = reverse_cumsum(inputs == last_token)[:, None, None]
    mask = post_mask.permute((0, 1, 3, 2)) == post_mask

    comp_mask = get_comp_mask(inputs, comp_token, dtype=torch.bool)[:, None, None]
    mask = mask | comp_mask

    assert mask.dtype == torch.bool

    if pad_token is not None:
        mask = mask & (inputs != pad_token)[:, None, None]
    if sink_token is not None:
        mask = mask | (inputs == sink_token)[:, None, None]

    return mask.type(dtype)


def get_comp_attn_mask_concat(
    inputs: torch.Tensor,
    comp_token: Optional[int],
    pad_token: Optional[int] = None,
    sink_token: Optional[int] = None,
    dtype=torch.int64,
) -> torch.Tensor:
    """Creates a 4D concat comp mask. (only input attend the <COMP> token)
    Example, where C is the <COMP> token and I is the input text token:

      a C b C I
    a 1 1 0 0 0
    C 1 1 0 0 0
    b 0 0 1 1 0
    C 0 0 1 1 0
    I 0 1 0 1 1

    Args:
        inputs: an array of shape (batch_size, seq_len) input tokens.
        comp_token: the integer id of the COMP token.
        pad_token: if supplied, mask out where inputs == pad_token.
        dtype: the dtype of the mask, default int64.
    Returns:
        The requested mask of shape (batch_size, 1, seq_len, seq_len)
    """
    if type(comp_token) == int:
        last_token = comp_token
    elif type(comp_token) == list:
        last_token = comp_token[-1]

    # Make block-diagonal matrix
    post_mask = reverse_cumsum(inputs == last_token)[:, None, None]
    mask = post_mask.permute((0, 1, 3, 2)) == post_mask

    comp_mask = get_comp_mask(inputs, comp_token, dtype=torch.bool)[:, None, None]
    comp_attn = (post_mask == 0).permute((0, 1, 3, 2)) & comp_mask

    mask = mask | comp_attn

    assert mask.dtype == torch.bool

    if pad_token is not None:
        mask = mask & (inputs != pad_token)[:, None, None]
    if sink_token is not None:
        mask = mask | (inputs == sink_token)[:, None, None]

    return mask.type(dtype)


def get_comp_attn_mask_merge(
    inputs: torch.Tensor,
    comp_token: Optional[int],
    pad_token: Optional[int] = None,
    sum_token: Optional[int] = None,
    sink_token: Optional[int] = None,
    dtype=torch.int64,
) -> torch.Tensor:
    """Creates a 4D merge comp mask. (only input attend the merged <COMP> token)
    Example, where C is the <COMP> token, S is the SUM token, I is the input text token:

      a C b C S I
    a 1 1 0 0 0 0
    C 1 1 0 0 0 0
    b 0 0 1 1 0 0
    C 0 0 1 1 0 0
    S 0 1 0 1 1 0
    I 0 0 0 0 1 1

    Args:
        inputs: an array of shape (batch_size, seq_len) input tokens.
        comp_token: the integer id of the COMP token.
        pad_token: if supplied, mask out where inputs == pad_token.
        dtype: the dtype of the mask, default int64.
    Returns:
        The requested mask of shape (batch_size, 1, seq_len, seq_len)
    """
    if type(comp_token) == int:
        last_token = comp_token
    elif type(comp_token) == list:
        last_token = comp_token[-1]

    assert sum_token is not None, "sum_token must be provided"

    # Make block-diagonal matrix
    post_mask = reverse_cumsum(inputs == last_token)[:, None, None]
    mask = post_mask.permute((0, 1, 3, 2)) == post_mask

    # Make Inputs attend sum token, and sum token not attend inputs
    mask_sum = get_comp_attn_mask_recur(inputs, sum_token, dtype=torch.bool)
    mask = mask & mask_sum

    comp_mask = get_comp_mask(inputs, comp_token, dtype=torch.bool)[:, None, None]
    sum_mask = get_comp_mask(inputs, sum_token, dtype=torch.bool)[:, None, None]

    comp_attn = sum_mask.permute((0, 1, 3, 2)) & comp_mask
    mask = mask | comp_attn

    assert mask.dtype == torch.bool

    if pad_token is not None:
        mask = mask & (inputs != pad_token)[:, None, None]
    if sink_token is not None:
        mask = mask | (inputs == sink_token)[:, None, None]

    return mask.type(dtype)


def cross_attention_comp_last(
    inputs: torch.Tensor,
    comp_token: int,
    pad_token: Optional[int] = None,
    sink_token: Optional[int] = None,
    dtype=torch.int64,
) -> torch.Tensor:
    """Returns a mask where all tokens prior to the last COMP token are masked out.
    Args:
        inputs: an array of input tokens where the last dimension is the
            sequence length.
        comp_token: the integer id of the COMP token.
        pad_token: if supplied, mask out where inputs == pad_token.
        dtype: the dtype of the mask, default int64.
    Returns:
        The requested mask.
    """
    if type(comp_token) == list:
        comp_token = comp_token[0]

    mask = (inputs == comp_token).cumsum(-1)
    mask = (mask == mask[:, -1:])

    if pad_token is not None:
        mask = mask & (inputs != pad_token)[:, None, None]
    if sink_token is not None:
        mask = mask | (inputs == sink_token)[:, None, None]
    return mask.type(dtype)


def cross_attention_comp_concat(
    inputs: torch.Tensor,
    comp_token: int,
    pad_token: Optional[int] = None,
    sink_token: Optional[int] = None,
    dtype=torch.int64,
) -> torch.Tensor:
    """Returns a mask where text tokens prior to the last COMP token are masked out.
    Args:
        inputs: an array of input tokens where the last dimension is the
            sequence length.
        comp_token: the integer id of the COMP token.
        pad_token: if supplied, mask out where inputs == pad_token.
        dtype: the dtype of the mask, default int64.
    Returns:
        The requested mask.
    """
    mask = cross_attention_comp_last(inputs, comp_token, dtype=torch.bool)
    comp_mask = get_comp_mask(inputs, comp_token, dtype=torch.bool)

    mask = mask | comp_mask

    if pad_token is not None:
        mask = mask & (inputs != pad_token)[:, None, None]
    if sink_token is not None:
        mask = mask | (inputs == sink_token)[:, None, None]
    return mask.type(dtype)


if __name__ == '__main__':
    comp_token = [32000, 32001]
    sum_token = [32002, 32003]

    pad_token = 0
    inputs = torch.tensor([
        [0, 1, 2, 4, 5, 32000, 32001, 32002, 32003, 7, 8, 9, 0],
        [0, 1, 32000, 32001, 32002, 32003, 4, 5, 32000, 32001, 32002, 32003, 0],
    ])

    mask = get_pos_attn_mask(inputs, comp_token, pad_token=pad_token)
    print("Positive")
    print(mask)

    mask = get_comp_mask(inputs, comp_token)
    print("Comp-mask")
    print(mask)
    mask = get_comp_attn_mask_concat(inputs, comp_token, pad_token=pad_token)
    print("Comp-concat")
    print(mask)
    mask = get_comp_attn_mask_concat_recur(inputs, comp_token, pad_token=pad_token)
    print("Comp-concat-recur")
    print(mask)
    mask = cross_attention_comp_concat(inputs, comp_token, pad_token=pad_token)
    print("Comp-cross-attend-concat")
    print(mask)

    mask = get_comp_attn_mask_merge(inputs, comp_token, pad_token=pad_token, sum_token=sum_token)
    print("Comp-merge")
    print(mask)
    mask = get_comp_attn_mask_recur(inputs, sum_token, pad_token=pad_token)
    print("Comp-merge-recur")
    print(mask)
    mask = cross_attention_comp_last(inputs, comp_token, pad_token=pad_token)
    print("Comp-cross-attend")
    print(mask)
