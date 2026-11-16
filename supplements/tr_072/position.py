"""I-authored supplemental equations, not external-generated architecture output.
CPU eval only. Explicit positions avoid inventing upstream padding/index conventions.
"""
import math
import torch


def _check(words, positions):
    if words.device.type != 'cpu' or words.dtype not in (torch.float32, torch.float64):
        raise ValueError('CPU float32/float64 required')
    if words.ndim != 3 or min(words.shape) <= 0 or not torch.isfinite(words).all():
        raise ValueError('finite nonempty B,T,D words required')
    if positions.device.type != 'cpu' or positions.dtype != torch.int64 or positions.shape != words.shape[:2]:
        raise ValueError('explicit CPU int64 B,T positions required')
    if (positions < 0).any():
        raise ValueError('positions must be nonnegative')


def convs2s_position(words, positions, table):
    """words are token embeddings; table rows are explicit caller position IDs."""
    _check(words, positions)
    if table.ndim != 2 or table.shape[1] != words.shape[-1] or table.dtype != words.dtype or table.device != words.device:
        raise ValueError('position table must match embedding width, dtype and device')
    if not torch.isfinite(table).all() or (positions >= table.shape[0]).any():
        raise ValueError('finite table and in-range IDs required')
    return words + table[positions]


def transformer_position(words, positions):
    """Unscaled token vectors -> sqrt(D)*words + PE. Even D selected."""
    _check(words, positions)
    d = words.shape[-1]
    if d % 2:
        raise ValueError('selected sinusoidal variant requires even D')
    exponent = torch.arange(0, d, 2, dtype=words.dtype) / d
    angles = positions.to(words.dtype).unsqueeze(-1) / torch.pow(10000.0, exponent)
    pe = torch.stack((torch.sin(angles), torch.cos(angles)), dim=-1).flatten(-2)
    return math.sqrt(d) * words + pe


def convs2s_forward(block, source_words, source_positions, source_table,
                    target_words, target_positions, target_table, source_padding_mask):
    # Decoder tokens/positions already shifted by caller. Mask goes unchanged to block.
    return block(convs2s_position(source_words, source_positions, source_table),
                 convs2s_position(target_words, target_positions, target_table), source_padding_mask)


def transformer_forward(block, words, positions, exclusion_mask):
    return block(transformer_position(words, positions), exclusion_mask)
