"""Version-checked scratch-layout optimization; no gate or kernel arithmetic change."""

import functools
import hashlib
import inspect

ALLOCATOR_SHA256 = "f76efb72d37e7fd82ac239494b27ed9d2e5e4a4eea55c78ed9ffb860b89215d7"


def capacity(rows):
    if not isinstance(rows, int) or rows < 0:
        raise ValueError("Expected nonnegative integer row capacity")
    return 0 if rows == 0 else 1 << (rows - 1).bit_length()


def make_allocator(original, uint32_dtype):
    @functools.wraps(original)
    def allocate(offset, shape, dtype, device, all_gather):
        if not all_gather and dtype is uint32_dtype and len(shape) == 2:
            shape = (shape[0], capacity(shape[1]))
        return original(offset, shape, dtype, device, all_gather)

    allocate._eldr_bucketed_scratch = True
    allocate._eldr_original_allocator = original
    return allocate


def install():
    import importlib

    import torch

    module = importlib.import_module("triton_kernels.topk")
    original = module.make_empty
    if getattr(original, "_eldr_bucketed_scratch", False):
        return original._eldr_original_allocator
    if (
        hashlib.sha256(inspect.getsource(original).encode()).hexdigest()
        != ALLOCATOR_SHA256
    ):
        raise ValueError("Unrecognized top-k allocator; refuse runtime replacement")
    module.make_empty = make_allocator(original, torch.uint32)
    return original
