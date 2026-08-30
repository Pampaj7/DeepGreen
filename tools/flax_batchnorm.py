# -*- coding: utf-8 -*-
"""Give flax's batch normalisation torch's averaging, not its own.

flax and torch both call the parameter `momentum` and mean opposite things by
it:

    flax   ra = momentum * ra + (1 - momentum) * batch      # retention
    torch  ra = (1 - momentum) * ra + momentum * batch      # update

so flax's `m` is torch's `1 - m`. torch's default of 0.1 keeps 90% of the
running average at each step. `flaxmodels` writes `momentum=0.1` at every one of
its eight BatchNorm sites, which in flax's convention keeps 10% and takes the
other 90% from the current batch -- the running statistics track the last batch
rather than the run.

That is not a small difference at evaluation time, which is where the running
statistics are used. One epoch of Tiny ImageNet, every stack training the same
architecture from the same distribution on the same data: six stacks landed
between 8.96% and 10.26% accuracy with a test loss of 4.31-4.42, and JAX landed
at 7.40% with a test loss of 4.68. It was the only stack whose normalisation
statistics were, in effect, one batch old.

Keras has the same trap in the other direction -- its default of 0.99 is a
retention factor too, and `tools/torch_init.py`'s companion fix sets 0.9 there.
Three libraries, three conventions, one specification clause.

`flaxmodels` passes the value explicitly, so a changed default would not reach
it. This replaces the class instead, which affects `flax.linen.BatchNorm` for
the whole process -- acceptable because a run trains one model and nothing else
in it uses flax.
"""

from __future__ import annotations

#: torch's momentum=0.1 expressed in flax's convention.
TORCH_EQUIVALENT_MOMENTUM = 0.9


def torchvision_kernel_init():
    """`kaiming_normal_(mode='fan_out', nonlinearity='relu')`, for flax.

    flax defaults every Conv and Dense to `lecun_normal` -- variance scaling on
    fan_in with a gain of 1 -- which for ResNet-18's 7x7 stem gives a standard
    deviation of 0.0835 against torchvision's 0.0253, 3.3x wider. The same class
    of divergence as Deeplearning4j's 4.6x, and the last of the four stacks that
    were building the right architecture from the wrong distribution.

    Dispatches on rank, because `flaxmodels` threads one `kernel_init` through
    every layer including the final Dense -- and torch does not initialise a
    Linear the way it initialises a Conv. Passing kaiming fan_out to everything
    gave the 512 -> 200 classifier a standard deviation of sqrt(2/200) = 0.100
    where torch's nn.Linear default is uniform on +/-1/sqrt(512), about 0.026:
    four times too wide, on the layer whose output is the loss.

      rank 4 (convolution): normal, sqrt(2 / fan_out)
      rank 2 (dense):       uniform on +/-1/sqrt(fan_in), which is what
                            kaiming_uniform_(a=sqrt(5)) reduces to
    """
    import jax
    import numpy as _np

    def init(key, shape, dtype="float32"):
        shape = tuple(int(d) for d in shape)
        if len(shape) == 4:
            # flax kernels are (kh, kw, in, out); fan_out is out * kh * kw
            fan_out = shape[3] * shape[0] * shape[1]
            return jax.random.normal(key, shape, dtype) * _np.sqrt(2.0 / fan_out)
        if len(shape) == 2:
            bound = 1.0 / _np.sqrt(shape[0])       # (in, out)
            return jax.random.uniform(key, shape, dtype, -bound, bound)
        return jax.random.normal(key, shape, dtype) * 0.01

    return init


def patch_flax_batchnorm(momentum: float = TORCH_EQUIVALENT_MOMENTUM) -> int:
    """Force every flax BatchNorm to `momentum`. Returns 1 if it took effect.

    Returns rather than asserts so the caller can refuse a silent no-op -- the
    failure mode a patch like this has, and one this repository has already had
    once, when an initialiser helper matched zero of 102 weight tensors and said
    nothing about it.
    """
    from flax import linen as nn

    if getattr(nn.BatchNorm, "_deepgreen_patched", False):
        return 1

    class _TorchMomentumBatchNorm(nn.BatchNorm):
        _deepgreen_patched = True

        def __post_init__(self):
            # Modules are frozen dataclasses, so the field is rewritten before
            # the parent finishes initialising it.
            object.__setattr__(self, "momentum", momentum)
            super().__post_init__()

    _TorchMomentumBatchNorm.__name__ = "BatchNorm"
    _TorchMomentumBatchNorm.__qualname__ = "BatchNorm"
    nn.BatchNorm = _TorchMomentumBatchNorm
    return 1 if getattr(nn.BatchNorm, "_deepgreen_patched", False) else 0
