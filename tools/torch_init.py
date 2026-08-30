# -*- coding: utf-8 -*-
"""Re-initialise a Keras model the way torchvision initialises its own.

Keras defaults every Conv2D and Dense to `glorot_uniform`; torchvision uses
`kaiming_normal_(mode='fan_out', nonlinearity='relu')` for convolutions and
leaves `nn.Linear` on its own default, which reduces to
U(-1/sqrt(fan_in), +1/sqrt(fan_in)). Those are different distributions, and the
difference is not cosmetic: holding framework, optimiser, learning rate and data
order fixed, the initialiser is what decides whether VGG-16 collapses to chance
at this learning rate -- 0 of 6 runs under He, 2 of 6 under Glorot, 4 of 6 under
Xavier.

The same correction was needed in Deeplearning4j, whose hand-rolled initialiser
was 4.6x wider than torchvision's. This stack was the other one still on its
framework's default, and one epoch of Fashion-MNIST showed it: 77.4% against
86-88% for the six stacks that were aligned.

Applied after construction rather than at layer definition, because the VGG-16
backbone comes from `keras.applications` and takes no initialiser argument.
"""

from __future__ import annotations

import numpy as np


def _fan_in_out(shape) -> tuple[float, float]:
    """torch's _calculate_fan_in_and_fan_out, for Keras's weight layout.

    Keras convolution kernels are (kh, kw, in, out) where torch's are
    (out, in, kh, kw), so the receptive field is the leading axes here and the
    trailing ones there. And torch does not divide by the stride, which is the
    trap that made every stride-2 convolution come out twice as wide when the
    same correction was made in Deeplearning4j.
    """
    if len(shape) == 2:                      # Dense: (in, out)
        return float(shape[0]), float(shape[1])
    receptive = float(np.prod(shape[:-2]))   # kh * kw
    return receptive * shape[-2], receptive * shape[-1]


def _leaves(model):
    """Every layer that owns its own weights.

    Leaves only. A container's `get_weights()` returns its children's tensors
    concatenated, so calling `set_weights` on it with one new kernel and zeros
    for the rest would rewrite the entire subtree -- zeroing every weight but
    the first. VGG-16 here is a Sequential wrapping the backbone as a nested
    model, so this is not hypothetical.
    """
    seen, stack, out = set(), [model], []
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        children = list(getattr(node, "layers", []) or [])
        if children:
            stack.extend(children)
        else:
            out.append(node)
    return out


def apply_torchvision_init(model, seed: int = 0) -> int:
    """Give every convolution and dense layer torchvision's initialisation.

    Duck-typed on the weight shapes rather than `isinstance`, because this stack
    uses two Keras implementations -- `tf_keras` 2.19 for one architecture and
    `tensorflow.keras` 3.15 for the other -- whose class hierarchies do not
    intersect: a tf_keras Conv2D is not an instance of a keras.layers.Conv2D,
    their bases being `Conv` and `BaseConv`. The first version of this function
    tested isinstance against Keras 3 and silently matched zero of 102 weight
    tensors on the Keras 2 model, which is the failure mode a helper like this
    has: it does nothing and says nothing.

    Returns the number of layers it initialised, so a caller can refuse a zero.
    """
    rng = np.random.default_rng(seed)
    touched = 0
    for layer in _leaves(model):
        weights = layer.get_weights()
        if not weights:
            continue
        kernel = weights[0]
        rank = len(kernel.shape)
        # A convolution kernel is rank 4 and a dense kernel rank 2; batch
        # normalisation's gamma is rank 1 and is left at its initial value, as
        # torch leaves BatchNorm weight at 1 and bias at 0.
        if rank == 4:
            _, fan_out = _fan_in_out(kernel.shape)
            new = rng.normal(0.0, np.sqrt(2.0 / fan_out), kernel.shape)
        elif rank == 2:
            fan_in, _ = _fan_in_out(kernel.shape)
            bound = 1.0 / np.sqrt(fan_in)
            new = rng.uniform(-bound, bound, kernel.shape)
        else:
            continue
        rest = [np.zeros_like(w) for w in weights[1:]]   # biases to zero, as torch does
        layer.set_weights([new.astype(kernel.dtype)] + rest)
        touched += 1
    return touched
