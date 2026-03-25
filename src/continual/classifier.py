"""Cosine linear classifier head for L2-normalised face embeddings."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class CosineLinear(nn.Module):
    """
    Cosine similarity classifier: logits[c] = sigma * cos(x, w_c) for L2-normalised x and w_c.

    Starts with zero classes (`weight` is None); call `expand` to add output dimensions.
    New rows can be initialised from the mean direction of exemplar embeddings.
    """

    def __init__(
        self,
        in_features: int = 128,
        out_features: int = 0,
        *,
        learnable_scale: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        if out_features < 0:
            raise ValueError("out_features must be >= 0")
        if out_features > 0:
            self.weight = nn.Parameter(torch.empty(out_features, in_features))
            self._init_weight_uniform(self.weight)
        else:
            self.register_parameter("weight", None)

        if learnable_scale:
            self.sigma = nn.Parameter(torch.ones(1))
        else:
            self.register_buffer("sigma", torch.ones(1), persistent=False)

    @staticmethod
    def _init_weight_uniform(weight: torch.Tensor) -> None:
        stdv = 1.0 / math.sqrt(weight.size(1))
        nn.init.uniform_(weight, -stdv, stdv)

    @property
    def out_features(self) -> int:
        if self.weight is None:
            return 0
        return int(self.weight.shape[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight is None or self.weight.shape[0] == 0:
            raise RuntimeError("CosineLinear has no classes; call expand() first.")
        if x.dim() != 2 or x.shape[1] != self.in_features:
            raise ValueError(
                f"Expected x of shape (N, {self.in_features}), got {tuple(x.shape)}"
            )
        x_n = F.normalize(x, p=2, dim=1, eps=1e-12)
        w_n = F.normalize(self.weight, p=2, dim=1, eps=1e-12)
        return self.sigma * F.linear(x_n, w_n)

    def expand(
        self,
        n_new_classes: int,
        init_from_embeddings: Optional[torch.Tensor | np.ndarray] = None,
    ) -> None:
        """
        Append `n_new_classes` rows to the weight matrix.

        If `n_new_classes == 1` and `init_from_embeddings` is provided with shape (N, D),
        the new row is the L2-normalised mean embedding. Otherwise new rows use uniform
        init (same scheme as `_init_weight_uniform`). For `n_new_classes > 1`, embeddings
        init is not supported and `init_from_embeddings` must be None.
        """
        if n_new_classes < 1:
            raise ValueError("n_new_classes must be >= 1")
        if n_new_classes > 1 and init_from_embeddings is not None:
            raise ValueError(
                "init_from_embeddings is only supported when n_new_classes == 1"
            )

        device = self.sigma.device
        dtype = self.sigma.dtype
        new_rows = torch.empty(n_new_classes, self.in_features, device=device, dtype=dtype)

        if init_from_embeddings is not None:
            if n_new_classes != 1:
                raise RuntimeError("internal: init requires n_new_classes == 1")
            emb = torch.as_tensor(init_from_embeddings, device=device, dtype=torch.float32)
            if emb.ndim != 2 or emb.shape[1] != self.in_features:
                raise ValueError(
                    "init_from_embeddings must have shape (N, in_features), got "
                    f"{tuple(emb.shape)}"
                )
            with torch.no_grad():
                direction = emb.mean(dim=0)
                direction = F.normalize(direction, p=2, dim=0, eps=1e-12)
                new_rows[0].copy_(direction.to(dtype=dtype))
        else:
            self._init_weight_uniform(new_rows)

        if self.weight is None:
            self.weight = nn.Parameter(new_rows)
        else:
            combined = torch.cat([self.weight.data, new_rows], dim=0)
            self.weight = nn.Parameter(combined)
