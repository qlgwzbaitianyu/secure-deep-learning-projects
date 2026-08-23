from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SecureSMC:
    """Two-share secure arithmetic used by the PCNN experiments.

    Algorithmic parameters are supplied externally through configuration.
    """

    bit_width: int = 8
    rounding_offset: float = 0.5
    signed: bool = True
    random_seed: int | None = None

    def __post_init__(self) -> None:
        if self.bit_width < 2:
            raise ValueError("bit_width must be at least 2")
        if not self.signed:
            raise ValueError("SecureSMC currently requires signed arithmetic")
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        if not 0 <= self.rounding_offset < 1:
            raise ValueError("rounding_offset must be in [0, 1)")

    @property
    def share_upper_bound(self) -> int:
        return 2 ** (self.bit_width - 1)

    @property
    def mask(self) -> int:
        return 2 ** self.bit_width - 1

    @property
    def signed_max(self) -> int:
        return 2 ** (self.bit_width - 1) - 1

    def _random_share(self, shape: Any = None) -> np.ndarray | int:
        return np.random.randint(0, self.share_upper_bound, size=shape)

    def SecBitMul(self, x_1, x_2, y_1, y_2):
        a_1 = self._random_share()
        a = self._random_share()
        b_1 = self._random_share()
        b = self._random_share()
        a_2 = a_1 ^ a
        b_2 = b_1 ^ b

        c = a & b
        c_1 = self._random_share()
        c_2 = c ^ c_1

        alpha = (x_1 ^ a_1) ^ (x_2 ^ a_2)
        beta = (y_1 ^ b_1) ^ (y_2 ^ b_2)

        f_1 = c_1 ^ (b_1 & alpha) ^ (a_1 & beta)
        f_2 = c_2 ^ (b_2 & alpha) ^ (a_2 & beta) ^ (alpha & beta)
        return f_1, f_2

    def SecBitAdd(self, x_1, x_2, y_1, y_2):
        u_1 = x_1 ^ y_1
        u_2 = x_2 ^ y_2
        c_1, c_2 = self.SecBitMul(x_1, x_2, y_1, y_2)
        c_1 <<= 1
        c_2 <<= 1

        while (c_1 ^ c_2) != 0:
            t_1 = u_1 ^ c_1
            t_2 = u_2 ^ c_2
            c_1, c_2 = self.SecBitMul(u_1, u_2, c_1, c_2)
            c_1 = c_1 << 1 & self.mask
            c_2 = c_2 << 1 & self.mask
            u_1 = t_1 & self.mask
            u_2 = t_2 & self.mask

        if u_1 > self.signed_max:
            u_1 = ~(u_1 ^ self.mask)
        if u_2 > self.signed_max:
            u_2 = ~(u_2 ^ self.mask)
        return u_1, u_2

    def SecBitExtra(self, u_1, u_2):
        r_1 = self._random_share()
        r_2 = self._random_share()
        r = r_1 ^ r_2

        s_1 = self._random_share()
        s_2 = r - s_1
        v = (u_1 - s_1) + (u_2 - s_2)
        if v < 0:
            v = round(v - self.rounding_offset)
        v = round(v)

        v_1 = self._random_share()
        v_2 = v ^ v_1
        uu_1, uu_2 = self.SecBitAdd(v_1, v_2, r_1, r_2)

        msb_1 = int(uu_1 < 0)
        msb_2 = int(uu_2 < 0)
        return uu_1, uu_2, msb_1, msb_2

    def SecMaxIndex(self, X, Y):
        x, y = 0, 0
        if ((X[x, y] - X[0, 1]) + (Y[x, y] - Y[0, 1])) < 0:
            x, y = 0, 1
        if ((X[x, y] - X[1, 0]) + (Y[x, y] - Y[1, 0])) < 0:
            x, y = 1, 0
        if ((X[x, y] - X[1, 1]) + (Y[x, y] - Y[1, 1])) < 0:
            x, y = 1, 1
        return x, y
