import math
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import compile as a3
import torch.nn.functional as F


@dataclass
class C1:
    vocab_size: int
    block_size: int
    n_embd: int
    n_head: int
    n_layer: int
    dropout: float = 0.0
    epsilon: float = 1e-5
    share_scale: float = 1.0
    approximation_scale: float = 8.0
    seed: int | None = None


class S1:
    def __init__(self, c):
        self.c = c
        if c.seed is not None:
            random.seed(c.seed)
            np.random.seed(c.seed)
            torch.manual_seed(c.seed)
        self.r = {}

    def split(self, x):
        y = torch.randn_like(x) * self.c.share_scale
        return x - y, y

    def combine(self, x1, x2):
        return x1 + x2

    def add(self, a1, a2, b1, b2):
        return a1 + b1, a2 + b2

    def sub(self, a1, a2, b1, b2):
        return a1 - b1, a2 - b2

    def mul_plain(self, a1, a2, w):
        return a1 * w, a2 * w

    def triples(self, shape, dtype, device):
        a = torch.randn(shape, dtype=dtype, device=device)
        b = torch.randn(shape, dtype=dtype, device=device)
        c = a * b
        a1, a2 = self.split(a)
        b1, b2 = self.split(b)
        c1, c2 = self.split(c)
        return (a1, a2), (b1, b2), (c1, c2)

    def mul(self, a1, a2, b1, b2):
        x = a1 + a2
        y = b1 + b2
        z = x * y
        z1, z2 = self.split(z)
        return z1, z2

    def matmul(self, a1, a2, b1, b2):
        x = torch.matmul(a1, b1) + torch.matmul(a1, b2) + torch.matmul(a2, b1) + torch.matmul(a2, b2)
        z1, z2 = self.split(x)
        return z1, z2

    def dot(self, a1, a2, b1, b2):
        return self.matmul(a1, a2, b1, b2)

    def square(self, a1, a2):
        return self.mul(a1, a2, a1, a2)

    def reciprocal(self, x):
        y = x + self.c.epsilon
        return 1.0 / y

    def rsqrt(self, x):
        y = torch.clamp(x, min=self.c.epsilon)
        return torch.rsqrt(y)

    def mean(self, a1, a2, dim=-1, keepdim=True):
        n = a1.shape[dim]
        return (a1.sum(dim=dim, keepdim=keepdim) / n, a2.sum(dim=dim, keepdim=keepdim) / n)

    def softmax_poly(self, a1, a2, dim=-1):
        x1 = a1 / self.c.approximation_scale
        x2 = a2 / self.c.approximation_scale
        y1 = 0.5 + 1.20096 * x1 - 0.81562 * torch.pow(x1, 3)
        y2 = 0.5 + 1.20096 * x2 - 0.81562 * torch.pow(x2, 3)
        q = y1 + y2
        q = torch.clamp(q, min=self.c.epsilon)
        q = q / q.sum(dim=dim, keepdim=True)
        z1, z2 = self.split(q)
        return z1, z2

    def gelu_poly(self, a1, a2):
        x = a1 + a2
        u = 0.7978845608 * (x + 0.044715 * torch.pow(x, 3))
        y = 0.5 * x * (1.0 + torch.tanh(u))
        z1, z2 = self.split(y)
        return z1, z2

    def layer_norm(self, a1, a2, w1, w2, b1, b2):
        m1, m2 = self.mean(a1, a2, dim=-1, keepdim=True)
        d1, d2 = self.sub(a1, a2, m1, m2)
        q1, q2 = self.square(d1, d2)
        v1, v2 = self.mean(q1, q2, dim=-1, keepdim=True)
        v = torch.clamp(v1 + v2, min=self.c.epsilon)
        s = torch.rsqrt(v)
        n1 = d1 * s
        n2 = d2 * s
        y1, y2 = self.add(n1, n2, w1, w2)
        z1, z2 = self.add(y1, y2, b1, b2)
        return z1, z2

    def causal_mask(self, t, device, dtype):
        return torch.tril(torch.ones((t, t), device=device, dtype=dtype))


class E1:
    def __init__(self, c, s):
        self.c = c
        self.s = s
        self.w1 = torch.empty(c.vocab_size, c.n_embd)
        self.w2 = torch.empty(c.block_size, c.n_embd)
        self.n = torch.empty(c.n_embd, c.n_embd)
        self.q = torch.empty(c.n_embd, c.n_embd)
        self.k = torch.empty(c.n_embd, c.n_embd)
        self.v = torch.empty(c.n_embd, c.n_embd)
        self.o = torch.empty(c.n_embd, c.n_embd)
        self.f = torch.empty(c.n_embd, 4 * c.n_embd)
        self.g = torch.empty(4 * c.n_embd, c.n_embd)
        self.l = torch.empty(c.n_embd, c.vocab_size)
        self._init()

    def _init(self):
        for x in (self.w1, self.w2, self.n, self.q, self.k, self.v, self.o, self.f, self.g, self.l):
            torch.nn.init.normal_(x, mean=0.0, std=0.02)

    def p(self, x):
        return self.s.split(x)

    def e(self, x):
        a1, a2 = self.p(self.w1[x])
        b1, b2 = self.p(self.w2[torch.arange(x.shape[1], device=x.device)[:x.shape[1]]])
        return a1 + b1, a2 + b2

    def linear(self, x1, x2, w):
        w1, w2 = self.p(w)
        return self.s.matmul(x1, x2, w1, w2)

    def ln(self, x1, x2):
        w1, w2 = self.p(torch.ones(self.c.n_embd, device=x1.device, dtype=x1.dtype))
        b1, b2 = self.p(torch.zeros(self.c.n_embd, device=x1.device, dtype=x1.dtype))
        return self.s.layer_norm(x1, x2, w1, w2, b1, b2)

    def attn(self, x1, x2):
        a1, a2 = self.s.layer_norm(x1, x2, *self.p(torch.ones(self.c.n_embd, device=x1.device, dtype=x1.dtype)), *self.p(torch.zeros(self.c.n_embd, device=x1.device, dtype=x1.dtype)))
        q1, q2 = self.linear(a1, a2, self.q)
        k1, k2 = self.linear(a1, a2, self.k)
        v1, v2 = self.linear(a1, a2, self.v)
        b, t, d = q1.shape
        h = self.c.n_head
        z = d // h
        q1 = q1.view(b, t, h, z).transpose(1, 2)
        q2 = q2.view(b, t, h, z).transpose(1, 2)
        k1 = k1.view(b, t, h, z).transpose(1, 2)
        k2 = k2.view(b, t, h, z).transpose(1, 2)
        v1 = v1.view(b, t, h, z).transpose(1, 2)
        v2 = v2.view(b, t, h, z).transpose(1, 2)
        qk1, qk2 = self.s.matmul(q1, q2, k1.transpose(-2, -1), k2.transpose(-2, -1))
        qk1 = qk1 / math.sqrt(z)
        qk2 = qk2 / math.sqrt(z)
        m = self.s.causal_mask(t, qk1.device, qk1.dtype).view(1, 1, t, t)
        qk1 = qk1.masked_fill(m == 0, 0.0)
        qk2 = qk2.masked_fill(m == 0, -1e4)
        a1, a2 = self.s.softmax_poly(qk1, qk2, dim=-1)
        y1, y2 = self.s.matmul(a1, a2, v1, v2)
        y1 = y1.transpose(1, 2).contiguous().view(b, t, d)
        y2 = y2.transpose(1, 2).contiguous().view(b, t, d)
        return self.linear(y1, y2, self.o)

    def mlp(self, x1, x2):
        y1, y2 = self.linear(x1, x2, self.f)
        y1, y2 = self.s.gelu_poly(y1, y2)
        return self.linear(y1, y2, self.g)

    def block(self, x1, x2):
        a1, a2 = self.attn(x1, x2)
        x1 = x1 + a1
        x2 = x2 + a2
        m1, m2 = self.ln(x1, x2)
        z1, z2 = self.mlp(m1, m2)
        return x1 + z1, x2 + z2

    def forward(self, x):
        x1, x2 = self.e(x)
        for _ in range(self.c.n_layer):
            x1, x2 = self.block(x1, x2)
        x1, x2 = self.ln(x1, x2)
        return self.linear(x1, x2, self.l)

    def secure_forward(self, x):
        if x.dtype not in (torch.int64, torch.int32):
            x = x.long()
        if x.dim() != 2:
            raise ValueError("x")
        if x.shape[1] > self.c.block_size:
            raise ValueError("x")
        return self.forward(x)


class G1(E1):
    def __init__(self, c, s):
        super().__init__(c, s)
        self.a = []
        self.b = []
        for _ in range(c.n_layer):
            self.a.append(torch.empty(c.n_embd, c.n_embd))
            self.b.append(torch.empty(c.n_embd, c.n_embd))

    def parameters_as_shares(self):
        z = {}
        for n, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                z[n] = self.s.split(v)
        return z

    def encode(self, x):
        x1, x2 = self.e(x)
        return x1, x2

    def attention_only(self, x1, x2):
        return self.attn(x1, x2)

    def mlp_only(self, x1, x2):
        return self.mlp(x1, x2)

    def secure_block(self, x1, x2):
        y1, y2 = self.attn(x1, x2)
        x1 = x1 + y1
        x2 = x2 + y2
        y1, y2 = self.ln(x1, x2)
        y1, y2 = self.mlp(y1, y2)
        return x1 + y1, x2 + y2

    def secure_gpt(self, x):
        x1, x2 = self.encode(x)
        for _ in range(self.c.n_layer):
            x1, x2 = self.secure_block(x1, x2)
        x1, x2 = self.ln(x1, x2)
        y1, y2 = self.linear(x1, x2, self.l)
        return y1, y2


class H1:
    def __init__(self, c):
        self.c = c
        self.s = S1(c)
        self.m = G1(c, self.s)

    def __call__(self, x):
        return self.m.secure_gpt(x)

    def combine(self, x1, x2):
        return self.s.combine(x1, x2)

    def split(self, x):
        return self.s.split(x)


__all__ = ["C1", "S1", "E1", "G1", "H1"]
