import sys, itertools, torch, math
sys.path.insert(0, ".")
from grow.field import bayes_argmax, PROBIT
torch.manual_seed(0); worse = 0; gap = []
for t in range(300):
    D, L = 2, 5
    a = torch.randn(1, D, L) * 1.5; b = torch.randn(1, D, L); v = torch.rand(1, D, L) * 2; M = torch.ones(1, D, L, dtype=torch.bool)
    beta = torch.randn(1) * 3
    x = bayes_argmax(a, b, v, M, beta)
    def f(xx):
        xf = xx.float(); xr = torch.nn.functional.pad(xf, (0, 1))[..., 1:]
        return ((beta + (xf * a).sum() + (xf * xr * b).sum()) / torch.sqrt(1 + PROBIT * (xf * v).sum())).item()
    best = max(f(torch.tensor(bits).reshape(1, D, L)) for bits in itertools.product([0, 1], repeat=D * L))
    g = best - f(x); gap.append(g); worse += g > 1e-5
print("bayes_argmax suboptimal in %d/300 cases, mean gap %.4f, max gap %.4f" % (worse, sum(gap) / 300, max(gap)))
