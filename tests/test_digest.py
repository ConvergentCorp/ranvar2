import ranvar as mc
import numpy as np
import pickle
import copy


def test_nActiveCount():
    """Tests that the frequency column adds up to the number of points added"""
    std = 100
    mu  = 100
    N   = 100
    data = np.random.randn(N)*std + mu
    x = mc.Digest(maxBins=16)
    for d in data:
        x.add(d)

    # print(x.nAc-tive)

    assert x.getActiveBinCount() == 16


def test_freqAddsUp():
    """Tests that the frequency column adds up to the number of points added"""
    std = 100
    mu  = 100
    N   = 100
    data = np.random.randn(N)*std + mu
    x = mc.Digest(maxBins=16)
    for d in data:
        x.add(d)

    # print(sum(x.freq))

    assert (sum(x.getWeights()) == N)


def test_normalApprox():
    """"""
    std = 100
    mu  = 100
    np.random.seed(31337)
    data = np.random.randn(10_000)*std + mu
    x = mc.Digest(maxBins=32)
    for d in data:
        x.add(d)


    prob1 = x.cdf(mu + std*1) - x.cdf(mu - std*1)
    prob2 = x.cdf(mu + std*2) - x.cdf(mu - std*2)
    prob3 = x.cdf(mu + std*3) - x.cdf(mu - std*3)

    # We are not using a lot of samples and keeping the accuracy bar low, otherwise
    # running the tests will take to long. In practice increasing the number of
    # sample points will increase the accuracy.

    print(prob1)
    print(prob2)
    print(prob3)

    assert abs(0.6827 - prob1) <= 1e-1
    assert abs(0.9545 - prob2) <= 1e-2
    assert abs(0.9973 - prob3) <= 1e-3

def test_normalApprox_quantile():
    """"""
    std = 100
    mu  = 100
    np.random.seed(31337)
    data = np.random.randn(10_000)*std + mu
    x = mc.Digest(maxBins=64)
    for d in data:
        x.add(d)

    DATA = [
        (-64.485, 0.05),
        (32.551, 0.25),
        (100.00, 0.50),
        (167.449, 0.75),
        (264.485, 0.95)
    ]

    for v, p in DATA:
        dv = abs((v - x.quantile(p)) / v)
        print(f'{v} : {x.quantile(p)} : {dv}')
        assert dv <= 2.5e-2



def test_ccdf():
    """Tests that ccdf(k) == 1 - cdf(k) for various points."""
    std = 100
    mu  = 100
    np.random.seed(31337)
    data = np.random.randn(10_000)*std + mu
    x = mc.Digest(maxBins=32)
    for d in data:
        x.add(d)

    # Test at various points including boundaries
    test_points = [mu - 2*std, mu - std, mu, mu + std, mu + 2*std]
    for k in test_points:
        assert abs(x.ccdf(k) - (1.0 - x.cdf(k))) < 1e-12

    # Boundary conditions
    assert x.ccdf(x.lower() - 1) == 1.0
    assert x.ccdf(x.upper() + 1) == 0.0


def test_dcdf():
    """Tests dcdf by verifying it approximates a numerical derivative of cdf."""
    std = 100
    mu  = 100
    np.random.seed(31337)
    data = np.random.randn(10_000)*std + mu
    x = mc.Digest(maxBins=32)
    for d in data:
        x.add(d)

    # The CDF is piecewise linear, so the derivative should match
    # a numerical finite difference within a single segment.
    h = 0.001
    test_points = [mu - std, mu, mu + std]
    for k in test_points:
        numerical_deriv = (x.cdf(k + h) - x.cdf(k - h)) / (2 * h)
        analytical_deriv = x.dcdf(k)
        assert abs(numerical_deriv - analytical_deriv) < 1e-4, \
            f"At k={k}: numerical={numerical_deriv}, analytical={analytical_deriv}"

    # Derivative should be non-negative (CDF is non-decreasing)
    for k in test_points:
        assert x.dcdf(k) >= 0

    # Outside the range, derivative should be zero
    assert x.dcdf(x.lower() - 1) == 0.0
    assert x.dcdf(x.upper() + 1) == 0.0


def test_dccdf():
    """Tests dccdf by verifying dccdf(k) == -dcdf(k)."""
    std = 100
    mu  = 100
    np.random.seed(31337)
    data = np.random.randn(10_000)*std + mu
    x = mc.Digest(maxBins=32)
    for d in data:
        x.add(d)

    test_points = [mu - std, mu, mu + std]
    for k in test_points:
        assert abs(x.dccdf(k) - (-x.dcdf(k))) < 1e-12

    # CCDF derivative should be non-positive
    for k in test_points:
        assert x.dccdf(k) <= 0

    # Outside the range, derivative should be zero
    assert x.dccdf(x.lower() - 1) == 0.0
    assert x.dccdf(x.upper() + 1) == 0.0


def test_pickle():
    std = 100
    mu  = 100
    np.random.seed(31337)
    data = np.random.randn(10_000)*std + mu
    x = mc.Digest(maxBins=64)
    for d in data:
        x.add(d)

    out = pickle.dumps(x)

    x2 = pickle.loads(out)

    np.testing.assert_array_equal(x.getWeights(), x2.getWeights())
    np.testing.assert_array_equal(x.getBins(), x2.getBins())    




    
def test_copy():
    std = 100
    mu  = 100
    np.random.seed(31337)
    data = np.random.randn(10_000)*std + mu
    x = mc.Digest(maxBins=64)
    for d in data:
        x.add(d)

    x2 = copy.deepcopy(x)

    np.testing.assert_array_equal(x.getWeights(), x2.getWeights())
    np.testing.assert_array_equal(x.getBins(), x2.getBins())    



def test_quantile_monotonic():
    """Tests that quantile() is non-decreasing in p and stays within the support."""
    std = 100
    mu  = 100
    np.random.seed(31337)
    data = np.random.randn(10_000)*std + mu
    x = mc.Digest(maxBins=32)
    for d in data:
        x.add(d)

    ps = np.linspace(0.001, 0.999, 2000)
    qs = [x.quantile(p) for p in ps]

    for p, q in zip(ps, qs):
        assert x.lower() <= q <= x.upper(), \
            f'quantile({p}) = {q} outside [{x.lower()}, {x.upper()}]'

    for i in range(len(ps) - 1):
        assert qs[i+1] >= qs[i], \
            f'quantile decreased between p={ps[i]} ({qs[i]}) and p={ps[i+1]} ({qs[i+1]})'


def test_quantile_covers_total_weight():
    """Tests that quantile()'s weight gaps add up to the total weight of the digest.

    quantile() walks a ladder of weight gaps between adjacent centroids. If any
    gap is under-weighted the ladder stops short of the total weight, and every p
    past that point falls out of the loop and returns upper() instead of
    interpolating. Rather than re-deriving the ladder here, we assert the
    observable consequence: the loop only runs out at p == 1, so for any p below
    that the result has to be strictly inside the support.
    """
    std = 100
    mu  = 100
    np.random.seed(31337)
    data = np.random.randn(10_000)*std + mu
    x = mc.Digest(maxBins=32)
    for d in data:
        x.add(d)

    for p in [0.5, 0.9, 0.99, 0.999, 0.99999]:
        assert x.quantile(p) < x.upper(), f'quantile({p}) clamped to upper()'


def test_quantile_heavy_last_centroid():
    """Tests interpolation into a final centroid carrying most of the weight.

    The shortfall in the ladder is half of the last centroid's weight, so it is
    nearly invisible while that centroid is a singleton. Here 1000 of 1010 points
    sit on the top centroid, which puts the shortfall at half the range.
    """
    x = mc.Digest(maxBins=16)
    for v in range(1, 11):
        x.add(float(v))
    for _ in range(1000):
        x.add(100.0)

    assert x.getActiveBinCount() == 11
    assert x.getWeights()[-1] == 1000

    for p in [0.6, 0.75, 0.9, 0.99]:
        assert x.quantile(p) < x.upper(), f'quantile({p}) clamped to upper()'


def test_quantile_two_centroids():
    """Tests the degenerate digest whose only segment is both the first and the last.

    Both centroids have to contribute their full weight to the single gap,
    otherwise the ladder covers less than the total weight.
    """
    x = mc.Digest(maxBins=16)
    x.add(1.0)
    x.add(5.0, 9.0)

    assert x.getActiveBinCount() == 2
    assert sum(x.getWeights()) == 10

    assert x.quantile(0.5) > x.lower()
    assert x.quantile(0.99) < x.upper()
