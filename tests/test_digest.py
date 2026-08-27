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

