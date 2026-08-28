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


def test_sample_is_not_quantised():
    """Tests that sample() draws continuous values rather than rounded integers."""
    std = 100
    mu  = 100
    np.random.seed(31337)
    data = np.random.randn(10_000)*std + mu
    x = mc.Digest(maxBins=64)
    for d in data:
        x.add(d)

    mc.seed(1234)
    draws = [x.sample() for _ in range(200)]

    assert any(not float(v).is_integer() for v in draws), \
        'every draw landed on an integer, sample() is still rounding'

    for v in draws:
        assert x.lower() <= v <= x.upper(), f'draw {v} outside the support'


def test_seed_is_reproducible():
    """Tests that seed() pins the sequence of draws sample() produces."""
    std = 100
    mu  = 100
    np.random.seed(31337)
    data = np.random.randn(10_000)*std + mu
    x = mc.Digest(maxBins=64)
    for d in data:
        x.add(d)

    mc.seed(42)
    first = [x.sample() for _ in range(20)]

    mc.seed(42)
    again = [x.sample() for _ in range(20)]

    mc.seed(43)
    other = [x.sample() for _ in range(20)]

    assert first == again, 'the same seed produced a different sequence'
    assert first != other, 'a different seed produced the same sequence'


def test_mean():
    """Tests that mean() is the exact mean of the data added to the digest.

    Merging replaces two centroids with their weighted average, which leaves the
    total weighted sum untouched, so no accuracy is lost no matter how much the
    digest compresses.
    """
    std = 100
    mu  = 100
    np.random.seed(31337)
    data = np.random.randn(10_000)*std + mu
    x = mc.Digest(maxBins=32)
    for d in data:
        x.add(d)

    assert x.getActiveBinCount() == 32, 'expected the digest to have compressed'
    np.testing.assert_allclose(x.mean(), data.mean(), rtol=1e-12)


def test_empty_digest_raises():
    """Tests that querying an empty digest raises rather than reading out of bounds.

    The centroid arrays are indexed with bounds checking off, and upper() reads
    nActive-1, which is -1 before anything has been added. Every accessor that
    reaches one of those reads has to refuse the empty digest instead of
    returning whatever happens to sit in front of the buffer.
    """
    x = mc.Digest(maxBins=32)

    accessors = [
        ('lower()',        lambda: x.lower()),
        ('upper()',        lambda: x.upper()),
        ('mean()',         lambda: x.mean()),
        ('cdf()',          lambda: x.cdf(5.0)),
        ('ccdf()',         lambda: x.ccdf(5.0)),
        ('dcdf()',         lambda: x.dcdf(5.0)),
        ('dccdf()',        lambda: x.dccdf(5.0)),
        ('quantile(0.0)',  lambda: x.quantile(0.0)),
        ('quantile(0.5)',  lambda: x.quantile(0.5)),
        ('quantile(1.0)',  lambda: x.quantile(1.0)),
        ('sample()',       lambda: x.sample()),
    ]

    for name, call in accessors:
        try:
            call()
        except ValueError:
            pass
        else:
            assert False, f'{name} on an empty digest should raise ValueError'

    # And the guards must not fire once there is something to report on.
    x.fit([1.0, 2.0, 3.0, 4.0, 5.0])

    for name, call in accessors:
        call()


def test_quantile_matches_a_linear_search():
    """Tests the binary search against a naive scan of the same ladder.

    Boundaries are where a binary search goes wrong, so this checks a dense
    sweep of probabilities against an independent, obviously correct walk of the
    weight ladder rather than against the digest's own arithmetic.
    """
    np.random.seed(31337)
    x = mc.Digest(maxBins=64)
    for d in np.random.randn(10_000)*100 + 100:
        x.add(d)

    m = x.getWeights()
    c = x.getBins()
    n = len(m)
    W = m.sum()

    def naive(p):
        wi = 0.0
        w_ = p*W

        for i in range(n-1):
            gap = (m[i] if i == 0 else m[i]/2) + (m[i+1] if i == n-2 else m[i+1]/2)

            if wi <= w_ < wi + gap:
                return (w_ - wi)/gap * (c[i+1] - c[i]) + c[i]

            wi = wi + gap

        return x.upper()

    for p in np.linspace(1e-9, 1 - 1e-9, 3000):
        got = x.quantile(p)
        want = naive(p)
        assert abs(got - want) <= 1e-9 * max(1.0, abs(want)), \
            f'p={p}: binary search gave {got}, linear walk gave {want}'


def test_quantile_reflects_points_added_after_a_query():
    """Tests that the cached weight ladder is rebuilt once more points arrive.

    quantile() caches the ladder and rebuilds it only when a point has been
    added since, so a missed invalidation would answer as though the later
    points had never been added.
    """
    np.random.seed(31337)
    data = np.random.randn(400)*10 + 50

    live = mc.Digest(maxBins=16)

    for i, d in enumerate(data):
        live.add(d)

        if i % 37 != 0:
            continue

        # Querying here populates the cache, so the next add has to clear it.
        live.quantile(0.5)

        fresh = mc.Digest(maxBins=16)
        for e in data[:i+1]:
            fresh.add(e)

        for p in [0.05, 0.25, 0.5, 0.75, 0.95]:
            assert live.quantile(p) == fresh.quantile(p), \
                f'stale ladder after {i+1} points, p={p}'


def test_ladder_is_rebuilt_after_pickle_and_copy():
    """Tests that a restored digest rebuilds the ladder it never received.

    The cache is not serialised, so a copy that did not mark it stale would
    search a ladder full of zeros.
    """
    np.random.seed(31337)
    x = mc.Digest(maxBins=32)
    for d in np.random.randn(2_000)*100 + 100:
        x.add(d)

    x.quantile(0.5)

    for y in [pickle.loads(pickle.dumps(x)), copy.deepcopy(x)]:
        for p in [0.01, 0.1, 0.5, 0.9, 0.99]:
            assert y.quantile(p) == x.quantile(p), f'restored digest differs at p={p}'
