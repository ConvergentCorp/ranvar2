import ranvar2 as mc
import numpy as np
import pytest
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
        ('cdfIntegral()',  lambda: x.cdfIntegral(1.0, 5.0)),
        ('ccdfIntegral()', lambda: x.ccdfIntegral(1.0, 5.0)),
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


# cdfIntegral()/ccdfIntegral(): the areas under the two curves, read off the
# centroids and weights rather than integrated numerically. Each is checked
# against a midpoint rule over the very function it claims to integrate,
# which assumes nothing about the shape of that function. ------------------

def _midpointIntegral(f, a, b, n=200_000):
    """Integrate f over [a, b] by the midpoint rule.

    Deliberately naive: it only evaluates f, so it is a check on the
    analytical integral rather than a restatement of it. Its own error is
    the rule's, of order (b - a)/n at each jump in f, which the tolerances
    below leave room for.
    """
    edges = np.linspace(a, b, n + 1)
    mids  = 0.5*(edges[:-1] + edges[1:])
    h     = (b - a) / n

    return float(sum(f(float(m)) for m in mids) * h)


def _integralWindows(d):
    """A spread of limits over a digest: inside one segment, across several,
    hanging off each end of the support, and the support itself."""
    lo, hi = d.lower(), d.upper()
    span   = hi - lo

    return [
        (lo, hi),
        (lo - span, hi + span),
        (lo + 0.1*span, lo + 0.4*span),
        (lo + 0.37*span, hi - 0.11*span),
        (lo - 2*span, lo + 0.25*span),
        (hi - 0.25*span, hi + 3*span),
    ]


def _fittedDigest():
    """A digest over enough points to have merged centroids of mixed weight."""
    np.random.seed(31337)

    x = mc.Digest(maxBins=32)
    x.fit(np.random.randn(10_000)*100 + 100)

    return x


def _weightedDigest():
    """A digest built by hand so that centroids of weight one sit next to
    heavier ones, which is what drives the interpolation through all four of
    its cases."""
    x = mc.Digest(maxBins=32)

    for point, count in [(0.0, 1.0), (1.0, 5.0), (2.5, 1.0),
                         (4.0, 3.0), (7.0, 1.0), (9.0, 12.0)]:
        x.add(point, count)

    return x


def _twoCentroidDigest():
    """The smallest digest with an interior to integrate over."""
    x = mc.Digest(maxBins=32)
    x.add(2.0, 3.0)
    x.add(8.0, 4.0)

    return x


def _unitDigest():
    """Every centroid of weight one, which interpolates to a staircase."""
    x = mc.Digest(maxBins=32)
    x.fit([1, 2, 3, 4, 5])

    return x


def test_cdfIntegral_matches_a_numerical_integral():
    """Tests cdfIntegral() against a midpoint rule over cdf() itself, on
    digests covering all four interpolation cases.
    """
    for build in (_fittedDigest, _weightedDigest, _twoCentroidDigest, _unitDigest):
        x = build()

        for a, b in _integralWindows(x):
            exact     = x.cdfIntegral(a, b)
            numerical = _midpointIntegral(x.cdf, a, b)

            assert abs(exact - numerical) <= 1e-5*(b - a) + 1e-9, \
                f'{build.__name__} cdf over [{a}, {b}]: {exact} vs {numerical}'


def test_ccdfIntegral_matches_a_numerical_integral():
    """Tests ccdfIntegral() the same way, against a midpoint rule over
    ccdf().
    """
    for build in (_fittedDigest, _weightedDigest, _twoCentroidDigest, _unitDigest):
        x = build()

        for a, b in _integralWindows(x):
            exact     = x.ccdfIntegral(a, b)
            numerical = _midpointIntegral(x.ccdf, a, b)

            assert abs(exact - numerical) <= 1e-5*(b - a) + 1e-9, \
                f'{build.__name__} ccdf over [{a}, {b}]: {exact} vs {numerical}'


def test_integrals_of_the_two_curves_add_up_to_the_width():
    """Tests that the two areas partition the rectangle over [a, b], since
    the curves sum to 1 everywhere.
    """
    x = _fittedDigest()

    for a, b in _integralWindows(x):
        total = x.cdfIntegral(a, b) + x.ccdfIntegral(a, b)

        assert abs(total - (b - a)) <= 1e-9*(b - a)


def test_integrals_are_additive_over_adjacent_windows():
    """Tests that splitting a window anywhere, including exactly on a
    centroid, splits the area with it.
    """
    x = _weightedDigest()

    lo, hi = x.lower(), x.upper()

    for m in (lo, 1.0, 2.5, 3.3, 7.0, hi, hi + 1.0):
        a, b = lo - 2.0, hi + 2.0

        assert x.cdfIntegral(a, b) == pytest.approx(
            x.cdfIntegral(a, m) + x.cdfIntegral(m, b), abs=1e-12)
        assert x.ccdfIntegral(a, b) == pytest.approx(
            x.ccdfIntegral(a, m) + x.ccdfIntegral(m, b), abs=1e-12)


def test_integrals_of_an_empty_window_and_reversed_limits():
    """Tests the two conventions every integral follows: no width, no area,
    and swapping the limits negates it.
    """
    x = _weightedDigest()

    assert x.cdfIntegral(3.0, 3.0)  == 0.0
    assert x.ccdfIntegral(3.0, 3.0) == 0.0

    assert x.cdfIntegral(6.0, 2.0)  == -x.cdfIntegral(2.0, 6.0)
    assert x.ccdfIntegral(6.0, 2.0) == -x.ccdfIntegral(2.0, 6.0)


def test_integrals_outside_the_support_are_flat():
    """Tests the two tails: the CDF contributes nothing below the support
    and the full width above it, and the CCDF the other way around.
    """
    x = _weightedDigest()

    lo, hi = x.lower(), x.upper()

    assert x.cdfIntegral(lo - 7.0, lo)  == 0.0
    assert x.ccdfIntegral(hi, hi + 7.0) == 0.0

    assert x.cdfIntegral(hi, hi + 7.0) == pytest.approx(7.0)
    assert x.ccdfIntegral(lo - 7.0, lo) == pytest.approx(7.0)


def test_integrals_accept_the_unbounded_side():
    """Tests that the limit the curve is zero towards may be infinite, and
    that the other one reports an infinite area rather than a finite wrong
    answer.
    """
    x = _fittedDigest()

    lo, hi = x.lower(), x.upper()
    span   = hi - lo

    # Far enough out that nothing is left to accumulate, so the infinite
    # limit has to agree with it.
    assert x.cdfIntegral(float('-inf'), hi) == pytest.approx(x.cdfIntegral(lo - 1e3*span, hi))
    assert x.ccdfIntegral(lo, float('inf')) == pytest.approx(x.ccdfIntegral(lo, hi + 1e3*span))

    assert x.cdfIntegral(lo, float('inf'))  == float('inf')
    assert x.ccdfIntegral(float('-inf'), hi) == float('inf')


def _interpolatedMean(x):
    """The mean of the curve cdf() describes, taken as a first moment.

    Rebuilds the cumulative weight ladder from the centroids and weights the
    way _rebuildLadder() defines it, then sums mass times position over the
    segments. The curve is continuous, so every segment spreads its weight
    evenly and sits at its own midpoint, and there is no mass on any single
    point to account for separately.

    That is the other route to the mean: the integrals under test take the
    area under the survival function instead, so the two share the ladder
    and nothing else.
    """
    c = np.asarray(x.getBins(), dtype=float)
    m = np.asarray(x.getWeights(), dtype=float)

    n = len(c)
    W = m.sum()

    if n == 1:
        return float(c[0])

    # An outer centroid gives its whole weight to its only segment, an
    # interior one half to each side.
    gaps = []
    for i in range(n - 1):
        gap  = m[i] if i == 0 else m[i]/2
        gap += m[i+1] if i == n - 2 else m[i+1]/2

        gaps.append(gap)

    return float(sum(gaps[i]/W * (c[i] + c[i+1])/2 for i in range(n - 1)))


@pytest.mark.parametrize('build', [
    _fittedDigest, _weightedDigest, _twoCentroidDigest, _unitDigest,
])
def test_integrals_of_the_ccdf_give_the_mean_of_the_interpolated_curve(build):
    """Tests the identity E[X] = lower + the integral of the CCDF over the
    whole support, against the same curve's first moment.

    This is what pins down the ends: the identity only holds if the area
    accounts for every jump, including the one at the bottom centroid and
    the one at the top where the curve is clamped, so a tail dropped or
    counted twice shows up here as a shifted mean rather than as a small
    numerical residual.

    The digest's own mean() is the mean of the data added to it, which the
    interpolated curve need not reproduce exactly, so the reference is the
    curve rather than mean().
    """
    x = build()

    lo = x.lower()

    assert lo + x.ccdfIntegral(lo, float('inf')) == pytest.approx(
        _interpolatedMean(x), rel=1e-12)


def test_integrals_reject_nan_limits():
    """Tests that a NaN limit is refused rather than quietly returning NaN."""
    x = _weightedDigest()

    nan = float('nan')

    for call in (lambda: x.cdfIntegral(nan, 1.0),
                 lambda: x.cdfIntegral(1.0, nan),
                 lambda: x.ccdfIntegral(nan, 1.0),
                 lambda: x.ccdfIntegral(1.0, nan)):
        try:
            call()
        except ValueError:
            pass
        else:
            assert False, 'a NaN limit should raise ValueError'


def test_integrals_of_a_single_centroid_digest():
    """Tests the degenerate digest, which has no segment to interpolate
    over: its CDF is a single step at the one centroid, so the area is
    whatever of the window lies on the right side of it.
    """
    x = mc.Digest(maxBins=32)
    x.add(4.0, 3.0)

    assert x.cdfIntegral(0.0, 10.0)  == pytest.approx(6.0)
    assert x.ccdfIntegral(0.0, 10.0) == pytest.approx(4.0)

    assert x.cdfIntegral(0.0, 10.0)  == pytest.approx(_midpointIntegral(x.cdf, 0.0, 10.0),  abs=1e-4)
    assert x.ccdfIntegral(0.0, 10.0) == pytest.approx(_midpointIntegral(x.ccdf, 0.0, 10.0), abs=1e-4)

    assert x.cdfIntegral(float('-inf'), 4.0) == 0.0
    assert x.ccdfIntegral(4.0, float('inf')) == 0.0


# The two extreme centroids, where a weight of one means the interpolation
# takes its Case II/III branch on the very first or very last segment. Those
# are the segments that meet the clamped ends of the CDF, so they are the
# ones where an integral can lose or double count a tail. ------------------

ENDPOINT_WEIGHTS = [
    ('both ends singleton', [(0., 1.), (1., 5.), (2.5, 1.), (4., 3.), (7., 1.)]),
    ('only the first',      [(0., 1.), (1., 5.), (2.5, 1.), (4., 3.), (7., 9.)]),
    ('only the last',       [(0., 4.), (1., 5.), (2.5, 1.), (4., 3.), (7., 1.)]),
    ('neither',             [(0., 4.), (1., 5.), (2.5, 1.), (4., 3.), (7., 9.)]),
    ('every centroid',      [(0., 1.), (1., 1.), (2.5, 1.), (4., 1.), (7., 1.)]),
]


def _endpointDigest(spec):
    x = mc.Digest(maxBins=32)

    for point, count in spec:
        x.add(point, count)

    return x


@pytest.mark.parametrize('name, spec', ENDPOINT_WEIGHTS)
def test_integrals_with_singleton_endpoints(name, spec):
    """Tests both integrals against a numerical one on digests whose first
    and last centroids carry a weight of one, in every combination.

    A singleton at an end drives the interpolation of the first or last
    segment through a different case than its neighbours, so these are the
    digests where the segment walk and the flat tail outside the support
    have to agree on where the support ends.
    """
    x = _endpointDigest(spec)

    lo, hi = x.lower(), x.upper()

    windows = [
        (lo, hi),
        (lo - 3.0, hi + 3.0),
        (lo, lo + 0.5),
        (hi - 0.5, hi),
        (lo - 3.0, lo + 0.25),
        (hi - 0.25, hi + 3.0),
    ]

    for a, b in windows:
        assert x.cdfIntegral(a, b) == pytest.approx(
            _midpointIntegral(x.cdf, a, b), abs=1e-4), f'{name} cdf over [{a}, {b}]'
        assert x.ccdfIntegral(a, b) == pytest.approx(
            _midpointIntegral(x.ccdf, a, b), abs=1e-4), f'{name} ccdf over [{a}, {b}]'


@pytest.mark.parametrize('name, spec', ENDPOINT_WEIGHTS)
def test_integrals_continue_flat_past_a_singleton_endpoint(name, spec):
    """Tests that the area does not step at either end of the support.

    Past the top the CDF is 1, so widening the window there adds exactly the
    width it adds and nothing else; below the bottom the CCDF is 1 and the
    same holds. Whatever mass the interpolation leaves for the jump at an
    end, the integral has to join the flat tail to the segment walk without
    a seam, and a singleton endpoint jumps by a different amount than a
    heavy one.
    """
    x = _endpointDigest(spec)

    lo, hi = x.lower(), x.upper()

    for d in (1e-6, 0.25, 4.0):
        assert x.cdfIntegral(lo, hi + d) == pytest.approx(
            x.cdfIntegral(lo, hi) + d, abs=1e-12)
        assert x.ccdfIntegral(lo - d, hi) == pytest.approx(
            x.ccdfIntegral(lo, hi) + d, abs=1e-12)

        # And from the other side: a window wholly outside the support is
        # all of one curve and none of the other.
        assert x.cdfIntegral(hi, hi + d)  == pytest.approx(d, abs=1e-12)
        assert x.ccdfIntegral(lo - d, lo) == pytest.approx(d, abs=1e-12)
        assert x.cdfIntegral(lo - d, lo)  == 0.0
        assert x.ccdfIntegral(hi, hi + d) == 0.0


@pytest.mark.parametrize('name, spec', ENDPOINT_WEIGHTS)
def test_integrals_split_exactly_on_the_endpoints(name, spec):
    """Tests that splitting a window at either end of the support splits the
    area with it, which is where a tail counted twice or not at all would
    show up.
    """
    x = _endpointDigest(spec)

    lo, hi = x.lower(), x.upper()
    a, b   = lo - 2.0, hi + 2.0

    for m in (lo, hi):
        assert x.cdfIntegral(a, b) == pytest.approx(
            x.cdfIntegral(a, m) + x.cdfIntegral(m, b), abs=1e-12)
        assert x.ccdfIntegral(a, b) == pytest.approx(
            x.ccdfIntegral(a, m) + x.ccdfIntegral(m, b), abs=1e-12)

    assert x.cdfIntegral(a, b) + x.ccdfIntegral(a, b) == pytest.approx(b - a)


@pytest.mark.parametrize('name, spec', ENDPOINT_WEIGHTS)
def test_no_mass_sits_on_an_endpoint(name, spec):
    """Tests that the curve meets the ends of its support continuously,
    whatever the extreme centroids weigh.

    A jump at an end would be mass sitting on a single point, which is the
    one thing the area under the curve cannot see and quantile() cannot
    reproduce. So the CDF has to reach 0 at the bottom centroid and 1 at the
    top one, not merely be clamped there.
    """
    x = _endpointDigest(spec)

    lo, hi = x.lower(), x.upper()

    for d in (1e-6, 1e-9):
        assert x.cdf(lo + d) == pytest.approx(0.0, abs=1e-4)
        assert x.cdf(hi - d) == pytest.approx(1.0, abs=1e-4)


@pytest.mark.parametrize('name, spec', ENDPOINT_WEIGHTS)
def test_no_mass_sits_on_an_interior_centroid(name, spec):
    """Tests the same at every centroid in between, where a weight of one
    used to put the two neighbouring segments at odds with each other.
    """
    x = _endpointDigest(spec)

    bins = x.getBins()

    for c in bins[1:-1]:
        d = 1e-7

        assert x.cdf(float(c) + d) == pytest.approx(x.cdf(float(c) - d), abs=1e-6)


@pytest.mark.parametrize('name, spec', ENDPOINT_WEIGHTS)
def test_singleton_endpoints_carry_their_mass_into_the_area(name, spec):
    """Tests the mean identity on each endpoint configuration.

    The area under the CCDF over the whole support is E[X] - lower, and the
    mass at the two ends is exactly what a singleton there changes: a first
    or last centroid of weight one shifts the curve's jumps, and any of that
    mass the area failed to carry would move the mean it implies. Matching
    the curve's own first moment to the last bits rules that out.
    """
    x = _endpointDigest(spec)

    lo = x.lower()

    assert lo + x.ccdfIntegral(lo, float('inf')) == pytest.approx(
        _interpolatedMean(x), rel=1e-12)


def test_a_single_centroid_carries_its_mass_into_the_area():
    """Tests the same identity on the digest with no interior at all, where
    the whole distribution is one atom and the area is the flat tail alone.
    """
    x = mc.Digest(maxBins=32)
    x.add(4.0, 3.0)

    assert x.lower() + x.ccdfIntegral(x.lower(), float('inf')) == pytest.approx(
        _interpolatedMean(x), rel=1e-12)


# cdf() and quantile() read one ladder, so they describe one distribution.
# They used to disagree: quantile() gave an outer centroid its whole weight
# across its only segment, while cdf() gave it half and left the rest as a
# jump, which put the two as far as 0.4 apart in probability on a small
# digest. ------------------------------------------------------------------

ALL_DIGESTS = [
    ('fitted',         _fittedDigest),
    ('weighted',       _weightedDigest),
    ('two centroids',  _twoCentroidDigest),
    ('unit weights',   _unitDigest),
] + [(name, lambda spec=spec: _endpointDigest(spec)) for name, spec in ENDPOINT_WEIGHTS]


@pytest.mark.parametrize('name, build', ALL_DIGESTS)
def test_cdf_inverts_quantile(name, build):
    """Tests that cdf() undoes quantile() exactly, across the whole
    probability range and on every shape of digest.

    This is what it means for the two to describe one distribution. It is an
    equality rather than an approximation: both read the same rungs, so the
    weight quantile() spreads across a segment is the weight cdf() finds
    there.
    """
    x = build()

    for p in np.linspace(0.001, 0.999, 999):
        assert x.cdf(x.quantile(float(p))) == pytest.approx(float(p), abs=1e-12), \
            f'{name} at p={p}'


@pytest.mark.parametrize('name, build', ALL_DIGESTS)
def test_quantile_inverts_cdf(name, build):
    """Tests the round trip the other way, from a point in the support back
    to itself, which pins the two together at the ends of each segment as
    well as inside it.
    """
    x = build()

    lo, hi = x.lower(), x.upper()

    for f in np.linspace(0.0, 1.0, 101):
        k = lo + f*(hi - lo)

        assert x.quantile(x.cdf(k)) == pytest.approx(k, abs=1e-9*(hi - lo) + 1e-12), \
            f'{name} at k={k}'


@pytest.mark.parametrize('name, build', ALL_DIGESTS)
def test_cdf_agrees_with_the_ladder_at_every_centroid(name, build):
    """Tests cdf() against the cumulative weight at each centroid, worked
    out from the weights alone.

    An outer centroid gives its whole weight to its only segment and an
    interior one half to each side, which is the rule quantile()'s ladder
    follows and the one cdf() used to break at the two ends.
    """
    x = build()

    c = x.getBins()
    m = x.getWeights()

    n = len(c)
    W = float(sum(m))

    if n == 1:
        return

    cum = 0.0

    for i in range(n - 1):
        assert x.cdf(float(c[i])) == pytest.approx(cum/W, abs=1e-12), \
            f'{name} at centroid {i}'

        gap  = m[i] if i == 0 else m[i]/2
        gap += m[i+1] if i == n - 2 else m[i+1]/2

        cum += gap

    assert cum == pytest.approx(W)
    assert x.cdf(float(c[n-1])) == 1.0


@pytest.mark.parametrize('name, build', ALL_DIGESTS)
def test_density_integrates_to_the_cdf(name, build):
    """Tests that dcdf() is the gradient of the very curve cdf() reads, by
    integrating the density back into the CDF numerically.
    """
    x = build()

    lo, hi = x.lower(), x.upper()

    for f in (0.25, 0.5, 0.9):
        k = lo + f*(hi - lo)

        assert _midpointIntegral(x.dcdf, lo, k, 100_000) == pytest.approx(
            x.cdf(k), abs=1e-4), f'{name} at k={k}'
