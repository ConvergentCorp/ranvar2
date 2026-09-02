import ranvar2 as mc
import numpy as np
import pickle
import copy
import pytest


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
    """Tests the CDF against the sigma bands of the data it was given.

    Measured against the sample's own band fractions rather than the textbook
    0.6827/0.9545/0.9973: a digest approximates the data it was fed, and at
    10,000 points the sample's own 1-sigma fraction sits about 0.005 away from
    the theoretical one, which is more than the bar below allows the digest.

    The 3-sigma bar is looser at maxBins=32 than at 128 on purpose. Centroids
    are now placed by quantile, so the outermost bucket of a 32 bin digest
    spans about 0.24% of the weight however far out in value the points are.
    Merging by distance instead, as this used to, spent most of its bins on the
    sparse tail of a normal -- which read better here and cost an order of
    magnitude on skewed data, where the tail is where the bins were needed
    least. The resolution is still there for the asking: at 128 bins the same
    band is exact to 1e-4.
    """
    std = 100
    mu  = 100
    np.random.seed(31337)
    data = np.random.randn(10_000)*std + mu

    # What the sample itself says, which is the most a digest of it could know.
    empirical = [float(np.mean(np.abs(data - mu) <= std*k)) for k in (1, 2, 3)]

    for maxBins, bars in [(32, [1e-3, 5e-3, 5e-3]), (128, [1e-3, 1e-3, 1e-3])]:
        x = mc.Digest(maxBins=maxBins)
        x.fit(data)

        for k, want, bar in zip((1, 2, 3), empirical, bars):
            got = x.cdf(mu + std*k) - x.cdf(mu - std*k)

            assert abs(want - got) <= bar, \
                f'maxBins={maxBins}, {k} sigma: {got} vs {want}'


def test_normalApprox_quantile():
    """Tests quantile() against the sample's own percentiles.

    The textbook normal quantiles this used to compare against are not what a
    digest of 10,000 draws can reproduce: at p=0.25 this sample's own 25th
    percentile is 2.9% away from the theoretical one, which on its own exceeds
    the 2.5% the test allows. Judging the digest against the data it was given
    separates its error from the sample's.
    """
    std = 100
    mu  = 100
    np.random.seed(31337)
    data = np.random.randn(10_000)*std + mu

    x = mc.Digest(maxBins=64)
    x.fit(data)

    for p in [0.05, 0.25, 0.5, 0.75, 0.95]:
        want = float(np.percentile(data, p*100))
        got  = x.quantile(p)

        assert abs((want - got) / want) <= 2.5e-2, f'p={p}: {got} vs {want}'


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

    assert x.getActiveBinCount() <= 16
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
    """Tests that a query sees every point added before it.

    Two caches could hide a point: the weight ladder quantile() searches, which
    is rebuilt only when the digest has changed since, and the buffer points
    now wait in until something reads them. A missed invalidation on either
    would answer as though the later points had never arrived.

    Checked against the data rather than against a digest built in one go.
    Reading folds the buffer in, so when a digest is read is now part of what
    its centroids become, and two digests given the same points in the same
    order but read at different times are close rather than identical. What is
    still exact is what these assert: the bounds are kept whole rather than
    merged, and no weight is created or lost.
    """
    np.random.seed(31337)
    data = np.random.randn(400)*10 + 50

    live = mc.Digest(maxBins=16)

    for i, d in enumerate(data):
        live.add(d)

        seen = data[:i+1]

        assert live.lower() == seen.min(), f'lower() stale after {i+1} points'
        assert live.upper() == seen.max(), f'upper() stale after {i+1} points'
        assert sum(live.getWeights()) == pytest.approx(i + 1), \
            f'weight lost or duplicated after {i+1} points'

        if i % 37 != 0:
            continue

        # Querying here populates the ladder, so the next add has to clear it.
        previous = live.quantile(0.5)

        assert live.lower() <= previous <= live.upper()

        for p in [0.05, 0.25, 0.5, 0.75, 0.95]:
            q = live.quantile(p)

            assert live.lower() <= q <= live.upper(), \
                f'quantile({p}) outside the support after {i+1} points'

    # A batch of points well past the current support has to move the median,
    # which a query answering from a stale ladder would not do.
    before = live.quantile(0.5)

    for _ in range(400):
        live.add(1000.0)

    assert live.quantile(0.5) > before


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


def test_maxbins_below_three_is_refused():
    """Tests that a digest too small to merge is refused at construction.

    Once full a digest holds maxBins + 1 centroids, and the merge step keeps
    the first and last for the tails, so it needs an interior pair left to
    merge. Below three bins _findMinimumDifference() scans an empty range and
    returns its -1 sentinel, which _add() would then use as an index -- one
    slot before the arrays, with bounds checking off.
    """
    for maxBins in [-1, 0, 1, 2]:
        with pytest.raises(ValueError, match='at least 3'):
            mc.Digest(maxBins=maxBins)

    # Three is the smallest that can merge, and has to keep working.
    x = mc.Digest(maxBins=3)
    for d in [5.0, 1.0, 9.0, 3.0, 7.0, 2.0]:
        x.add(d)

    assert x.getActiveBinCount() == 3
    assert x.lower() == 1.0
    assert x.upper() == 9.0
    assert sum(x.getWeights()) == 6


def test_maxbins_bound_is_enforced_for_every_subclass():
    """Tests that the parametric subclasses validate maxBins as well.

    They pass it through to RanVar and never build a digest with it, but they
    would hand the same corrupting value on to anything that later did.
    """
    for build in [
        lambda n: mc.Digest(maxBins=n),
        lambda n: mc.Normal(mean=0.0, std=1.0, maxBins=n),
        lambda n: mc.NegBinom(mean=1.0, dispersion=1.0, maxBins=n),
        lambda n: mc.Constant(value=1.0, maxBins=n),
    ]:
        with pytest.raises(ValueError, match='at least 3'):
            build(2)

        assert build(3) is not None


def test_fit_agrees_across_input_types():
    """Tests that the typed bulk path fits a numpy array exactly as iterating it did.

    fit() hands a contiguous numeric array straight to a C loop rather than
    iterating it in Python. That has to be an optimisation only: every input
    shape has to produce the same centroids, to the last bit.
    """
    values = [float(v) for v in range(1000)]

    reference = mc.Digest(maxBins=32)
    reference.fit(values)

    sources = [
        values,
        tuple(values),
        (v for v in values),
        iter(values),
        np.array(values, dtype=np.float64),
        np.array(values, dtype=np.float32),
        np.arange(1000, dtype=np.int64),
        np.array(values, dtype=np.float64)[::-1][::-1],   # a non-contiguous view
    ]

    for source in sources:
        x = mc.Digest(maxBins=32)
        x.fit(source)

        assert np.array_equal(x.getBins(), reference.getBins())
        assert np.array_equal(x.getWeights(), reference.getWeights())


def test_a_digest_allocates_nothing_until_it_is_used():
    """Tests that a digest that has never held a point still behaves like one.

    The centroid buffers are allocated on first use rather than in __init__,
    because Normal, NegBinom and Constant never read them at all and asRanVar()
    wraps every plain number in a Constant. Nothing above that should be able
    to tell.
    """
    x = mc.Digest(maxBins=16)

    assert x.getActiveBinCount() == 0
    assert list(x.getBins()) == []
    assert list(x.getWeights()) == []
    assert x.toDict()['bins'] == []

    for empty in [x.lower, x.upper, x.mean]:
        with pytest.raises(ValueError):
            empty()

    # A round trip before the buffers exist has to leave a working digest.
    for y in [pickle.loads(pickle.dumps(x)), copy.deepcopy(x), mc.Digest.fromDict(x.toDict())]:
        assert y.getActiveBinCount() == 0

        for e in [3.0, 1.0, 2.0]:
            y.add(e)

        assert list(y.getBins()) == [1.0, 2.0, 3.0]

    # And so does using the original afterwards.
    for e in [3.0, 1.0, 2.0]:
        x.add(e)

    assert list(x.getBins()) == [1.0, 2.0, 3.0]
    assert x.quantile(0.5) == 2.0


def test_cdf_interpolates_single_weight_centroids_as_before():
    """Tests the CDF's special cases for centroids carrying a weight of one.

    A centroid of weight one sits on an exact observation, so a segment between
    two of them has no interior to interpolate across and the CDF steps rather
    than ramps. These are the branches a digest holding fewer points than bins
    takes, and they are the ones a total taken from the ladder's cache would be
    wrong for if it were not accumulated the same way.
    """
    steps = mc.Digest(maxBins=8)
    steps.fit([1.0, 2.0, 3.0])

    assert list(steps.getWeights()) == [1.0, 1.0, 1.0]
    assert steps.cdf(1.0) == 0.0
    assert steps.cdf(1.5) == 0.0
    assert steps.cdf(2.5) == 1.0 / 3.0
    assert steps.cdf(3.0) == 1.0

    # One interior centroid heavy enough to interpolate across.
    ramp = mc.Digest(maxBins=8)
    ramp.fit([1.0, 2.0, 2.0, 2.0, 3.0])

    assert list(ramp.getWeights()) == [1.0, 3.0, 1.0]
    assert ramp.cdf(2.5) == 0.65
    assert ramp.dcdf(2.5) == 0.3


def test_cdf_and_dcdf_survive_points_added_after_a_query():
    """Tests that both readers see centroids added since they last ran.

    They take the total weight from the cache the ladder keeps, which is only
    valid while the ladder is, so they have to honour the same stale flag
    quantile() does.
    """
    np.random.seed(31337)
    data = np.random.randn(400)*10 + 50

    live = mc.Digest(maxBins=16)

    for i, e in enumerate(data):
        live.add(e)

        if i < 2:
            continue

        # Querying here populates the cache, so the next add has to clear it.
        live.cdf(50.0)

        # The total the cache holds has to be the weight actually present; a
        # stale one would put the CDF off the [0, 1] it is a probability over,
        # and leave the density inconsistent with it.
        assert 0.0 <= live.cdf(50.0) <= 1.0, f'stale total after {i+1} points'
        assert live.cdf(live.lower()) == 0.0
        assert live.cdf(live.upper()) == 1.0
        assert live.dcdf(50.0) >= 0.0, f'negative density after {i+1} points'
        assert sum(live.getWeights()) == pytest.approx(i + 1)


@pytest.mark.parametrize('maxBins', [3, 4, 8, 16, 32])
@pytest.mark.parametrize('n', [0, 1, 2, 3, 5, 17, 100, 5000])
def test_the_digest_holds_its_invariants(maxBins, n):
    """Tests what a digest promises, whatever it was given.

    Points are buffered and folded in a batch at a time, so these are checked
    across sizes either side of both the bin ceiling and the buffer: at most
    maxBins centroids, in ascending order, carrying exactly the weight that was
    put in, with the extremes kept whole and every quantile inside them.
    """
    np.random.seed(31337)
    data = list(np.random.randn(n)*100 + 100)

    x = mc.Digest(maxBins=maxBins)
    x.fit(data)

    bins    = list(x.getBins())
    weights = list(x.getWeights())

    assert len(bins) <= maxBins
    assert len(bins) == x.getActiveBinCount()
    assert len(weights) == len(bins)
    assert bins == sorted(bins)
    assert sum(weights) == pytest.approx(n)

    if not n:
        return

    # Kept whole rather than merged, so these are exact and not estimates.
    assert x.lower() == min(data)
    assert x.upper() == max(data)

    for p in [0.0, 0.01, 0.25, 0.5, 0.75, 0.99, 1.0]:
        assert x.lower() <= x.quantile(p) <= x.upper()


def test_repeated_values_do_not_waste_centroids():
    """Tests that a digest fed one value many times keeps one centroid for it.

    Centroids are placed by quantile, so copies of a single value spread across
    many buckets. Centroids sitting on the same point span no interval for a
    quantile to interpolate across, so they are folded back together.
    """
    x = mc.Digest(maxBins=16)
    x.fit([4.0]*1000)

    assert list(x.getBins()) == [4.0]
    assert list(x.getWeights()) == [1000.0]
    assert x.quantile(0.5) == 4.0
    assert x.lower() == x.upper() == 4.0


def test_weighted_points_are_merged_by_weight():
    """Tests that add()'s count is carried through the merge."""
    x = mc.Digest(maxBins=8)

    for value, count in [(1.0, 2.0), (5.0, 3.0), (3.0, 1.5), (2.0, 0.5)]:
        x.add(value, count)

    assert sum(x.getWeights()) == pytest.approx(7.0)
    assert x.lower() == 1.0
    assert x.upper() == 5.0

    # The weighted mean is exact through a merge: it is the one thing merging
    # two centroids into their weighted average preserves.
    assert x.mean() == pytest.approx((1*2 + 5*3 + 3*1.5 + 2*0.5) / 7.0)
