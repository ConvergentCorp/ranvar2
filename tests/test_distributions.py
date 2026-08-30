import cython
import math
import pickle

import numpy as np
import pytest

import ranvar2 as mc
from ranvar2 import RanVar, Normal, NegBinom


# Models are written once and shared, since each distinct one costs a compile.

def identity(x: mc.RanVar) -> cython.double:
    """A model that just draws from whatever digest it is given."""
    return ~x


compiledIdentity = mc.cfunc(identity, samples=200_000)


def empiricalMoments(digest):
    """Compute the mean and variance a compiled model's output digest holds."""
    bins    = digest.getBins()
    weights = digest.getWeights()
    mean    = np.average(bins, weights=weights)
    var     = np.average((bins - mean) ** 2, weights=weights)
    return mean, var


# Normal. ------------------------------------------------------------------

def test_normal_is_a_ranvar():
    """Tests that Normal can be used anywhere a RanVar is expected."""
    assert isinstance(Normal(mean=0.0, std=1.0), RanVar)


def test_normal_rejects_nonpositive_std():
    """Tests that a Normal cannot be built with a non-positive spread."""
    with pytest.raises(ValueError):
        Normal(std=0.0)

    with pytest.raises(ValueError):
        Normal(std=-1.0)


def test_normal_accessors():
    """Tests that mean()/std() return the constructor's parameters."""
    n = Normal(mean=12.5, std=3.0)

    assert n.mean() == 12.5
    assert n.std() == 3.0


def test_normal_sample_matches_its_parameters():
    """Tests that many draws from Normal recover its mean and std."""
    mc.seed(1)

    n  = Normal(mean=50.0, std=8.0)
    xs = np.array([n.sample() for _ in range(200_000)])

    assert xs.mean() == pytest.approx(50.0, abs=0.15)
    assert xs.std()  == pytest.approx(8.0, abs=0.15)


def test_normal_pickles_to_the_same_class():
    """Tests that a Normal round trips through pickle as a Normal, with its
    parameters intact, rather than unpickling to the plain RanVar base.
    """
    n  = Normal(mean=7.0, std=2.0)
    n2 = pickle.loads(pickle.dumps(n))

    assert type(n2) is Normal
    assert n2.mean() == 7.0
    assert n2.std() == 2.0

    n3 = pickle.loads(pickle.dumps(n2))
    assert type(n3) is Normal


def test_normal_compiles_through_a_ranvar_annotated_model():
    """Tests that a compiled model draws from Normal's own sample(), not the
    empty t-digest RanVar.__init__ would otherwise leave it holding.

    The model annotates its parameter as plain RanVar, since that is the type
    ranvar2.pxd declares sample() cpdef on; Normal being passed in still gets
    its own overridden sample() called through the vtable.
    """
    mc.seed(2)

    n   = Normal(mean=30.0, std=5.0)
    out = compiledIdentity(n)

    mean, var = empiricalMoments(out)

    assert mean == pytest.approx(30.0, abs=0.15)
    assert var  == pytest.approx(25.0, abs=1.0)


def test_normal_lower_upper_are_unbounded():
    """Tests that lower()/upper() give the analytical support bounds.

    Unlike RanVar.lower()/upper(), these do not require any data to have
    been added first.
    """
    n = Normal(mean=0.0, std=1.0)

    assert n.lower() == float('-inf')
    assert n.upper() == float('inf')


def test_normal_cdf_matches_the_error_function():
    """Tests cdf()/ccdf()/dcdf()/dccdf() against a reference built from
    math.erf(), independently of ranvar2's own erf() cimport.
    """
    n = Normal(mean=10.0, std=2.0)

    def phi(x):
        return 0.5 * (1 + math.erf((x - 10.0) / (2.0 * math.sqrt(2))))

    def density(x):
        return math.exp(-0.5 * ((x - 10.0) / 2.0) ** 2) / (2.0 * math.sqrt(2 * math.pi))

    for x in (4.0, 8.0, 10.0, 13.5, 17.0):
        assert n.cdf(x)   == pytest.approx(phi(x))
        assert n.ccdf(x)  == pytest.approx(1 - phi(x))
        assert n.dcdf(x)  == pytest.approx(density(x))
        assert n.dccdf(x) == pytest.approx(-density(x))

    assert n.cdf(10.0) == pytest.approx(0.5)


def test_normal_quantile_is_not_implemented():
    """Tests that quantile() refuses rather than silently reading the
    (empty, or unrelated to mean/std) fitted digest RanVar.quantile() would.
    """
    with pytest.raises(NotImplementedError):
        Normal(mean=0.0, std=1.0).quantile(0.5)


def test_normal_add_is_not_implemented():
    """Tests that add() refuses rather than silently building a t-digest
    sample() would then ignore.
    """
    with pytest.raises(NotImplementedError):
        Normal(mean=0.0, std=1.0).add(1.0)


def test_normal_fit_reestimates_mean_and_std():
    """Tests that fit() replaces mean/std with the sample's, by the method
    of moments (sample std, Bessel's correction), rather than adding points
    to a digest sample() would then ignore.
    """
    data = [10.0, 12.0, 8.0, 11.0, 9.0, 13.0, 7.0, 10.0]

    n = Normal(mean=0.0, std=1.0)
    n.fit(data)

    assert n.mean() == pytest.approx(np.mean(data))
    assert n.std()  == pytest.approx(np.std(data, ddof=1))

    # sample() draws from the newly fit parameters, not the constructor's.
    mc.seed(6)
    xs = np.array([n.sample() for _ in range(200_000)])
    assert xs.mean() == pytest.approx(n.mean(), abs=0.1)
    assert xs.std()  == pytest.approx(n.std(), abs=0.1)


def test_normal_fit_needs_at_least_two_points():
    """Tests that fit() refuses data a standard deviation can't be read
    from, rather than silently producing a std of 0 or nan.
    """
    with pytest.raises(ValueError):
        Normal(mean=0.0, std=1.0).fit([5.0])

    with pytest.raises(ValueError):
        Normal(mean=0.0, std=1.0).fit([])


# NegBinom. ------------------------------------------------------------------

def test_negbinom_is_a_ranvar():
    """Tests that NegBinom can be used anywhere a RanVar is expected."""
    assert isinstance(NegBinom(mean=1.0, dispersion=1.0), RanVar)


def test_negbinom_rejects_negative_mean():
    """Tests that a NegBinom cannot be built with a negative mean."""
    with pytest.raises(ValueError):
        NegBinom(mean=-1.0)


def test_negbinom_rejects_nonpositive_dispersion():
    """Tests that a NegBinom cannot be built with non-positive dispersion."""
    with pytest.raises(ValueError):
        NegBinom(dispersion=0.0)

    with pytest.raises(ValueError):
        NegBinom(dispersion=-2.0)


def test_negbinom_accessors():
    """Tests that mean()/dispersion() return the constructor's parameters."""
    nb = NegBinom(mean=4.0, dispersion=0.25)

    assert nb.mean() == 4.0
    assert nb.dispersion() == 0.25


def test_negbinom_pickles_to_the_same_class():
    """Tests that a NegBinom round trips through pickle as a NegBinom, with
    its parameters intact, rather than unpickling to the plain RanVar base.
    """
    nb  = NegBinom(mean=9.0, dispersion=2.0)
    nb2 = pickle.loads(pickle.dumps(nb))

    assert type(nb2) is NegBinom
    assert nb2.mean() == 9.0
    assert nb2.dispersion() == 2.0

    nb3 = pickle.loads(pickle.dumps(nb2))
    assert type(nb3) is NegBinom


def test_negbinom_zero_mean_is_degenerate():
    """Tests that a zero mean NegBinom always draws zero.

    Exercises the boundary where the Gamma half of the mixture has a zero
    scale, which the Poisson step then has to turn into a zero rate rather
    than dividing by it.
    """
    mc.seed(3)

    nb = NegBinom(mean=0.0, dispersion=1.0)

    assert all(nb.sample() == 0.0 for _ in range(1_000))


@pytest.mark.parametrize('mean, dispersion', [
    (3.0, 0.5),   # small mean: exercises Knuth's Poisson method
    (60.0, 0.3),  # large mean: exercises the transformed rejection method
])
def test_negbinom_sample_matches_its_moments(mean, dispersion):
    """Tests that many draws from NegBinom recover the NB2 mean/variance.

    variance = mean + dispersion * mean**2 is the parameterization NegBinom
    documents; both branches of the internal Poisson sampler are covered by
    picking a small and a large mean.
    """
    mc.seed(4)

    nb       = NegBinom(mean=mean, dispersion=dispersion)
    xs       = np.array([nb.sample() for _ in range(200_000)])
    expected = mean + dispersion * mean ** 2

    assert xs.mean() == pytest.approx(mean, rel=0.05, abs=0.1)
    assert xs.var()  == pytest.approx(expected, rel=0.08)
    assert (xs >= 0).all()
    assert np.allclose(xs, np.round(xs))


def test_negbinom_compiles_through_a_ranvar_annotated_model():
    """Tests that a compiled model draws from NegBinom's own sample()."""
    mc.seed(5)

    nb  = NegBinom(mean=20.0, dispersion=0.4)
    out = compiledIdentity(nb)

    mean, var = empiricalMoments(out)
    expected  = 20.0 + 0.4 * 20.0 ** 2

    assert mean == pytest.approx(20.0, rel=0.05, abs=0.1)
    assert var  == pytest.approx(expected, rel=0.1)


def test_negbinom_lower_upper_are_analytical():
    """Tests that lower()/upper() give the analytical support bounds.

    Unlike RanVar.lower()/upper(), these do not require any data to have
    been added first.
    """
    nb = NegBinom(mean=1.0, dispersion=1.0)

    assert nb.lower() == 0.0
    assert nb.upper() == float('inf')


@pytest.mark.parametrize('method, arg', [
    ('cdf', 3.0),
    ('ccdf', 3.0),
    ('dcdf', 3.0),
    ('dccdf', 3.0),
    ('quantile', 0.5),
])
def test_negbinom_has_no_closed_form_for(method, arg):
    """Tests that the methods with no NB2 closed form refuse rather than
    silently reading the (empty, or unrelated to mean/dispersion) fitted
    digest the inherited RanVar implementation would.
    """
    nb = NegBinom(mean=5.0, dispersion=1.0)

    with pytest.raises(NotImplementedError):
        getattr(nb, method)(arg)


def test_negbinom_add_is_not_implemented():
    """Tests that add() refuses rather than silently building a t-digest
    sample() would then ignore.
    """
    with pytest.raises(NotImplementedError):
        NegBinom(mean=1.0, dispersion=1.0).add(1.0)


def test_negbinom_fit_reestimates_mean_and_dispersion():
    """Tests that fit() replaces mean/dispersion with the sample's, by
    inverting NegBinom's own variance = mean + dispersion * mean**2 with the
    sample mean and sample variance (Bessel's correction), rather than
    adding points to a digest sample() would then ignore.
    """
    counts = [2, 5, 0, 8, 1, 9, 3, 12, 0, 4]

    nb = NegBinom(mean=1.0, dispersion=1.0)
    nb.fit(counts)

    sampleMean = np.mean(counts)
    sampleVar  = np.var(counts, ddof=1)
    expectedDispersion = (sampleVar - sampleMean) / sampleMean ** 2

    assert nb.mean() == pytest.approx(sampleMean)
    assert nb.dispersion() == pytest.approx(expectedDispersion)

    # sample() draws from the newly fit parameters, not the constructor's.
    mc.seed(7)
    xs       = np.array([nb.sample() for _ in range(200_000)])
    expected = sampleMean + expectedDispersion * sampleMean ** 2

    assert xs.mean() == pytest.approx(sampleMean, rel=0.05, abs=0.1)
    assert xs.var()  == pytest.approx(expected, rel=0.1)


@pytest.mark.parametrize('data', [
    [5.0],           # fewer than 2 points
    [],               # no points
    [0.0, 0.0, 0.0],  # zero sample mean
    [5.0, 5.0, 5.0],  # underdispersed: sample variance == 0 < mean
])
def test_negbinom_fit_rejects_data_with_no_valid_dispersion(data):
    """Tests that fit() refuses data that cannot yield a positive
    dispersion, rather than silently producing a negative or undefined one.
    """
    with pytest.raises(ValueError):
        NegBinom(mean=1.0, dispersion=1.0).fit(data)


# Every public RanVar method is present, whether inherited unchanged (the
# ones with no data dependency), overridden analytically, or overridden to
# refuse. Digest-only accessors (getActiveBinCount/getBins/getWeights) are
# not part of this contract, since add()/fit() no longer build a digest for
# either subclass. --------------------------------------------------------

PUBLIC_METHODS = [
    'add', 'lower', 'upper', 'cdf', 'ccdf', 'dcdf', 'dccdf', 'quantile',
    'sample', 'mean', 'fit',
]


@pytest.mark.parametrize('cls, kwargs', [
    (Normal, dict(mean=0.0, std=1.0)),
    (NegBinom, dict(mean=1.0, dispersion=1.0)),
])
def test_all_public_ranvar_methods_are_present(cls, kwargs):
    """Tests that neither subclass is missing a method RanVar exposes."""
    instance = cls(**kwargs)

    for name in PUBLIC_METHODS:
        assert hasattr(instance, name), f'{cls.__name__} is missing {name}()'
