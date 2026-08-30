import cython
import math
import pickle

import numpy as np
import pytest

import ranvar2 as mc
from ranvar2 import RanVar, Normal, NegBinom, Constant


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


def test_negbinom_accepts_the_r_p_parameterization():
    """Tests that r/p is converted to mean/dispersion at construction.

    mean = r(1 - p)/p and dispersion = 1/r, the NB2 form of the textbook
    parameterization, and nothing downstream sees r or p.
    """
    nb = NegBinom(r=4.0, p=0.25)

    assert nb.mean()       == pytest.approx(4.0 * 0.75 / 0.25)
    assert nb.dispersion() == pytest.approx(1.0 / 4.0)


def test_negbinom_r_p_sample_matches_the_textbook_moments():
    """Tests that draws from an r/p NegBinom recover r(1 - p)/p and
    r(1 - p)/p**2, the mean and variance that parameterization defines.
    """
    mc.seed(8)

    r, p = 5.0, 0.4
    nb   = NegBinom(r=r, p=p)
    xs   = np.array([nb.sample() for _ in range(200_000)])

    assert xs.mean() == pytest.approx(r * (1 - p) / p, rel=0.05)
    assert xs.var()  == pytest.approx(r * (1 - p) / p ** 2, rel=0.08)


def test_negbinom_r_p_pickles_as_mean_and_dispersion():
    """Tests that an r/p NegBinom round trips, since it is stored (and so
    reconstructed) in the mean/dispersion form.
    """
    nb  = NegBinom(r=3.0, p=0.5)
    nb2 = pickle.loads(pickle.dumps(nb))

    assert type(nb2) is NegBinom
    assert nb2.mean()       == pytest.approx(nb.mean())
    assert nb2.dispersion() == pytest.approx(nb.dispersion())


def test_negbinom_p_of_one_is_degenerate():
    """Tests that p = 1 gives the distribution that always draws zero, the
    boundary of the r/p range.
    """
    mc.seed(9)

    nb = NegBinom(r=2.0, p=1.0)

    assert nb.mean() == 0.0
    assert all(nb.sample() == 0.0 for _ in range(1_000))


@pytest.mark.parametrize('kwargs', [
    dict(mean=2.0, r=3.0, p=0.5),         # both parameterizations at once
    dict(dispersion=1.0, r=3.0, p=0.5),
    dict(r=3.0),                          # only half of r/p
    dict(p=0.5),
    dict(r=0.0, p=0.5),                   # r out of range
    dict(r=-1.0, p=0.5),
    dict(r=3.0, p=0.0),                   # p out of range
    dict(r=3.0, p=1.5),
    dict(r=3.0, p=-0.5),
])
def test_negbinom_rejects_invalid_r_p(kwargs):
    """Tests that the two parameterizations cannot be mixed, that r/p has to
    be given as a pair, and that both are range checked.
    """
    with pytest.raises(ValueError):
        NegBinom(**kwargs)


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


# Constant. ------------------------------------------------------------------

def test_constant_is_a_ranvar():
    """Tests that Constant can be used anywhere a RanVar is expected."""
    assert isinstance(Constant(value=1.0), RanVar)


def test_constant_accessors():
    """Tests that value()/mean() return the constructor's parameter."""
    c = Constant(value=7.5)

    assert c.value() == 7.5
    assert c.mean()  == 7.5


def test_constant_defaults_to_zero():
    """Tests that a Constant built with no arguments sits at 0.0."""
    assert Constant().sample() == 0.0


def test_constant_sample_never_varies():
    """Tests that every draw returns the same value, seed regardless."""
    mc.seed(6)

    c = Constant(value=-3.25)

    assert all(c.sample() == -3.25 for _ in range(1_000))


def test_constant_pickles_to_the_same_class():
    """Tests that a Constant round trips through pickle as a Constant, with
    its value intact, rather than unpickling to the plain RanVar base.
    """
    c  = Constant(value=4.0)
    c2 = pickle.loads(pickle.dumps(c))

    assert type(c2) is Constant
    assert c2.value() == 4.0

    c3 = pickle.loads(pickle.dumps(c2))
    assert type(c3) is Constant


def test_constant_compiles_through_a_ranvar_annotated_model():
    """Tests that a compiled model draws from Constant's own sample()."""
    mc.seed(7)

    out = compiledIdentity(Constant(value=12.0))

    mean, var = empiricalMoments(out)

    assert mean == pytest.approx(12.0)
    assert var  == pytest.approx(0.0, abs=1e-9)


def test_constant_lower_upper_are_the_value():
    """Tests that the support is the single point, with no data added."""
    c = Constant(value=2.0)

    assert c.lower() == 2.0
    assert c.upper() == 2.0


def test_constant_quantile_is_the_value_everywhere():
    """Tests that every quantile of a degenerate distribution is its point,
    including at and outside the ends of the probability range.
    """
    c = Constant(value=5.0)

    for p in (0.0, 0.01, 0.5, 0.99, 1.0, -1.0, 2.0):
        assert c.quantile(p) == 5.0


def test_constant_cdf_is_a_step():
    """Tests that cdf()/ccdf() step at the value rather than reading the
    (empty) inherited digest.
    """
    c = Constant(value=5.0)

    assert c.cdf(4.999) == 0.0
    assert c.cdf(5.0)   == 1.0
    assert c.cdf(5.001) == 1.0

    assert c.ccdf(4.999) == 1.0
    assert c.ccdf(5.0)   == 0.0
    assert c.ccdf(5.001) == 0.0


@pytest.mark.parametrize('method', ['dcdf', 'dccdf'])
def test_constant_has_no_density(method):
    """Tests that the density methods refuse, since a step CDF has no
    derivative at the point it steps at.
    """
    c = Constant(value=5.0)

    with pytest.raises(NotImplementedError):
        getattr(c, method)(5.0)


def test_constant_add_replaces_the_value():
    """Tests that add() sets the value, unlike Normal/NegBinom, since one
    point does fully determine a constant.
    """
    c = Constant(value=1.0)
    c.add(8.0)

    assert c.value()  == 8.0
    assert c.sample() == 8.0


def test_constant_fit_uses_the_sample_mean():
    """Tests that fit() replaces the value with the mean of the data, and
    that a single point is enough, unlike Normal/NegBinom.
    """
    c = Constant(value=0.0)
    c.fit([2.0, 4.0, 6.0])

    assert c.value() == pytest.approx(4.0)

    c.fit([9.0])

    assert c.value() == pytest.approx(9.0)


def test_constant_fit_needs_at_least_one_point():
    """Tests that fit() refuses an empty collection."""
    with pytest.raises(ValueError):
        Constant().fit([])


# Every public RanVar method is present, whether inherited unchanged (the
# ones with no data dependency), overridden analytically, or overridden to
# refuse. Digest-only accessors (getActiveBinCount/getBins/getWeights) are
# not part of this contract, since add()/fit() no longer build a digest for
# any of these subclasses. ------------------------------------------------

PUBLIC_METHODS = [
    'add', 'lower', 'upper', 'cdf', 'ccdf', 'dcdf', 'dccdf', 'quantile',
    'sample', 'mean', 'fit',
]


@pytest.mark.parametrize('cls, kwargs', [
    (Normal, dict(mean=0.0, std=1.0)),
    (NegBinom, dict(mean=1.0, dispersion=1.0)),
    (Constant, dict(value=1.0)),
])
def test_all_public_ranvar_methods_are_present(cls, kwargs):
    """Tests that no subclass is missing a method RanVar exposes."""
    instance = cls(**kwargs)

    for name in PUBLIC_METHODS:
        assert hasattr(instance, name), f'{cls.__name__} is missing {name}()'
