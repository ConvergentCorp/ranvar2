import cython
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


def test_normal_pickles():
    """Tests that a Normal round trips through pickle with its parameters."""
    n  = Normal(mean=7.0, std=2.0)
    n2 = pickle.loads(pickle.dumps(n))

    assert n2.mean() == 7.0
    assert n2.std() == 2.0
    assert isinstance(n2, Normal)


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


def test_negbinom_pickles():
    """Tests that a NegBinom round trips through pickle with its parameters."""
    nb  = NegBinom(mean=9.0, dispersion=2.0)
    nb2 = pickle.loads(pickle.dumps(nb))

    assert nb2.mean() == 9.0
    assert nb2.dispersion() == 2.0
    assert isinstance(nb2, NegBinom)


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
