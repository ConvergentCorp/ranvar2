import json

import numpy as np
import pytest

import ranvar2 as mc
from ranvar2 import RanVar, Normal, NegBinom, Constant


def makeDigest(maxBins=16):
    """Build a fitted t-digest to serialise."""
    digest = RanVar(maxBins=maxBins)
    digest.fit([1.0, 2.0, 3.0, 4.0, 5.0])

    return digest


# Every payload is JSON, and names what wrote it. -----------------------------

@pytest.mark.parametrize('instance', [
    makeDigest(),
    Normal(mean=2.0, std=3.0),
    NegBinom(mean=4.0, dispersion=0.5),
    Constant(value=9.5),
])
def test_todict_is_json(instance):
    """Tests that a payload survives a round trip through JSON, so it can be
    stored anywhere plain types can.
    """
    payload = instance.toDict()

    assert json.loads(json.dumps(payload)) == payload
    assert payload['type'] == type(instance).__name__


# fromDict rebuilds the class that wrote the payload. -------------------------

def test_ranvar_round_trips_its_centroids():
    """Tests that a fitted digest comes back with the same centroids, and so
    answers quantile() the same way.
    """
    digest = makeDigest()
    back   = RanVar.fromDict(digest.toDict())

    assert type(back) is RanVar
    assert back.getActiveBinCount() == digest.getActiveBinCount()
    assert np.allclose(back.getBins(), digest.getBins())
    assert np.allclose(back.getWeights(), digest.getWeights())
    assert back.quantile(0.5) == pytest.approx(digest.quantile(0.5))
    assert back.mean() == pytest.approx(digest.mean())


def test_normal_round_trips_as_a_normal():
    """Tests that a Normal comes back a Normal, parameters intact, rather
    than as the bare RanVar base.
    """
    back = RanVar.fromDict(Normal(mean=2.0, std=3.0).toDict())

    assert type(back) is Normal
    assert back.mean() == 2.0
    assert back.std()  == 3.0


def test_negbinom_round_trips_as_a_negbinom():
    """Tests that a NegBinom comes back a NegBinom, in the mean/dispersion
    form it stores whichever parameterization built it.
    """
    back = RanVar.fromDict(NegBinom(r=4.0, p=0.25).toDict())

    assert type(back) is NegBinom
    assert back.mean()       == pytest.approx(12.0)
    assert back.dispersion() == pytest.approx(0.25)


def test_constant_round_trips_as_a_constant():
    """Tests that a Constant comes back a Constant, on the same point."""
    back = RanVar.fromDict(Constant(value=9.5).toDict())

    assert type(back) is Constant
    assert back.value() == 9.5


def test_maxbins_survives_the_round_trip():
    """Tests that the bin ceiling is carried, not defaulted, since a digest
    restored with a different one would merge differently from here on.
    """
    back = RanVar.fromDict(makeDigest(maxBins=64).toDict())

    back.fit(np.random.randn(500))

    assert back.getActiveBinCount() <= 64
    assert back.getActiveBinCount() > 32


def test_restored_digest_keeps_taking_points():
    """Tests that a restored digest is a working digest rather than an inert
    copy of one: the cached quantile ladder has to be rebuilt from the
    centroids that came back.
    """
    back = RanVar.fromDict(makeDigest().toDict())

    back.add(6.0)

    assert back.getActiveBinCount() == 6
    assert back.upper() == 6.0


# Dispatch. -------------------------------------------------------------------

def test_fromdict_finds_a_subclass_defined_elsewhere():
    """Tests that dispatch walks the live subclass tree rather than a fixed
    registry, so a subclass defined outside the package is found too.
    """
    class Doubled(Constant):
        """A Constant that draws twice its value."""

        def sample(self):
            return 2.0 * self.value()

    back = RanVar.fromDict(Doubled(value=3.0).toDict())

    assert type(back) is Doubled
    assert back.sample() == 6.0


def test_fromdict_through_a_subclass_accepts_its_own():
    """Tests that Normal.fromDict() rebuilds a Normal payload."""
    assert type(Normal.fromDict(Normal(mean=1.0, std=1.0).toDict())) is Normal


def test_fromdict_through_a_subclass_refuses_another():
    """Tests that Normal.fromDict() refuses a payload written by something
    that is not a Normal, rather than quietly returning another type.
    """
    with pytest.raises(ValueError):
        Normal.fromDict(Constant(value=1.0).toDict())


@pytest.mark.parametrize('payload', [
    {},                                  # no type key
    {'nActive': 0},
    'not a dict',
    None,
    {'type': 'NoSuchDistribution'},      # a type key naming nothing
    {'type': 'RanVarArray'},             # a real class, but not a RanVar
])
def test_fromdict_refuses_a_payload_it_cannot_place(payload):
    """Tests that a payload naming no known RanVar raises rather than
    guessing at a class to rebuild.
    """
    with pytest.raises(ValueError):
        RanVar.fromDict(payload)


@pytest.mark.parametrize('missing', ['mean', 'std', 'maxBins'])
def test_fromdict_reports_a_missing_field(missing):
    """Tests that a payload short of a field the class needs says which one,
    rather than raising a bare KeyError from inside the constructor.
    """
    payload = Normal(mean=1.0, std=2.0).toDict()

    del payload[missing]

    with pytest.raises(ValueError, match=missing):
        RanVar.fromDict(payload)


@pytest.mark.parametrize('edit', [
    lambda p: p.update(nActive=p['nActive'] + 1),      # more centroids than carried
    lambda p: p.update(bins=p['bins'][:-1]),           # fewer centroids than claimed
    lambda p: p.update(weights=p['weights'][:-1]),     # weights out of step with them
    lambda p: p.update(maxBins=2),                     # ceiling below the count
])
def test_fromdict_refuses_inconsistent_centroids(edit):
    """Tests that a payload whose centroids do not describe the digest it
    claims raises, rather than restoring a digest that is quietly wrong.
    """
    payload = makeDigest().toDict()

    edit(payload)

    with pytest.raises(ValueError):
        RanVar.fromDict(payload)


# The parametric subclasses carry parameters, not centroids. ------------------

@pytest.mark.parametrize('instance, expected', [
    (Normal(mean=2.0, std=3.0), {'mean', 'std'}),
    (NegBinom(mean=4.0, dispersion=0.5), {'mean', 'dispersion'}),
    (Constant(value=9.5), {'value'}),
])
def test_parametric_payloads_carry_their_parameters(instance, expected):
    """Tests that each subclass writes its own parameters and no centroids,
    since neither add() nor fit() builds a digest for any of them.
    """
    payload = instance.toDict()

    assert expected <= set(payload)
    assert 'bins' not in payload
    assert 'weights' not in payload


def test_restored_distribution_still_compiles_into_a_model():
    """Tests that what fromDict() returns is a working model input: sampling
    from it has to reach the subclass's own sample() through the vtable, the
    same as the instance it was rebuilt from.
    """
    import cython

    def identity(x: mc.RanVar) -> cython.double:
        return ~x

    compiled = mc.cfunc(identity, samples=50_000)

    mc.seed(11)

    back = RanVar.fromDict(Normal(mean=50.0, std=1.0).toDict())
    out  = compiled(back)

    assert out.mean() == pytest.approx(50.0, rel=0.01)
