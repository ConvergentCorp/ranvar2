import ranvar as mc
import numpy as np
import pytest


def makeSource():
    """Build a digest to draw from, so simulations have something to sample."""
    np.random.seed(31337)
    x = mc.Digest(maxBins=64)
    for d in np.random.randn(10_000)*100 + 100:
        x.add(d)
    return x


def makeDie():
    """Build a digest approximating a six sided die."""
    x = mc.Digest(maxBins=8)
    x.fit([1, 2, 3, 4, 5, 6]*100)
    return x


def test_returns_a_ranvar():
    """Tests that calling a decorated function returns the collected distribution."""
    src = makeSource()

    @mc.func(samples=100)
    def model():
        return ~src

    out = model()

    assert isinstance(out, mc.RanVar)


def test_decoration_is_lazy():
    """Tests that the simulation runs on call, not at decoration time."""
    calls = []

    @mc.func(samples=5)
    def model():
        calls.append(1)
        return 1.0

    assert calls == [], 'decorating the function already ran it'

    model()

    assert len(calls) == 5


def test_default_sample_count():
    """Tests that a plain call uses the default number of samples."""
    calls = []

    @mc.func
    def model():
        calls.append(1)
        return 1.0

    model()

    assert model.samples == mc.decorators.DEFAULT_SAMPLES
    assert len(calls) == mc.decorators.DEFAULT_SAMPLES


def test_samples_option():
    """Tests that the decorator's samples option sets the default for the model."""
    calls = []

    @mc.func(samples=7)
    def model():
        calls.append(1)
        return 1.0

    model()

    assert model.samples == 7
    assert len(calls) == 7


def test_withSamples_overrides_the_count():
    """Tests running a model with a different number of samples."""
    calls = []

    @mc.func(samples=7)
    def model():
        calls.append(1)
        return 1.0

    model.withSamples(3)()

    assert len(calls) == 3
    assert model.samples == 7, 'the override leaked into the original wrapper'

    calls.clear()
    model()

    assert len(calls) == 7, 'the default call was affected by the override'


def test_withSamples_is_reusable():
    """Tests that the wrapper withSamples() returns can be kept and called again."""
    calls = []

    @mc.func(samples=7, maxBins=16)
    def model():
        calls.append(1)
        return 1.0

    small = model.withSamples(2)
    small()
    small()

    assert len(calls) == 4
    assert small.samples == 2
    assert small.maxBins == model.maxBins, 'withSamples() dropped maxBins'


def test_maxBins_option():
    """Tests that maxBins is passed through to the returned RanVar."""
    src = makeSource()

    @mc.func(samples=500, maxBins=16)
    def model():
        return ~src

    assert model().getActiveBinCount() == 16


def test_arguments_are_forwarded():
    """Tests that call arguments reach the wrapped function on every run."""
    seen = []

    @mc.func(samples=4)
    def model(a, b=0):
        seen.append((a, b))
        return a + b

    out = model(3, b=4)

    assert seen == [(3, 4)]*4
    assert out.mean() == 7


def test_samples_does_not_shadow_the_wrapped_signature():
    """Tests that a model may have its own parameter named samples.

    The sample count is deliberately not a keyword argument on the wrapper, so
    a model is free to use that name for its own purposes.
    """
    seen = []

    @mc.func(samples=3)
    def model(samples):
        seen.append(samples)
        return float(samples)

    out = model(samples=99)

    assert seen == [99]*3, 'the wrapper intercepted the argument'
    assert out.mean() == 99


def test_simulation_approximates_the_distribution():
    """Tests that the collected RanVar describes the simulated quantity."""
    die = makeDie()
    mc.seed(2024)

    @mc.func(samples=20_000)
    def roll():
        return ~die + ~die

    out = roll()

    assert abs(out.mean() - 7.0) < 0.2
    assert out.lower() >= 2.0
    assert out.upper() <= 12.0
    assert out.quantile(0.5) == pytest.approx(7.0, abs=0.5)


def test_metadata_is_preserved():
    """Tests that the wrapper keeps the identity of the function it wraps."""

    @mc.func
    def documented():
        """A documented model."""
        return 1.0

    assert documented.__name__ == 'documented'
    assert documented.__doc__ == 'A documented model.'
    assert documented.__wrapped__.__name__ == 'documented'


def test_rejects_bad_sample_counts():
    """Tests that an unusable sample count is refused at decoration time."""
    for bad in [0, -1]:
        with pytest.raises(ValueError):
            mc.func(samples=bad)

    for bad in [1.5, '10', None, True]:
        with pytest.raises(TypeError):
            mc.func(samples=bad)

    @mc.func
    def model():
        return 1.0

    with pytest.raises(ValueError):
        model.withSamples(0)


def test_rejects_a_non_callable():
    """Tests that the options form is not mistaken for the bare form."""
    with pytest.raises(TypeError):
        mc.func(10_000)


def test_rejects_a_non_numeric_result():
    """Tests that a model returning something uncollectable fails clearly."""

    @mc.func(samples=3)
    def model():
        return 'not a number'

    with pytest.raises(TypeError):
        model()
