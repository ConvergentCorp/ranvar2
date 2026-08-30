import ranvar2 as mc
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


def test_multiple_outputs_give_a_ranvar_each():
    """Tests that a function returning a tuple is collected into one RanVar each."""
    src = makeSource()

    @mc.func(samples=200)
    def model():
        return ~src, ~src, ~src

    out = model()

    assert isinstance(out, tuple)
    assert len(out) == 3
    assert all(isinstance(o, mc.RanVar) for o in out)


def test_multiple_outputs_unpack():
    """Tests that the results unpack the way the return statement reads."""
    src = makeSource()

    @mc.func(samples=200)
    def model():
        return ~src, ~src

    first, second = model()

    assert isinstance(first, mc.RanVar)
    assert isinstance(second, mc.RanVar)


def test_each_output_collects_its_own_value():
    """Tests that values are collected into the digest for their own position."""
    src = makeSource()

    @mc.func(samples=200)
    def model():
        v = ~src
        return v, v + 1000.0

    low, high = model()

    # mean() is exact, so the offset survives the collection intact.
    assert abs((high.mean() - low.mean()) - 1000.0) < 1e-9


def test_namedtuple_is_preserved():
    """Tests that named outputs come back under the same names."""
    from collections import namedtuple

    Result = namedtuple('Result', 'revenue profit')
    src = makeSource()

    @mc.func(samples=200)
    def model():
        revenue = ~src
        return Result(revenue=revenue, profit=revenue - 50.0)

    out = model()

    assert isinstance(out, Result)
    assert out._fields == ('revenue', 'profit')
    assert isinstance(out.revenue, mc.RanVar)
    assert abs((out.revenue.mean() - out.profit.mean()) - 50.0) < 1e-9


def test_list_output_stays_a_list():
    """Tests that the results come back in the shape the function returned."""
    src = makeSource()

    @mc.func(samples=100)
    def model():
        return [~src, ~src]

    out = model()

    assert isinstance(out, list)
    assert len(out) == 2


def test_single_element_tuple_keeps_its_shape():
    """Tests that a one value tuple is not unwrapped into a bare RanVar."""
    src = makeSource()

    @mc.func(samples=100)
    def model():
        return (~src,)

    out = model()

    assert isinstance(out, tuple)
    assert len(out) == 1


def test_single_output_is_not_wrapped():
    """Tests that returning a bare number still gives a bare RanVar."""
    src = makeSource()

    @mc.func(samples=100)
    def model():
        return ~src

    assert isinstance(model(), mc.RanVar)


def test_multiple_outputs_run_the_function_once_per_sample():
    """Tests that deciding the shape does not cost an extra run.

    The first run is what reveals how many values there are, so it has to be
    counted as the first sample rather than run again.
    """
    for samples in [1, 2, 5, 100]:
        calls = []

        @mc.func(samples=samples)
        def model():
            calls.append(1)
            return 1.0, 2.0

        out = model()

        assert len(calls) == samples, \
            f'{samples} samples ran the function {len(calls)} times'

        for o in out:
            assert sum(o.getWeights()) == samples


def test_multiple_outputs_honour_maxBins():
    """Tests that maxBins applies to every returned RanVar."""
    src = makeSource()

    @mc.func(samples=2_000, maxBins=16)
    def model():
        return ~src, ~src

    assert [o.getActiveBinCount() for o in model()] == [16, 16]


def test_multiple_outputs_with_withSamples():
    """Tests that the sample count override works for multiple outputs too."""
    calls = []

    @mc.func(samples=9)
    def model():
        calls.append(1)
        return 1.0, 2.0

    out = model.withSamples(4)()

    assert len(calls) == 4
    assert len(out) == 2
    assert all(sum(o.getWeights()) == 4 for o in out)


def test_rejects_a_changing_number_of_outputs():
    """Tests that the shape has to be the same on every run.

    Collecting runs of different widths would quietly mix different quantities
    into the same distribution.
    """
    state = {'n': 0}

    @mc.func(samples=10)
    def widens():
        state['n'] += 1
        return (1.0, 2.0) if state['n'] < 5 else (1.0, 2.0, 3.0)

    with pytest.raises(ValueError):
        widens()

    state['n'] = 0

    @mc.func(samples=10)
    def toScalar():
        state['n'] += 1
        return (1.0, 2.0) if state['n'] < 5 else 1.0

    with pytest.raises(ValueError):
        toScalar()

    state['n'] = 0

    @mc.func(samples=10)
    def toTuple():
        state['n'] += 1
        return 1.0 if state['n'] < 5 else (1.0, 2.0)

    with pytest.raises(ValueError):
        toTuple()


def test_rejects_an_empty_result():
    """Tests that a function returning nothing to collect is refused."""

    @mc.func(samples=5)
    def model():
        return ()

    with pytest.raises(ValueError):
        model()


def test_rejects_a_non_numeric_output():
    """Tests that a bad value inside a tuple is reported, not just a bad tuple."""

    @mc.func(samples=5)
    def model():
        return 1.0, 'not a number'

    with pytest.raises(TypeError):
        model()

    @mc.func(samples=5)
    def nested():
        return 1.0, (2.0, 3.0)

    with pytest.raises(TypeError):
        nested()


def test_errors_from_the_model_are_not_swallowed():
    """Tests that an exception raised by the model itself propagates unchanged.

    The single output loop catches TypeError and ValueError to report an
    uncollectable result, so it has to tell those apart from the same errors
    coming out of the model.
    """

    @mc.func(samples=5)
    def raisesType():
        raise TypeError('inside the model')

    with pytest.raises(TypeError, match='inside the model'):
        raisesType()

    @mc.func(samples=5)
    def raisesValue():
        raise ValueError('inside the model')

    with pytest.raises(ValueError, match='inside the model'):
        raisesValue()


def test_decorating_a_method():
    """Tests that the wrapper binds as a method and reaches self.

    A plain function is a descriptor, and that is what supplies self. The
    wrapper is an object rather than a function, so binding is its own to do.
    """
    src = makeSource()

    class Model():
        def __init__(self, factor):
            self.factor = factor

        @mc.func(samples=200)
        def scaled(self):
            return ~src * self.factor

    single = Model(1.0)
    double = Model(2.0)

    mc.seed(31337)
    one = single.scaled()

    mc.seed(31337)
    two = double.scaled()

    assert two.mean() == pytest.approx(2*one.mean(), rel=1e-9)


def test_a_bound_method_keeps_the_wrapper_api():
    """Tests that binding does not lose withSamples or the wrapper's settings."""
    src = makeSource()

    class Model():
        @mc.func(samples=9, maxBins=16)
        def draw(self):
            return ~src

    model = Model()

    assert model.draw.samples == 9
    assert model.draw.maxBins == 16
    assert sum(model.draw().getWeights()) == 9
    assert sum(model.draw.withSamples(4)().getWeights()) == 4


def test_reaching_the_wrapper_through_the_class():
    """Tests that the class attribute is the wrapper itself, unbound.

    Reached that way the model keeps its own signature, so the instance is
    passed explicitly the way an ordinary function would take it.
    """
    src = makeSource()

    class Model():
        def __init__(self, factor):
            self.factor = factor

        @mc.func(samples=50)
        def scaled(self):
            return ~src * self.factor

    model = Model(3.0)

    assert isinstance(Model.scaled, mc.decorators.MonteCarlo)
    assert sum(Model.scaled(model).getWeights()) == 50
