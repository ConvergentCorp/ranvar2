import ranvar as mc
import numpy as np
import pytest


def makeSource(mean=100.0, spread=10.0, seed=31337):
    """Build a digest to draw from in a compiled model."""
    np.random.seed(seed)
    x = mc.Digest(maxBins=64)
    for d in np.random.randn(5_000)*spread + mean:
        x.add(d)
    return x


source = makeSource()
other  = makeSource(mean=20.0, spread=5.0, seed=99)
rate   = 0.05


# Compiling is slow enough that the models are shared across the tests that only
# need to look at a compiled result.

@mc.cfunc(samples=2_000)
def difference():
    return ~source - ~other


@mc.cfunc(samples=2_000)
def pair():
    p = ~source
    return p, p - ~other


def test_returns_a_ranvar():
    """Tests that a compiled model collects into a RanVar like func does."""
    assert isinstance(difference(), mc.RanVar)


def test_matches_the_interpreted_executor():
    """Tests that compiling a model does not change what it computes.

    The C loop and the Python one draw from the same generator, so seeding both
    the same way has to give the same simulation, not merely a similar one.
    """
    @mc.func(samples=2_000)
    def interpreted():
        return ~source - ~other

    mc.seed(4321)
    compiled = difference()

    mc.seed(4321)
    plain = interpreted()

    assert compiled.mean() == plain.mean()
    np.testing.assert_array_equal(compiled.getBins(), plain.getBins())
    np.testing.assert_array_equal(compiled.getWeights(), plain.getWeights())


def test_compilation_is_lazy_and_cached():
    """Tests that the model compiles on first use and not at decoration."""

    @mc.cfunc(samples=10)
    def model():
        return ~source

    assert model.module is None, 'decorating the model already compiled it'

    model()

    assert model.module is not None

    compiled = model.module
    model()

    assert model.module is compiled, 'the model was compiled a second time'


def test_sample_count():
    """Tests that a compiled model runs the requested number of samples."""

    @mc.cfunc(samples=250)
    def model():
        return ~source

    assert sum(model().getWeights()) == 250
    assert sum(model.withSamples(70)().getWeights()) == 70
    assert model.samples == 250


def test_withSamples_shares_the_compiled_module():
    """Tests that overriding the sample count does not compile the model again."""
    difference()

    smaller = difference.withSamples(50)

    assert smaller.module is difference.module
    assert sum(smaller().getWeights()) == 50


def test_maxBins_is_honoured():
    """Tests that the returned digest is built with the requested resolution."""

    @mc.cfunc(samples=2_000, maxBins=16)
    def model():
        return ~source

    assert model().getActiveBinCount() == 16


def test_arguments_reach_the_model():
    """Tests that a compiled model can take numbers from the call."""

    @mc.cfunc(samples=500)
    def scaled(factor, offset=1.0):
        return ~source * factor + offset

    doubled = scaled(2.0)
    shifted = scaled(1.0, offset=100.0)
    plain   = scaled(1.0, offset=0.0)

    assert abs(doubled.mean() - 2*plain.mean()) < 1e-6 * abs(plain.mean()) + 1.0
    assert abs((shifted.mean() - plain.mean()) - 100.0) < 1.0


def test_free_variables_are_reread_each_call():
    """Tests that rebinding a number the model reads is seen by the next call.

    The free names are passed into the compiled loop as arguments rather than
    burned in when it was built, so the model keeps Python's scoping.
    """
    global rate

    @mc.cfunc(samples=500)
    def model():
        return ~source * rate

    before = model().mean()

    rate = 0.5
    try:
        after = model().mean()
    finally:
        rate = 0.05

    assert abs(after / before - 10.0) < 0.5


def test_multiple_outputs():
    """Tests that a compiled model returning a tuple gives a RanVar each."""
    out = pair()

    assert isinstance(out, tuple)
    assert len(out) == 2
    assert all(isinstance(o, mc.RanVar) for o in out)

    first, second = out

    assert abs((first.mean() - second.mean()) - other.mean()) < 1.0


def test_branches_are_compiled():
    """Tests that if and else are translated rather than refused."""

    @mc.cfunc(samples=2_000)
    def clipped():
        v = ~source

        if v > 100.0:
            return 100.0
        else:
            return v

    out = clipped()

    assert out.upper() <= 100.0


def test_errors_from_the_digest_propagate():
    """Tests that an exception raised inside the C loop is not swallowed.

    The loop runs in C, where an error has to be checked for and passed back
    rather than simply travelling up the Python stack on its own.
    """
    empty = mc.Digest(maxBins=8)

    @mc.cfunc(samples=100)
    def model():
        return ~empty

    with pytest.raises(ValueError):
        model()


def test_rejects_unsupported_models():
    """Tests that anything outside the compilable subset is refused outright.

    cfunc exists to keep the loop out of the interpreter, so falling back to it
    quietly would leave a model looking fast and running slow.
    """
    import math

    text = 'not a number'
    things = [1.0, 2.0, 3.0]

    def callsAFunction():
        return math.sqrt(~source)

    def usesAString():
        return ~source + text

    def neverReturns():
        value = ~source

    def returnsNothing():
        return

    def branchesWithoutElse():
        if ~source > 1.0:
            return 1.0

    def disagreesOnWidth():
        if ~source > 1.0:
            return 1.0, 2.0
        return 3.0

    def chainsAComparison():
        return 1.0 if 0.0 < ~source < 5.0 else 2.0

    def usesAReservedName():
        _rv_value = ~source
        return _rv_value

    def usesFloorDivision():
        return ~source // 2.0

    def unpacksATuple():
        first, second = 1.0, 2.0
        return first + second

    def readsAnUndefinedName():
        return ~source + missing

    def loopsOverAList():
        total = 0.0
        for value in things:
            total += value
        return total

    def loopsOverTwoNames():
        total = 0.0
        for first, second in range(3):
            total += first
        return total

    def loopWithElse():
        total = 0.0
        for _ in range(3):
            total += ~source
        else:
            total += 1.0
        return total

    models = [
        callsAFunction, usesAString, neverReturns,
        returnsNothing, branchesWithoutElse, disagreesOnWidth,
        chainsAComparison, usesAReservedName, usesFloorDivision,
        unpacksATuple, readsAnUndefinedName,
        loopsOverAList, loopsOverTwoNames, loopWithElse,
    ]

    for model in models:
        with pytest.raises(mc.RanVarCompileError):
            mc.cfunc(model).compile()


def test_rejects_drawing_from_a_non_ranvar():
    """Tests that ~ is refused on anything that is not a RanVar."""
    number = 3.0

    def model():
        return ~number

    with pytest.raises(mc.RanVarCompileError):
        mc.cfunc(model).compile()


def test_rejects_bad_sample_counts():
    """Tests that the sample count is validated the way func's is."""
    for bad in [0, -1]:
        with pytest.raises(ValueError):
            mc.cfunc(samples=bad)

    for bad in [1.5, '10', None, True]:
        with pytest.raises(TypeError):
            mc.cfunc(samples=bad)


def test_rejects_a_non_callable():
    """Tests that the options form is not mistaken for the bare form."""
    with pytest.raises(TypeError):
        mc.cfunc(10_000)


def test_metadata_is_preserved():
    """Tests that the wrapper keeps the identity of the model it wraps."""

    @mc.cfunc
    def documented():
        """A documented model."""
        return ~source

    assert documented.__name__ == 'documented'
    assert documented.__doc__ == 'A documented model.'
    assert documented.__wrapped__.__name__ == 'documented'


def test_for_loop_is_compiled():
    """Tests that a for loop over range is translated rather than refused."""
    count = 4

    @mc.cfunc(samples=1_000)
    def summed():
        total = 0.0

        for i in range(count):
            total += ~source

        return total

    @mc.func(samples=1_000)
    def interpreted():
        total = 0.0

        for i in range(count):
            total += ~source

        return total

    mc.seed(2468)
    compiled = summed()

    mc.seed(2468)
    plain = interpreted()

    assert compiled.mean() == plain.mean()
    np.testing.assert_array_equal(compiled.getWeights(), plain.getWeights())


def test_loop_counter_is_usable_as_a_number():
    """Tests that the loop variable can be used in the model's arithmetic.

    Counters are integers where everything else in a compiled model is a
    double, so using one in an expression has to still give the right answer.
    """

    @mc.cfunc(samples=200)
    def weighted():
        total = 0.0

        for i in range(4):
            total += i * 2.0

        return total

    # 2*(0 + 1 + 2 + 3) is 12 on every run, so the digest holds one value.
    assert weighted().mean() == 12.0


def test_while_loop_is_compiled():
    """Tests that a while loop is translated rather than refused."""

    @mc.cfunc(samples=1_000)
    def accumulate():
        total = 0.0

        while total < 300.0:
            total += ~source

        return total

    @mc.func(samples=1_000)
    def interpreted():
        total = 0.0

        while total < 300.0:
            total += ~source

        return total

    mc.seed(1357)
    compiled = accumulate()

    mc.seed(1357)
    plain = interpreted()

    assert compiled.lower() >= 300.0
    assert compiled.mean() == plain.mean()
    np.testing.assert_array_equal(compiled.getWeights(), plain.getWeights())


def test_break_and_continue_are_compiled():
    """Tests that break and continue work inside a compiled loop."""

    @mc.cfunc(samples=500)
    def counted():
        total = 0.0
        seen = 0.0

        for i in range(10):
            if i < 2.0:
                continue

            if i > 5.0:
                break

            total += 1.0

        return total

    # i runs 2, 3, 4, 5 before the break, whatever the draws do.
    assert counted().mean() == 4.0


def test_elif_and_nested_ifs_are_compiled():
    """Tests that an elif chain and a nested if both translate.

    Python represents elif as an if nested in the else of the one before, so
    this leans on the same translation as a plain else.
    """

    @mc.cfunc(samples=2_000)
    def banded():
        value = ~source

        if value > 110.0:
            band = 3.0
        elif value > 100.0:
            band = 2.0
        else:
            band = 1.0

        return band

    out = banded()

    assert out.lower() >= 1.0
    assert out.upper() <= 3.0
