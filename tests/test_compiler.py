import cython
import math

import ranvar2 as mc
import numpy as np
import pytest


def makeSource(mean=100.0, spread=10.0, seed=31337):
    """Build a digest for a compiled model to draw from."""
    np.random.seed(seed)
    x = mc.Digest(maxBins=64)
    for d in np.random.randn(5_000)*spread + mean:
        x.add(d)
    return x


source = makeSource()
other  = makeSource(mean=20.0, spread=5.0, seed=99)
rate   = 0.05


# Models are written once and shared, since each distinct one costs a compile.

def difference(revenue: mc.RanVar, cost: mc.RanVar) -> cython.double:
    """A model with every type annotated."""
    price: cython.double = ~revenue
    spend: cython.double = ~cost
    return price - spend


def pair(a: mc.RanVar, b: mc.RanVar):
    """A model returning two values, so it has no scalar return type."""
    first: cython.double = ~a
    return first, first - ~b


compiled = mc.cfunc(difference, samples=2_000)
compiledPair = mc.cfunc(pair, samples=2_000)


def test_returns_a_ranvar():
    """Tests that a compiled model collects into a RanVar like func does."""
    assert isinstance(compiled(source, other), mc.RanVar)


def test_matches_the_interpreted_executor():
    """Tests that compiling a model does not change what it computes.

    The same function is run both ways, so seeding them alike has to give the
    same simulation rather than merely a similar one.
    """
    interpreted = mc.func(difference, samples=2_000)

    mc.seed(4321)
    fast = compiled(source, other)

    mc.seed(4321)
    plain = interpreted(source, other)

    assert fast.mean() == plain.mean()
    np.testing.assert_array_equal(fast.getBins(), plain.getBins())
    np.testing.assert_array_equal(fast.getWeights(), plain.getWeights())


def test_compilation_is_lazy_and_cached():
    """Tests that the model compiles on first use and not at decoration."""

    def model(x: mc.RanVar) -> cython.double:
        return ~x

    wrapped = mc.cfunc(model, samples=10)

    assert wrapped.module is None, 'decorating the model already compiled it'

    wrapped(source)

    assert wrapped.module is not None

    built = wrapped.module
    wrapped(source)

    assert wrapped.module is built, 'the model was compiled a second time'


def test_sample_count():
    """Tests that a compiled model runs the requested number of samples."""
    assert sum(compiled(source, other).getWeights()) == 2_000

    smaller = compiled.withSamples(70)

    assert sum(smaller(source, other).getWeights()) == 70
    assert smaller.module is compiled.module, 'the override recompiled the model'
    assert compiled.samples == 2_000


def test_maxBins_is_honoured():
    """Tests that the returned digest is built with the requested resolution."""
    coarse = mc.cfunc(difference, samples=2_000, maxBins=16)

    assert coarse(source, other).getActiveBinCount() == 16


def test_numeric_arguments():
    """Tests that a compiled model can take plain numbers alongside digests."""

    def scaled(x: mc.RanVar, factor: cython.double) -> cython.double:
        return ~x * factor

    wrapped = mc.cfunc(scaled, samples=500)

    # Seeded alike, both runs draw the same values, so the factor comes through
    # exactly rather than only on average.
    mc.seed(11)
    single = wrapped(source, 1.0)

    mc.seed(11)
    double = wrapped(source, 2.0)

    assert double.mean() == pytest.approx(2*single.mean(), rel=1e-9)


def test_multiple_outputs():
    """Tests that a model returning a tuple gives a RanVar for each value."""
    out = compiledPair(source, other)

    assert isinstance(out, tuple)
    assert len(out) == 2
    assert all(isinstance(o, mc.RanVar) for o in out)

    first, second = out

    assert abs((first.mean() - second.mean()) - other.mean()) < 1.0


def test_module_names_are_carried_across():
    """Tests that a model may use what its own module has imported.

    The generated module has a namespace of its own, so anything the model
    reads from the module it was written in has to be put there.
    """

    def usesModule(x: mc.RanVar) -> cython.double:
        return math.sqrt(abs(~x)) * rate

    wrapped = mc.cfunc(usesModule, samples=500)

    assert wrapped(source).mean() > 0.0


def test_module_names_are_reread_each_call():
    """Tests that rebinding a name the model reads is seen by the next call."""
    global rate

    def scaledByGlobal(x: mc.RanVar) -> cython.double:
        return ~x * rate

    wrapped = mc.cfunc(scaledByGlobal, samples=500)

    before = wrapped(source).mean()

    rate = 0.5
    try:
        after = wrapped(source).mean()
    finally:
        rate = 0.05

    assert abs(after / before - 10.0) < 0.5


def test_rejects_a_digest_read_from_the_module():
    """Tests that drawing from a digest that is not a parameter is refused.

    A module level name is a Python object to Cython whatever it holds, so such
    a draw would compile but would not be a C call, which is the one thing cfunc
    exists to guarantee.
    """

    def readsGlobal() -> cython.double:
        return ~source

    with pytest.raises(mc.RanVarCompileError):
        mc.cfunc(readsGlobal).compile()


def test_invert_is_only_rewritten_for_digests():
    """Tests that ~ on an integer still means bitwise not.

    A draw is rewritten to sample() because the invert slot cannot return a C
    number. Only parameters annotated as digests are rewritten, so ~ keeps its
    ordinary meaning everywhere else.
    """

    def bitwise(x: mc.RanVar) -> cython.double:
        n: cython.int = 5
        return float(~n) + ~x * 0.0

    wrapped = mc.cfunc(bitwise, samples=50)

    assert wrapped(source).mean() == float(~5)


def test_ordinary_python_compiles():
    """Tests that constructs with no C equivalent still compile and run.

    The model's own source is what gets compiled, so anything Python allows is
    allowed here; annotations decide how much of it becomes C rather than
    whether it is accepted at all.
    """

    def withLoops(x: mc.RanVar) -> cython.double:
        total: cython.double = 0.0

        for _ in range(3):
            total += ~x

        while total < 400.0:
            total += 1.0

        return total

    def withCall(x: mc.RanVar) -> cython.double:
        return math.sqrt(abs(~x))

    def withComprehension(x: mc.RanVar) -> cython.double:
        return float(sum([x.sample() for _ in range(3)]))

    def withTry(x: mc.RanVar) -> cython.double:
        try:
            return ~x
        except ZeroDivisionError:
            return 0.0

    for model in [withLoops, withCall, withComprehension, withTry]:
        out = mc.cfunc(model, samples=200)(source)

        assert isinstance(out, mc.RanVar), f'{model.__name__} did not run'


def test_errors_from_the_digest_propagate():
    """Tests that an exception raised inside the C loop is not swallowed."""
    empty = mc.Digest(maxBins=8)

    def model(x: mc.RanVar) -> cython.double:
        return ~x

    wrapped = mc.cfunc(model, samples=100)

    with pytest.raises(ValueError):
        wrapped(empty)


def test_rejects_a_signature_the_driver_cannot_mirror():
    """Tests that a model taking *args or **kwargs is refused."""

    def starred(x: mc.RanVar, *rest) -> cython.double:
        return ~x

    def keyworded(x: mc.RanVar, **rest) -> cython.double:
        return ~x

    for model in [starred, keyworded]:
        with pytest.raises(mc.RanVarCompileError):
            mc.cfunc(model).compile()


def test_rejects_a_reserved_parameter_name():
    """Tests that a parameter the generated module would shadow is refused."""

    def model(_rv_out: mc.RanVar) -> cython.double:
        return ~_rv_out

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
    def documented(x: mc.RanVar) -> cython.double:
        """A documented model."""
        return ~x

    assert documented.__name__ == 'documented'
    assert documented.__doc__ == 'A documented model.'
    assert documented.__wrapped__.__name__ == 'documented'


def test_decorating_a_method():
    """Tests that a compiled model can be a method.

    self stays an ordinary Python object, which costs nothing while the model
    does not read digests off it: it is touched once per call rather than once
    per sample.
    """

    class Model():
        def __init__(self, factor):
            self.factor = factor

        @mc.cfunc(samples=500)
        def scaled(self, x: mc.RanVar, factor: cython.double) -> cython.double:
            return ~x * factor

    model = Model(2.0)

    mc.seed(31337)
    single = model.scaled(source, 1.0)

    mc.seed(31337)
    double = model.scaled(source, 2.0)

    assert double.mean() == pytest.approx(2*single.mean(), rel=1e-9)
    assert model.scaled.samples == 500
    assert sum(model.scaled.withSamples(20)(source, 1.0).getWeights()) == 20


def test_rejects_drawing_from_an_attribute():
    """Tests that ~self.digest is refused the way a module level digest is.

    An attribute is a Python object to Cython whatever it holds, so the draw
    would compile without being the C call the decorator is for. There is no
    value to inspect at compile time, so it is the shape that is judged.
    """

    class Model():
        def __init__(self, digest):
            self.digest = digest

        @mc.cfunc(samples=10)
        def drawsFromSelf(self) -> cython.double:
            return ~self.digest

    model = Model(source)

    with pytest.raises(mc.RanVarCompileError):
        model.drawsFromSelf()


# RanVarArray. -------------------------------------------------------------

def makeArray():
    """Build a small RanVarArray of parametric digests to draw from."""
    return mc.RanVarArray.fromList([
        mc.Normal(100.0, 5.0),
        mc.Normal(200.0, 10.0),
        mc.Normal(50.0, 2.0),
    ])


def total(regions: mc.RanVarArray) -> cython.double:
    """A model that draws from every element of a RanVarArray with ~."""
    som: cython.double = 0.0

    for region in regions:
        som += ~region

    return som


def totalBySample(regions: mc.RanVarArray) -> cython.double:
    """The same model, drawing with .sample() instead of ~."""
    som: cython.double = 0.0

    for region in regions:
        som += region.sample()

    return som


def test_iterates_a_ranvar_array():
    """Tests that a compiled model can draw from every element of an array."""
    result = mc.cfunc(total, samples=50_000)(makeArray())

    assert isinstance(result, mc.RanVar)
    assert result.mean() == pytest.approx(350.0, abs=2.0)


def test_sample_method_works_the_same_as_invert_in_a_loop():
    """Tests that .sample() and ~ compile to the same thing inside the loop."""
    mc.seed(2024)
    viaInvert = mc.cfunc(total, samples=5_000)(makeArray())

    mc.seed(2024)
    viaSample = mc.cfunc(totalBySample, samples=5_000)(makeArray())

    assert viaInvert.mean() == viaSample.mean()


def test_array_loop_matches_the_interpreted_executor():
    """Tests that compiling a RanVarArray loop does not change the result."""
    arr = makeArray()

    mc.seed(13)
    fast = mc.cfunc(total, samples=5_000)(arr)

    mc.seed(13)
    plain = mc.func(total, samples=5_000)(arr)

    assert fast.mean() == plain.mean()


def test_array_loop_compiles_to_an_indexed_c_loop():
    """Tests that `for x in arr:` is rewritten away from generic iteration.

    The rewritten source should index through arr._at() rather than iterating
    arr directly, since that is what lets the fetch -- and the sample() call
    that follows -- compile to a direct C call instead of the generic (Python
    object producing) iteration protocol.
    """
    model = mc.cfunc(total).compile().model

    assert 'for region in regions' not in model.source
    assert 'regions._at(' in model.source
    assert 'region: RanVar' in model.source
    assert 'region.sample()' in model.source


def test_rejects_drawing_from_a_ranvar_array_directly():
    """Tests that ~arr is refused: an array has no single value to draw."""

    def drawsArrayDirectly(regions: mc.RanVarArray) -> cython.double:
        return ~regions

    with pytest.raises(mc.RanVarCompileError):
        mc.cfunc(drawsArrayDirectly).compile()


def test_rejects_unpacking_an_array_loop():
    """Tests that a RanVarArray can only be iterated one element at a time."""

    def unpacksTwo(regions: mc.RanVarArray) -> cython.double:
        som: cython.double = 0.0

        for a, b in regions:
            som += 1.0

        return som

    with pytest.raises(mc.RanVarCompileError):
        mc.cfunc(unpacksTwo).compile()


def test_rejects_a_ranvar_array_read_from_the_module():
    """Tests that drawing from an array that is not a parameter is refused."""
    moduleArray = makeArray()

    def readsGlobalArray() -> cython.double:
        som: cython.double = 0.0

        for r in moduleArray:
            som += ~r

        return som

    with pytest.raises(mc.RanVarCompileError):
        mc.cfunc(readsGlobalArray).compile()


def totalByIndex(arr: mc.RanVarArray) -> cython.double:
    """A model that draws from a fixed number of indexed elements with ~."""
    som: cython.double = 0.0

    for i in range(3):
        som += ~arr[i]

    return som


def totalByIndexSample(arr: mc.RanVarArray) -> cython.double:
    """The same model, drawing with .sample() instead of ~."""
    som: cython.double = 0.0

    for i in range(3):
        som += arr[i].sample()

    return som


def test_indexes_a_ranvar_array():
    """Tests that a compiled model can draw from indexed elements of an array."""
    result = mc.cfunc(totalByIndex, samples=50_000)(makeArray())

    assert isinstance(result, mc.RanVar)
    assert result.mean() == pytest.approx(350.0, abs=2.0)


def test_index_access_compiles_to_a_direct_at_call():
    """Tests that arr[i] is rewritten away from the generic __getitem__.

    The rewritten source should call arr._at() directly rather than
    subscripting arr, since that is what lets the fetch -- and the sample()
    call that follows -- compile to a C call instead of the generic (Python
    object producing, and here also boxed-index) subscript protocol.
    """
    model = mc.cfunc(totalByIndex).compile().model

    assert 'arr[i]' not in model.source
    assert 'arr._at(i)' in model.source
    assert 'arr._at(i).sample()' in model.source


def test_index_access_matches_the_interpreted_executor():
    """Tests that compiling an indexed draw does not change the result."""
    arr = makeArray()

    mc.seed(21)
    fast = mc.cfunc(totalByIndex, samples=5_000)(arr)

    mc.seed(21)
    plain = mc.func(totalByIndex, samples=5_000)(arr)

    assert fast.mean() == plain.mean()


def test_index_access_invert_matches_sample_method():
    """Tests that ~arr[i] and arr[i].sample() compile to the same thing."""
    arr = makeArray()

    mc.seed(99)
    viaInvert = mc.cfunc(totalByIndex, samples=5_000)(arr)

    mc.seed(99)
    viaSample = mc.cfunc(totalByIndexSample, samples=5_000)(arr)

    assert viaInvert.mean() == viaSample.mean()


def test_index_access_supports_negative_indices():
    """Tests that arr[-1] counts from the end, like a plain list."""

    def last(arr: mc.RanVarArray) -> cython.double:
        return ~arr[-1]

    result = mc.cfunc(last, samples=20_000)(makeArray())

    assert result.mean() == pytest.approx(50.0, abs=2.0)


def test_index_access_raises_for_an_out_of_range_index():
    """Tests that an out of range index raises rather than reading garbage.

    arr._at() keeps bounds checking on, unlike the internal methods elsewhere
    in ranvar.py, because it is reached from a user's own index expression
    rather than only from a loop counter the compiler generates itself.
    """

    def outOfRange(arr: mc.RanVarArray) -> cython.double:
        return ~arr[99]

    with pytest.raises(IndexError):
        mc.cfunc(outOfRange, samples=10)(makeArray())


def test_rejects_drawing_from_a_ranvar_array_slice():
    """Tests that ~arr[1:3] is refused: a slice is still several digests."""

    def slicedInvert(arr: mc.RanVarArray) -> cython.double:
        return ~arr[1:3]

    with pytest.raises(mc.RanVarCompileError):
        mc.cfunc(slicedInvert).compile()
