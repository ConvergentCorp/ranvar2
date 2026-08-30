# ranvar2

Fast probabilistic modelling for Python: build random variables as
[t-digests](https://arxiv.org/abs/1902.04023), combine them with ordinary
Python functions, and run the resulting Monte Carlo simulations either
interpreted or compiled to C via [Cython](https://cython.org/).

```python
import cython
from ranvar2 import RanVar, cfunc

revenue = RanVar(maxBins=64)
revenue.fit([95, 100, 105, 98, 102, 110, 90, 100, 101, 99])

cost = RanVar(maxBins=64)
cost.fit([40, 42, 38, 41, 39, 43, 37])

@cfunc(samples=200_000)
def profit(revenue: RanVar, cost: RanVar) -> cython.double:
    margin: cython.double = ~revenue - ~cost
    return margin

result = profit(revenue, cost)
print(result.mean(), result.quantile(0.05), result.quantile(0.95))
```

## Why

A `RanVar` is a random variable represented as a t-digest: a compact,
bounded-memory summary of a distribution that supports fast quantile lookup
and sampling. `~x` draws a sample from `x`. Ordinary Python functions that
draw from one or more `RanVar`s become Monte Carlo models via a decorator:

- `@func` runs the model in the interpreter, samples-per-call at a time.
- `@cfunc` compiles the model's own source with Cython (in [pure Python
  mode](https://cython.readthedocs.io/en/latest/src/tutorial/pure.html)) and
  runs the sampling loop natively in C, with no per-sample trip back into the
  interpreter. The first call compiles and caches the result; later calls
  (in this process or a later one) reuse it.

`Normal`, `NegBinom` and `Constant` are `RanVar` subclasses that draw
directly from a fixed distribution's own formula instead of a fitted digest,
so they can be used as model inputs — a prior, an assumption, a value held
fixed, anything you don't have data for yet — while still running at C speed
inside a `@cfunc` model.

## Requirements

- Python >= 3.12
- A C compiler toolchain (e.g. `build-essential` on Debian/Ubuntu, Xcode
  Command Line Tools on macOS). This is needed both to build the package
  itself and, at runtime, for `@cfunc`, which compiles each model the first
  time it's called and caches the result under `~/.cache/ranvar2`.

## Installation

This is a private repo, not published to PyPI — install directly from
GitHub, as an org member with access. With [uv](https://docs.astral.sh/uv/):

```sh
uv add git+https://github.com/ConvergentCorp/ranvar2.git
```

Or with pip:

```sh
pip install git+https://github.com/ConvergentCorp/ranvar2.git
```

The HTTPS URL needs git to be able to authenticate to GitHub with your
account; if you use the [`gh` CLI](https://cli.github.com/), `gh auth
login` followed by `gh auth setup-git` sets that up. If you have an SSH key
registered with GitHub instead, `git+ssh://git@github.com/ConvergentCorp/ranvar2.git`
works the same way.

To pin a specific commit or tag, append `@<ref>` to the URL.

## Quickstart

### Building a `RanVar` from data

```python
from ranvar2 import RanVar

x = RanVar(maxBins=64)
x.fit([95, 100, 105, 98, 102, 110, 90, 100, 101, 99])

x.mean()          # the exact weighted mean of what was added
x.quantile(0.5)   # the median, interpolated from the digest
x.sample()        # ~x: one draw from the fitted distribution
x.cdf(100)        # P(X <= 100)
```

### `@func`: an interpreted Monte Carlo model

```python
from ranvar2 import RanVar, func

d6 = RanVar(maxBins=8)
d6.fit([1, 2, 3, 4, 5, 6] * 100)

@func(samples=50_000)
def rollTwoDice():
    return ~d6 + ~d6

total = rollTwoDice()   # a RanVar over 50,000 simulated rolls
total.mean()             # ~7.0
```

### `@cfunc`: the same idea, compiled

The model's parameters and return type are annotated so Cython can type the
generated driver; `~x` on an annotated `RanVar` parameter compiles to a
direct C call (`x.sample()`) rather than a Python-level draw.

```python
import cython
from ranvar2 import RanVar, cfunc

@cfunc(samples=200_000)
def profit(revenue: RanVar, cost: RanVar) -> cython.double:
    margin: cython.double = ~revenue - ~cost
    return margin

result = profit(revenue, cost)   # revenue, cost from above
```

A model returning a tuple (or not annotated with a scalar C return type)
gets back a `RanVar` per position instead of one:

```python
@cfunc(samples=100_000)
def bothDice() -> None:
    return ~d6, ~d6

first, second = bothDice()
```

### `Normal`, `NegBinom` and `Constant`: parametric inputs

Use these where you have an assumption or a prior rather than data to fit a
digest to. All subclass `RanVar`, so they work anywhere a `RanVar` does,
including as a `@cfunc` model parameter — sampling still costs one C call.

```python
from ranvar2 import Normal, NegBinom, Constant, RanVar, cfunc

demand  = Normal(mean=500.0, std=50.0)
defects = NegBinom(mean=3.0, dispersion=0.8)  # variance = mean + dispersion * mean**2
retries = NegBinom(r=5.0, p=0.4)              # same thing, textbook parameterization
price   = Constant(value=19.99)               # degenerate: every draw is 19.99

@cfunc(samples=200_000)
def shortfall(capacity: cython.double, demand: RanVar) -> cython.double:
    d: cython.double = ~demand
    return d - capacity if d > capacity else 0.0

shortfall(450.0, demand).mean()
```

`NegBinom` takes either parameterization: `mean`/`dispersion` (NB2), or the
textbook `r`/`p` — failures before the `r`-th success at success probability
`p` — which is converted at construction and stored as
`mean = r(1 - p)/p`, `dispersion = 1/r`. The two can't be mixed in one call,
and everything past `__init__` (`mean()`, `dispersion()`, `fit()`, pickling)
works in `mean`/`dispersion` only. `r` may be any positive real, not just an
integer, since sampling goes through a Gamma-Poisson mixture.

`Constant` is the degenerate distribution: all of its weight sits on one
point, so it puts a plain number in a slot that has to be a `RanVar` — a
scenario held fixed, an assumption not yet given a spread — without the model
having to special case it.

`fit(x)` re-estimates a `Normal`/`NegBinom`'s parameters from data by the
method of moments (needs at least 2 points); `add()` is refused, since a
single point can't estimate a spread. `Constant` is the exception on both
counts: `fit(x)` takes its value from the sample mean of as little as one
point, and `add(point)` replaces the value outright, since one point does
fully determine a constant. Methods with no closed form in terms of standard
C `math.h` functions (`Normal.quantile()`,
`NegBinom.cdf()`/`ccdf()`/`dcdf()`/`dccdf()`/`quantile()`) raise
`NotImplementedError` rather than silently reading an unrelated digest, as do
`Constant.dcdf()`/`dccdf()`, whose CDF is a step with no density.

### `RanVarArray`: a collection of digests

Some models depend on a whole collection of `RanVar`s rather than a single
one — one per region, one per product line. `RanVarArray` wraps a list of
them behind the ordinary list interface (indexing, slicing, `append`,
`len`, `+`, iteration, ...), while still letting `@cfunc` compile a loop
over it:

```python
from ranvar2 import Normal, RanVarArray, cfunc

regions = RanVarArray.fromList([
    Normal(mean=100.0, std=10.0),
    Normal(mean=200.0, std=15.0),
    Normal(mean=50.0, std=5.0),
])

@cfunc(samples=200_000)
def totalDemand(regions: RanVarArray) -> cython.double:
    total: cython.double = 0.0
    for region in regions:
        total += ~region
    return total

totalDemand(regions).mean()   # ~350.0
```

`for region in regions:` over a parameter annotated `RanVarArray` is
rewritten by the compiler into an indexed loop that fetches each element
through a C call, so drawing from every region — with `~region` or
`region.sample()` — costs one C call per element, the same as a single
`RanVar` parameter does.

A fixed number of elements reached by indexing compiles the same way:

```python
@cfunc(samples=200_000)
def firstTwo(regions: RanVarArray) -> cython.double:
    total: cython.double = 0.0
    for i in range(2):
        total += ~regions[i]
    return total
```

`regions[i]`, `i` an index expression rather than a slice, is rewritten into
a direct (bounds checked, negative-index aware, like a plain list's) fetch,
so `~regions[i]` and `regions[i].sample()` are both a C call per element too.
Only these two patterns — a bare `RanVarArray` parameter iterated directly,
or indexed with a single expression — are rewritten; unpacking, or slicing
the array first, still works, just as ordinary Python.

Every element has to be a `RanVar` (`Normal`, `NegBinom` and `Constant`
included, since they all subclass it) — checked on `fromList()`, the plain constructor, and
every mutation. `~regions` and `~regions[1:3]` (drawing from the whole array
or a slice of it, rather than a single element) and reading a `RanVarArray`
from the surrounding module (rather than taking it as a parameter) are all
refused for the same reason a bare `RanVar` would be.

### `asRanVar`: coercing model inputs

A model's parameters are digests, but what a caller has on hand is often a
plain number or a plain list. `asRanVar` wraps those, and passes anything
already usable through unchanged:

```python
from ranvar2 import asRanVar

asRanVar(7)                      # Constant(value=7.0)
asRanVar(np.int64(7))            # Constant too: any real number, numpy scalars included
asRanVar([1.0, Normal(5, 1)])    # RanVarArray, each element coerced in turn
asRanVar(Normal(5, 1))           # unchanged
asRanVar(someRanVarArray)        # unchanged
```

Anything else raises `TypeError` rather than being passed on to fail deeper
inside a compiled model. The point is that a model stochastic in a parameter
today keeps working when that parameter is a constant tomorrow:

```python
self.D = self._model(asRanVar(demand), asRanVar(leadTime), asRanVar(cycle))
```

### Reproducibility

Sampling draws from a single process-global C generator (`rand()`/`srand()`
under the hood), shared by every `RanVar`, `Normal`, and `NegBinom`
(`Constant` draws nothing, so it is unaffected):

```python
from ranvar2 import seed

seed(1234)   # make every following draw in this process reproducible
```

## Development

```sh
git clone https://github.com/ConvergentCorp/ranvar2.git
cd ranvar2
uv sync
uv run pytest
```

`setup.py` compiles `ranvar2/ranvar.py` with Cython; `uv sync` (or
`pip install -e .`) rebuilds it as needed.

## License

MIT — see [LICENSE](LICENSE).
