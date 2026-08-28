import functools

from ranvar.ranvar import RanVar

# Enough draws for a stable distribution without making an interactive call feel
# slow. Override per model with @func(samples=...) or per call with withSamples().
DEFAULT_SAMPLES = 10_000

# Mirrors the RanVar default so a decorated model behaves like a hand-built digest.
DEFAULT_MAXBINS = 32


def _checkSamples(samples):
    """Validate a requested sample count.

    Args:
        samples (int): The number of simulation runs requested.

    Returns:
        int: The validated sample count.

    Raises:
        TypeError: If samples is not an integer.
        ValueError: If samples is less than one.
    """
    # bool is a subclass of int, and samples=True is far more likely to be a
    # mistake than a request for a single run.
    if isinstance(samples, bool) or not isinstance(samples, int):
        raise TypeError(
            f'samples must be an integer, got {type(samples).__name__!r}'
        )

    if samples < 1:
        raise ValueError(f'samples must be at least 1, got {samples}')

    return samples


class MonteCarlo():
    """A function wrapped in a monte carlo executor.

    Calling the wrapper runs the wrapped function repeatedly, collects each
    result into a RanVar and returns it, so an ordinary function that draws from
    random variables becomes a random variable itself. Nothing runs at decoration
    time; the simulation only happens when the wrapper is called.

    Arguments passed to the wrapper are forwarded unchanged to every run, so the
    wrapper keeps the signature of the function it wraps. The sample count is
    deliberately not a keyword argument for that reason: it would collide with a
    model that has a parameter of the same name. Use withSamples() instead.

    Attributes:
        samples (int): Number of runs a plain call performs.
        maxBins (int): Number of centroids the returned RanVar maintains.

    Example:
        >>> @func
        ... def profit():
        ...     return ~revenue - ~cost
        >>> profit()                      # DEFAULT_SAMPLES runs
        >>> profit.withSamples(1_000_000)()   # a longer run
    """

    def __init__(self, fn, samples=DEFAULT_SAMPLES, maxBins=DEFAULT_MAXBINS):
        """Wrap a function in a monte carlo executor.

        Args:
            fn (callable): The function to simulate.
            samples (int, optional): Runs per call. Defaults to DEFAULT_SAMPLES.
            maxBins (int, optional): Centroids in the returned RanVar.
                                   Defaults to DEFAULT_MAXBINS.

        Raises:
            TypeError: If fn is not callable.
        """
        if not callable(fn):
            raise TypeError(
                f'func expected a callable, got {type(fn).__name__!r}. '
                f'To set options use the keyword form, @func(samples=...)'
            )

        # Before assigning our own attributes: update_wrapper copies the wrapped
        # function's __dict__ over ours and would otherwise overwrite them.
        functools.update_wrapper(self, fn)

        self.samples = _checkSamples(samples)
        self.maxBins = maxBins

    def __call__(self, *args, **kwargs):
        """Run the simulation with the default sample count.

        Args:
            *args: Forwarded unchanged to every run of the wrapped function.
            **kwargs: Forwarded unchanged to every run of the wrapped function.

        Returns:
            RanVar: The distribution of the results.
        """
        return self._run(self.samples, args, kwargs)

    def withSamples(self, samples):
        """Get an equivalent wrapper that runs a different number of samples.

        The wrapped function and maxBins are shared, so this is cheap enough to
        call inline and the result can equally be kept and reused.

        Args:
            samples (int): Number of runs the returned wrapper performs.

        Returns:
            MonteCarlo: A wrapper around the same function with the new count.

        Example:
            >>> profit.withSamples(1_000_000)()
        """
        return MonteCarlo(
            self.__wrapped__, samples=samples, maxBins=self.maxBins
        )

    def _run(self, samples, args, kwargs):
        """Run the wrapped function and collect the results.

        Args:
            samples (int): Number of runs to perform.
            args (tuple): Positional arguments for the wrapped function.
            kwargs (dict): Keyword arguments for the wrapped function.

        Returns:
            RanVar: The distribution of the results.

        Raises:
            TypeError: If the wrapped function returns something non numeric.
        """
        fn  = self.__wrapped__
        out = RanVar(maxBins=self.maxBins)

        for _ in range(samples):
            result = fn(*args, **kwargs)

            try:
                value = float(result)
            except (TypeError, ValueError):
                raise TypeError(
                    f'{self.__name__}() must return a number for the '
                    f'simulation to collect, got {type(result).__name__!r}'
                ) from None

            out.add(value)

        return out

    def __repr__(self):
        return (
            f'<MonteCarlo {self.__name__} '
            f'samples={self.samples} maxBins={self.maxBins}>'
        )


def func(fn=None, *, samples=DEFAULT_SAMPLES, maxBins=DEFAULT_MAXBINS):
    """Wrap a function in a monte carlo simulation.

    The decorated function becomes a random variable: calling it runs the
    function repeatedly and returns a RanVar of the results rather than a single
    value. Usable bare or with options.

    Args:
        fn (callable, optional): The function to wrap, supplied when the
                               decorator is used bare.
        samples (int, optional): Runs per call. Defaults to DEFAULT_SAMPLES.
        maxBins (int, optional): Centroids in the returned RanVar.
                               Defaults to DEFAULT_MAXBINS.

    Returns:
        MonteCarlo: The wrapped function, or a decorator returning one when
                  options are given.

    Example:
        >>> @func
        ... def roll():
        ...     return ~d6 + ~d6

        >>> @func(samples=100_000, maxBins=64)
        ... def roll():
        ...     return ~d6 + ~d6
    """
    # Validate here as well so @func(samples=0) fails at decoration time rather
    # than at the first call.
    _checkSamples(samples)

    if fn is None:
        def decorator(fn):
            return MonteCarlo(fn, samples=samples, maxBins=maxBins)

        return decorator

    return MonteCarlo(fn, samples=samples, maxBins=maxBins)
