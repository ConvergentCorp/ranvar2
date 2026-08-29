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


class Bound():
    """A wrapper bound to the instance whose method it decorates.

    A plain function is a descriptor, and that is what supplies self to a
    method. The decorators are objects rather than functions, so binding is
    theirs to do, and this is what their __get__ hands back.

    Attributes:
        wrapper (object): The decorator wrapping the method.
        instance (object): The instance the method was reached through.
    """

    def __init__(self, wrapper, instance):
        """Bind a wrapper to an instance.

        Args:
            wrapper (object): The decorator wrapping the method.
            instance (object): The instance to pass as self.
        """
        self.wrapper  = wrapper
        self.instance = instance

    def __call__(self, *args, **kwargs):
        """Run the simulation, passing the instance as the model's first argument.

        Args:
            *args: The remaining arguments for the model.
            **kwargs: Keyword arguments for the model.

        Returns:
            RanVar: What the wrapper returns.
        """
        return self.wrapper(self.instance, *args, **kwargs)

    def withSamples(self, samples):
        """Get an equivalent binding that runs a different number of samples.

        Args:
            samples (int): Number of runs the result performs.

        Returns:
            Bound: The new wrapper, bound to the same instance.
        """
        return Bound(self.wrapper.withSamples(samples), self.instance)

    def __getattr__(self, name):
        """Fall back to the wrapper for anything binding does not change.

        Args:
            name (str): The attribute wanted.

        Returns:
            object: The wrapper's attribute.
        """
        return getattr(self.wrapper, name)

    def __repr__(self):
        """Describe this binding, for debugging.

        Returns:
            str: The wrapped decorator and the instance it is bound to.
        """
        return f'<Bound {self.wrapper!r} to {self.instance!r}>'


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

    A function may report several quantities at once by returning a tuple or a
    list of them. Each position is collected into its own RanVar and handed back
    in the same shape, so the results unpack the way the return statement reads.
    A namedtuple comes back as itself, which is the tidiest way to keep several
    outputs named.

    Attributes:
        samples (int): Number of runs a plain call performs.
        maxBins (int): Number of centroids each returned RanVar maintains.

    Example:
        >>> @func
        ... def profit():
        ...     return ~revenue - ~cost
        >>> profit()                      # DEFAULT_SAMPLES runs
        >>> profit.withSamples(1_000_000)()   # a longer run

        >>> @func
        ... def model():
        ...     revenue = ~price * ~volume
        ...     return revenue, revenue - ~cost
        >>> revenue, profit = model()     # a RanVar each
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

    def __get__(self, instance, owner=None):
        """Bind the wrapper when it decorates a method.

        Args:
            instance (object): The instance the attribute was reached through,
                             or None when reached through the class.
            owner (type, optional): The class the attribute was found on.

        Returns:
            object: This wrapper when reached through the class, otherwise one
                  bound to the instance.
        """
        if instance is None:
            return self

        return Bound(self, instance)

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

        The first run decides the shape: a function returning a bare number is
        collected into one RanVar, and a function returning a tuple or a list is
        collected into one RanVar per position, handed back in the same shape.
        The two are run by separate loops so a single output model does not pay
        for the bookkeeping the multiple output one needs.

        Args:
            samples (int): Number of runs to perform.
            args (tuple): Positional arguments for the wrapped function.
            kwargs (dict): Keyword arguments for the wrapped function.

        Returns:
            RanVar: The distribution of the results, for a function returning a
                  bare number.
            tuple | list: One RanVar per returned value, in the shape the
                        function returned, otherwise.

        Raises:
            TypeError: If the wrapped function returns something non numeric.
            ValueError: If it returns nothing, or a different number of values
                      on different runs.
        """
        fn    = self.__wrapped__
        first = fn(*args, **kwargs)

        if isinstance(first, (tuple, list)):
            return self._runMany(fn, first, samples, args, kwargs)

        return self._runOne(fn, first, samples, args, kwargs)

    def _runOne(self, fn, first, samples, args, kwargs):
        """Collect a function that returns a single number.

        Args:
            fn (callable): The wrapped function.
            first (object): The result of the run already performed.
            samples (int): Total number of runs, the one already done included.
            args (tuple): Positional arguments for the wrapped function.
            kwargs (dict): Keyword arguments for the wrapped function.

        Returns:
            RanVar: The distribution of the results.

        Raises:
            TypeError: If a run returns something non numeric.
            ValueError: If a run starts returning several values.
        """
        out    = RanVar(maxBins=self.maxBins)
        result = first

        # This is the hot loop of the whole library, so the conversion is inline
        # rather than behind _number(). A try block costs nothing while it
        # succeeds, and once something has gone wrong the handler can afford to
        # work out what.
        try:
            for _ in range(samples - 1):
                out.add(float(result))
                result = fn(*args, **kwargs)

            out.add(float(result))
        except (TypeError, ValueError):
            if isinstance(result, (tuple, list)):
                raise ValueError(
                    f'{self.__name__}() returned {self._describe(result)} on one '
                    f'run and a bare number on another; it has to return the '
                    f'same number of values every run'
                ) from None

            if not self._isNumber(result):
                raise self._numberError(result) from None

            # The last result converts cleanly, so the error came out of the
            # model itself rather than out of collecting what it returned.
            raise

        return out

    def _runMany(self, fn, first, samples, args, kwargs):
        """Collect a function that returns several numbers at once.

        Args:
            fn (callable): The wrapped function.
            first (tuple | list): The result of the run already performed.
            samples (int): Total number of runs, the one already done included.
            args (tuple): Positional arguments for the wrapped function.
            kwargs (dict): Keyword arguments for the wrapped function.

        Returns:
            tuple | list: One RanVar per returned value, in the shape the
                        function returned.
        """
        width = len(first)

        if width == 0:
            raise ValueError(
                f'{self.__name__}() returned no values for the simulation to '
                f'collect'
            )

        outs   = [RanVar(maxBins=self.maxBins) for _ in range(width)]
        result = first

        # The run that decided the shape is the first sample, so this fetches
        # one fewer and collects the last result after the loop.
        for _ in range(samples - 1):
            self._collect(outs, result, width)
            result = fn(*args, **kwargs)

        self._collect(outs, result, width)

        return self._shapeLike(first, outs)

    def _collect(self, outs, result, width):
        """Add one run's values to the digests collecting them.

        Args:
            outs (list): One RanVar per returned value.
            result (object): What this run returned.
            width (int): How many values the first run returned.

        Raises:
            TypeError: If one of the values is not numeric.
            ValueError: If this run returned a different number of values.
        """
        # Every run has to line up with the first, otherwise the collected
        # distributions would be a mix of different quantities.
        if not isinstance(result, (tuple, list)) or len(result) != width:
            raise ValueError(
                f'{self.__name__}() returned {self._describe(result)} on one run '
                f'and {width} values on another; it has to return the same '
                f'number of values every run'
            )

        for out, value in zip(outs, result):
            out.add(self._number(value))

    def _number(self, value):
        """Coerce one returned value into a number the digest can collect.

        Args:
            value (object): A value returned by the wrapped function.

        Returns:
            float: The value as a float.

        Raises:
            TypeError: If the value is not numeric.
        """
        try:
            return float(value)
        except (TypeError, ValueError):
            raise self._numberError(value) from None

    def _isNumber(self, value):
        """Report whether a value is something the digest could collect.

        Args:
            value (object): The value to test.

        Returns:
            bool: True if the value converts to a float.
        """
        try:
            float(value)
        except (TypeError, ValueError):
            return False

        return True

    def _numberError(self, value):
        """Build the error for a value the simulation cannot collect.

        Args:
            value (object): The offending value.

        Returns:
            TypeError: The error to raise.
        """
        return TypeError(
            f'{self.__name__}() must return a number, or a tuple or list of '
            f'numbers, for the simulation to collect, got '
            f'{type(value).__name__!r}'
        )

    def _describe(self, result):
        """Describe a badly shaped result for an error message.

        Args:
            result (object): The result to describe.

        Returns:
            str: A phrase naming what came back.
        """
        if isinstance(result, (tuple, list)):
            return f'{len(result)} values'

        return f'a bare {type(result).__name__}'

    def _shapeLike(self, template, outs):
        """Return the collected digests in the shape the function returned.

        A namedtuple is rebuilt as itself, so a model with named outputs hands
        back distributions under those same names.

        Args:
            template (tuple | list): The first result, used for its shape.
            outs (list): One RanVar per returned value.

        Returns:
            tuple | list: The digests in the shape of the template.
        """
        if isinstance(template, list):
            return outs

        # A namedtuple is a tuple whose type takes one argument per field.
        if hasattr(type(template), '_fields'):
            return type(template)(*outs)

        return tuple(outs)

    def __repr__(self):
        """Describe this wrapper, for debugging.

        Returns:
            str: The wrapped function's name and the wrapper's settings.
        """
        return (
            f'<MonteCarlo {self.__name__} '
            f'samples={self.samples} maxBins={self.maxBins}>'
        )


def func(fn=None, *, samples=DEFAULT_SAMPLES, maxBins=DEFAULT_MAXBINS):
    """Wrap a function in a monte carlo simulation.

    The decorated function becomes a random variable: calling it runs the
    function repeatedly and returns a RanVar of the results rather than a single
    value. A function returning a tuple or a list of numbers gets one RanVar per
    position, in the same shape. Usable bare or with options.

    Args:
        fn (callable, optional): The function to wrap, supplied when the
                               decorator is used bare.
        samples (int, optional): Runs per call. Defaults to DEFAULT_SAMPLES.
        maxBins (int, optional): Centroids in each returned RanVar.
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

        >>> @func
        ... def rolls():
        ...     return ~d6, ~d6           # a RanVar for each die
    """
    # Validate here as well so @func(samples=0) fails at decoration time rather
    # than at the first call.
    _checkSamples(samples)

    if fn is None:
        def decorator(fn):
            return MonteCarlo(fn, samples=samples, maxBins=maxBins)

        return decorator

    return MonteCarlo(fn, samples=samples, maxBins=maxBins)
