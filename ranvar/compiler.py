import ast
import functools
import hashlib
import importlib.machinery
import importlib.util
import inspect
import os
import subprocess
import sys
import textwrap

from ranvar.decorators import DEFAULT_MAXBINS, DEFAULT_SAMPLES, _checkSamples
from ranvar.ranvar import RanVar


class RanVarCompileError(Exception):
    """Raised when a function cannot be compiled into a C simulation.

    cfunc refuses anything it cannot turn into C rather than quietly running it
    through the interpreter, so that a model either has the performance the
    decorator promises or says why it does not.
    """


# The generated module names its own machinery with this prefix. A model using
# a name of its own that starts with it would be shadowed, so they are refused.
RESERVED = '_rv_'

# The arithmetic the generated C understands. Floor division and modulo are left
# out because their meaning for negative operands differs between Python and the
# C the generated module compiles to.
BINOPS = {
    ast.Add: '+',
    ast.Sub: '-',
    ast.Mult: '*',
    ast.Div: '/',
    ast.Pow: '**',
}

COMPARES = {
    ast.Lt: '<',
    ast.LtE: '<=',
    ast.Gt: '>',
    ast.GtE: '>=',
    ast.Eq: '==',
    ast.NotEq: '!=',
}


class Translator():
    """Turns a model function into the source of a Cython module.

    The result is a module holding the model body as a cdef function over C
    doubles and a driver that loops over it, so that once it is called nothing
    in the simulation goes back through the interpreter.

    Names the model reads but does not define are resolved at compile time to
    decide what they are, and passed into the loop as arguments at call time so
    that rebinding one is still seen.

    Attributes:
        params (list): The model's own parameter names.
        free (list): Names read from the enclosing scope, in argument order.
        draws (set): Which of those names are RanVars and so can be drawn from.
        width (int): How many values the model returns, 0 meaning a single one
                   and None meaning no return has been seen yet.
    """

    def __init__(self, fn):
        """Prepare to translate a function.

        Args:
            fn (callable): The model to translate.
        """
        self.fn     = fn
        self.name   = getattr(fn, '__name__', '<model>')
        self.params = []
        self.locals = []
        self.free   = []
        self.draws  = set()
        self.width  = None

    # Analysis. ----------------------------------------------------------------

    def fail(self, node, message):
        """Report that the model cannot be compiled.

        Args:
            node (ast.AST): The offending node, for its line number.
            message (str): What is wrong.

        Raises:
            RanVarCompileError: Always.
        """
        line = getattr(node, 'lineno', None)
        where = f' (line {line})' if line else ''

        raise RanVarCompileError(f'{self.name}(){where}: {message}')

    def tree(self):
        """Parse the model's source into the function definition node.

        Returns:
            ast.FunctionDef: The model's definition.

        Raises:
            RanVarCompileError: If the source cannot be found or is not a plain
                              function definition.
        """
        try:
            source = inspect.getsource(self.fn)
        except (OSError, TypeError):
            raise RanVarCompileError(
                f'{self.name}(): cannot read the source of this function, so it '
                f'cannot be compiled. cfunc needs a function defined in a file'
            ) from None

        parsed = ast.parse(textwrap.dedent(source))
        node   = parsed.body[0]

        if not isinstance(node, ast.FunctionDef):
            self.fail(node, 'only a plain function can be compiled')

        return node

    def collect(self, node):
        """Work out the model's parameters, locals and free names.

        Args:
            node (ast.FunctionDef): The model's definition.

        Raises:
            RanVarCompileError: If the signature or a name is unusable.
        """
        args = node.args

        if args.vararg is not None or args.kwarg is not None:
            self.fail(node, '*args and **kwargs cannot be compiled')

        for arg in args.posonlyargs + args.args + args.kwonlyargs:
            self.params.append(arg.arg)

        # Only the body is scanned. The decorator list and the default values
        # are part of the definition rather than of what runs, and naming a
        # decorator would otherwise look like a free variable of the model.
        for child in self.walk(node):
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
                if child.id not in self.locals and child.id not in self.params:
                    self.locals.append(child.id)

        for name in self.params + self.locals:
            if name.startswith(RESERVED):
                self.fail(node, f'{name!r} is reserved for the generated module')

        known = set(self.params) | set(self.locals)

        for child in self.walk(node):
            if not isinstance(child, ast.Name) or not isinstance(child.ctx, ast.Load):
                continue

            if child.id in known or child.id in self.free:
                continue

            self.free.append(child.id)

    def structure(self, node):
        """Reject unusable statements before any name is resolved.

        Running first means a for loop is reported as a for loop, rather than as
        whatever name it happens to mention on the way past.

        Args:
            node (ast.FunctionDef): The model's definition.

        Raises:
            RanVarCompileError: If a statement or a call is outside the subset.
        """
        for statement in node.body:
            self.statement(statement)

        for child in self.walk(node):
            if isinstance(child, ast.Call):
                self.fail(child, 'calling a function cannot be compiled')

    def statement(self, statement):
        """Check one statement, and anything nested inside it, is compilable.

        Args:
            statement (ast.AST): The statement to check.

        Raises:
            RanVarCompileError: If the statement is outside the subset.
        """
        if isinstance(statement, ast.If):
            for child in statement.body + statement.orelse:
                self.statement(child)

            return

        allowed = (ast.Expr, ast.Pass, ast.Assign, ast.AugAssign, ast.Return)

        if not isinstance(statement, allowed):
            self.fail(statement, f'{type(statement).__name__} cannot be compiled')

    def walk(self, node):
        """Visit every node in the model's body.

        Args:
            node (ast.FunctionDef): The model's definition.

        Yields:
            ast.AST: Each node under one of the body's statements.
        """
        for statement in node.body:
            yield from ast.walk(statement)

    def resolve(self, name):
        """Look a free name up in the scope the model was defined in.

        Args:
            name (str): The name to resolve.

        Returns:
            object: What the name is currently bound to.

        Raises:
            RanVarCompileError: If the name is not bound anywhere reachable.
        """
        code = self.fn.__code__

        if name in code.co_freevars and self.fn.__closure__ is not None:
            cell = self.fn.__closure__[code.co_freevars.index(name)]

            try:
                return cell.cell_contents
            except ValueError:
                raise RanVarCompileError(
                    f'{self.name}(): {name!r} is not set yet'
                ) from None

        if name in self.fn.__globals__:
            return self.fn.__globals__[name]

        raise RanVarCompileError(
            f'{self.name}(): {name!r} is not defined, so there is nothing to '
            f'compile it as'
        )

    def classify(self):
        """Decide what each free name is, so it can be given a C type.

        Raises:
            RanVarCompileError: If a name is neither a RanVar nor a number.
        """
        for name in self.free:
            value = self.resolve(name)

            if isinstance(value, RanVar):
                self.draws.add(name)
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                pass
            else:
                raise RanVarCompileError(
                    f'{self.name}(): {name!r} is a {type(value).__name__!r}; a '
                    f'compiled model can only use RanVars and numbers from the '
                    f'surrounding scope'
                )

    # Expressions. -------------------------------------------------------------

    def expr(self, node):
        """Translate an expression into C.

        Args:
            node (ast.AST): The expression node.

        Returns:
            str: The equivalent Cython expression.

        Raises:
            RanVarCompileError: If the expression is outside the subset.
        """
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                self.fail(node, f'{node.value!r} is not a number')

            return repr(float(node.value))

        if isinstance(node, ast.Name):
            return node.id

        if isinstance(node, ast.BinOp):
            op = BINOPS.get(type(node.op))

            if op is None:
                self.fail(node, f'the {type(node.op).__name__} operator cannot be compiled')

            return f'({self.expr(node.left)} {op} {self.expr(node.right)})'

        if isinstance(node, ast.UnaryOp):
            return self.unary(node)

        if isinstance(node, ast.Compare):
            if len(node.ops) != 1:
                self.fail(node, 'a chained comparison cannot be compiled')

            op = COMPARES.get(type(node.ops[0]))

            if op is None:
                self.fail(node, f'the {type(node.ops[0]).__name__} comparison cannot be compiled')

            return f'({self.expr(node.left)} {op} {self.expr(node.comparators[0])})'

        if isinstance(node, ast.BoolOp):
            joiner = ' and ' if isinstance(node.op, ast.And) else ' or '

            return '(' + joiner.join(self.expr(v) for v in node.values) + ')'

        if isinstance(node, ast.IfExp):
            return (
                f'({self.expr(node.body)} if {self.expr(node.test)} '
                f'else {self.expr(node.orelse)})'
            )

        if isinstance(node, ast.Call):
            self.fail(node, 'calling a function cannot be compiled')

        self.fail(node, f'{type(node).__name__} cannot be compiled')

    def unary(self, node):
        """Translate a unary expression, including a draw.

        Args:
            node (ast.UnaryOp): The expression node.

        Returns:
            str: The equivalent Cython expression.

        Raises:
            RanVarCompileError: If the operand of ~ is not a RanVar.
        """
        if isinstance(node.op, ast.USub):
            return f'(-{self.expr(node.operand)})'

        if isinstance(node.op, ast.UAdd):
            return f'(+{self.expr(node.operand)})'

        if isinstance(node.op, ast.Not):
            return f'(not {self.expr(node.operand)})'

        if isinstance(node.op, ast.Invert):
            operand = node.operand

            if not isinstance(operand, ast.Name) or operand.id not in self.draws:
                self.fail(node, '~ can only be applied to a RanVar from the surrounding scope')

            return f'{RESERVED}draw({operand.id})'

        self.fail(node, f'the {type(node.op).__name__} operator cannot be compiled')

    # Statements. --------------------------------------------------------------

    def block(self, body, level):
        """Translate a list of statements.

        Args:
            body (list): The statements.
            level (int): Indentation depth inside the generated function.

        Returns:
            list: The generated lines.
        """
        out = []

        for statement in body:
            out.extend(self.stmt(statement, level))

        if not out:
            out.append('    '*level + 'pass')

        return out

    def stmt(self, node, level):
        """Translate one statement.

        Args:
            node (ast.AST): The statement node.
            level (int): Indentation depth inside the generated function.

        Returns:
            list: The generated lines.

        Raises:
            RanVarCompileError: If the statement is outside the subset.
        """
        pad = '    '*level

        if isinstance(node, ast.Expr):
            # A docstring is the one expression statement worth keeping around,
            # and it has no effect to translate.
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                return []

            self.fail(node, 'an expression on its own has no effect and cannot be compiled')

        if isinstance(node, ast.Pass):
            return [pad + 'pass']

        if isinstance(node, ast.Assign):
            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                self.fail(node, 'only assignment to a single name can be compiled')

            return [f'{pad}{node.targets[0].id} = {self.expr(node.value)}']

        if isinstance(node, ast.AugAssign):
            if not isinstance(node.target, ast.Name):
                self.fail(node, 'only assignment to a single name can be compiled')

            op = BINOPS.get(type(node.op))

            if op is None:
                self.fail(node, f'the {type(node.op).__name__} operator cannot be compiled')

            return [f'{pad}{node.target.id} {op}= {self.expr(node.value)}']

        if isinstance(node, ast.If):
            lines = [f'{pad}if {self.expr(node.test)}:']
            lines.extend(self.block(node.body, level + 1))

            if node.orelse:
                lines.append(f'{pad}else:')
                lines.extend(self.block(node.orelse, level + 1))

            return lines

        if isinstance(node, ast.Return):
            return self.returns(node, pad)

        self.fail(node, f'{type(node).__name__} cannot be compiled')

    def returns(self, node, pad):
        """Translate a return statement.

        The first return decides how many values the model produces; the rest
        have to agree, since each value is collected into its own digest.

        Args:
            node (ast.Return): The statement node.
            pad (str): Leading indentation.

        Returns:
            list: The generated lines.

        Raises:
            RanVarCompileError: If the model returns nothing, or a different
                              number of values in different places.
        """
        if node.value is None:
            self.fail(node, 'a compiled model has to return a value')

        if isinstance(node.value, ast.Tuple):
            width = len(node.value.elts)

            if width == 0:
                self.fail(node, 'a compiled model has to return a value')
        else:
            width = 0

        if self.width is None:
            self.width = width
        elif self.width != width:
            self.fail(
                node,
                f'this returns {width or 1} value(s) where an earlier return '
                f'gives {self.width or 1}; they have to agree'
            )

        if width == 0:
            return [f'{pad}return {self.expr(node.value)}']

        lines = [
            f'{pad}{RESERVED}res[{i}] = {self.expr(element)}'
            for i, element in enumerate(node.value.elts)
        ]
        lines.append(f'{pad}return')

        return lines

    # Code generation. ---------------------------------------------------------

    def signature(self):
        """Build the C parameter list the model body takes.

        Returns:
            list: One 'type name' string per argument.
        """
        parts = [f'double {name}' for name in self.params]

        for name in self.free:
            kind = 'RanVar' if name in self.draws else 'double'
            parts.append(f'{kind} {name}')

        return parts

    def arguments(self):
        """Name the model body's arguments, in the order signature() gives them.

        Returns:
            list: The argument names.
        """
        return self.params + self.free

    def generate(self):
        """Translate the model into the source of a Cython module.

        Returns:
            str: The module source.

        Raises:
            RanVarCompileError: If the model is outside the compilable subset.
        """
        node = self.tree()

        self.structure(node)
        self.collect(node)
        self.classify()

        body = self.block(node.body, 1)

        if self.width is None:
            self.fail(node, 'a compiled model has to return a value')

        if not alwaysReturns(node.body):
            self.fail(node, 'every path through a compiled model has to return a value')

        signature = self.signature()
        arguments = self.arguments()

        lines = [
            '# Generated by ranvar.compiler. Edits here are overwritten.',
            f'# Model: {self.name}',
            '',
            'from ranvar.ranvar cimport RanVar',
            '',
            '',
            f'cdef inline double {RESERVED}draw(RanVar v):',
            '    return v.sample()',
            '',
            '',
        ]

        if self.width == 0:
            lines.append(f'cdef double {RESERVED}body({", ".join(signature)}):')
        else:
            declared = signature + [f'double* {RESERVED}res']
            lines.append(f'cdef void {RESERVED}body({", ".join(declared)}):')

        for name in self.locals:
            lines.append(f'    cdef double {name}')

        lines.extend(body)
        lines.extend(['', ''])
        lines.extend(self.driver(signature, arguments))

        return '\n'.join(lines) + '\n'

    def driver(self, signature, arguments):
        """Build the loop that runs the model and collects what it returns.

        This is the part that has to stay in C: it calls the body and feeds the
        digests directly, so a whole simulation crosses into the interpreter
        once rather than once per sample.

        Args:
            signature (list): The model body's C parameter list.
            arguments (list): The model body's argument names.

        Returns:
            list: The generated lines.
        """
        passed = ', '.join(arguments)
        comma  = ', ' if arguments else ''

        if self.width == 0:
            return [
                f'def run(RanVar {RESERVED}out, int {RESERVED}samples'
                f'{comma}{", ".join(signature)}):',
                f'    cdef int {RESERVED}i',
                '',
                f'    for {RESERVED}i in range({RESERVED}samples):',
                f'        {RESERVED}out._add({RESERVED}body({passed}), 1.0)',
            ]

        lines = [
            f'def run(list {RESERVED}outs, int {RESERVED}samples'
            f'{comma}{", ".join(signature)}):',
            f'    cdef int {RESERVED}i',
            f'    cdef double {RESERVED}res[{self.width}]',
        ]

        for i in range(self.width):
            lines.append(f'    cdef RanVar {RESERVED}o{i} = {RESERVED}outs[{i}]')

        lines.extend([
            '',
            f'    for {RESERVED}i in range({RESERVED}samples):',
            f'        {RESERVED}body({passed}{comma}{RESERVED}res)',
        ])

        for i in range(self.width):
            lines.append(f'        {RESERVED}o{i}._add({RESERVED}res[{i}], 1.0)')

        return lines


def alwaysReturns(body):
    """Report whether a block of statements returns on every path.

    The generated body is a C function, so a path that falls off the end would
    hand back whatever happened to be in the return register.

    Args:
        body (list): The statements to check.

    Returns:
        bool: True if every path returns.
    """
    if not body:
        return False

    last = body[-1]

    if isinstance(last, ast.Return):
        return True

    if isinstance(last, ast.If):
        return (
            bool(last.orelse)
            and alwaysReturns(last.body)
            and alwaysReturns(last.orelse)
        )

    return False


# Building. --------------------------------------------------------------------

# Run in a subprocess so a build failure cannot leave setuptools' global state
# behind in the interpreter that asked for it.
BUILD = '''
import os
import sys

from setuptools import Extension, setup
from Cython.Build import cythonize

name, source, target, include = sys.argv[1:5]

setup(
    name=name,
    ext_modules=cythonize(
        [Extension(name, [source])],
        include_path=[include],
        quiet=True,
        language_level=3,
    ),
    script_args=[
        'build_ext',
        '--build-lib', target,
        '--build-temp', os.path.join(target, 'temp'),
        '-q',
    ],
)
'''


def cacheDir():
    """Locate the directory compiled models are kept in.

    Returns:
        str: Path to the cache directory, which may not exist yet.
    """
    base = os.environ.get('XDG_CACHE_HOME')

    if not base:
        base = os.path.join(os.path.expanduser('~'), '.cache')

    return os.path.join(base, 'ranvar')


def includeDir():
    """Locate the directory that has to be on Cython's include path.

    The generated module cimports ranvar.ranvar, so Cython needs the directory
    holding the ranvar package, not the package itself.

    Returns:
        str: Path to put on the include path.
    """
    import ranvar

    return os.path.dirname(os.path.dirname(os.path.abspath(ranvar.__file__)))


def cacheKey(source):
    """Build the name a generated module is cached under.

    Keyed on the generated source, the interpreter and the compiled RanVar the
    module will link against, so that rebuilding ranvar or changing Python does
    not leave a stale module behind that was built for a different layout.

    Args:
        source (str): The generated module source.

    Returns:
        str: A hex digest identifying this build.
    """
    from ranvar import ranvar as compiled

    digest = hashlib.sha256()
    digest.update(source.encode('utf-8'))
    digest.update(sys.version.encode('utf-8'))
    digest.update(importlib.machinery.EXTENSION_SUFFIXES[0].encode('utf-8'))

    status = os.stat(compiled.__file__)
    digest.update(f'{compiled.__file__}:{status.st_mtime_ns}:{status.st_size}'.encode('utf-8'))

    return digest.hexdigest()[:32]


def build(source, key):
    """Compile generated source into an importable module, or reuse the cache.

    The module is built in a private directory and moved into place in one step,
    so a half built module is never importable and two processes compiling the
    same model at once cannot see each other's work in progress.

    Args:
        source (str): The generated module source.
        key (str): The cache key from cacheKey().

    Returns:
        module: The imported module, exposing run().

    Raises:
        RanVarCompileError: If the build fails.
    """
    import shutil
    import tempfile

    directory = cacheDir()
    os.makedirs(directory, exist_ok=True)

    name   = f'ranvar_gen_{key}'
    suffix = importlib.machinery.EXTENSION_SUFFIXES[0]
    target = os.path.join(directory, name + suffix)

    if not os.path.exists(target):
        work = tempfile.mkdtemp(prefix=f'{name}.', dir=directory)

        try:
            pyx = os.path.join(work, name + '.pyx')

            with open(pyx, 'w') as handle:
                handle.write(source)

            result = subprocess.run(
                [sys.executable, '-c', BUILD, name, pyx, work, includeDir()],
                capture_output=True,
                text=True,
                cwd=work,
            )

            if result.returncode != 0:
                raise RanVarCompileError(
                    f'compiling the model failed:\n\n'
                    f'{(result.stderr or result.stdout).strip()[-2000:]}'
                )

            # Kept beside the module so a compiled model can be read back.
            shutil.copyfile(pyx, os.path.join(directory, name + '.pyx'))
            os.replace(os.path.join(work, name + suffix), target)
        finally:
            shutil.rmtree(work, ignore_errors=True)

    spec   = importlib.util.spec_from_file_location(name, target)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


# The decorator. ---------------------------------------------------------------

class CompiledMonteCarlo():
    """A function compiled into a monte carlo simulation that runs in C.

    Like MonteCarlo, calling the wrapper runs the model repeatedly and collects
    the results into a RanVar. Unlike it, the model is translated into Cython
    and the loop around it is compiled, so a call crosses into the interpreter
    once rather than once per sample.

    The translation only covers arithmetic over numbers and draws from RanVars.
    Anything else is refused when the model is compiled rather than run through
    the interpreter, so a model either gets the speed the decorator is for or
    says why it cannot.

    Compiling takes several seconds, so it happens on the first call rather than
    at decoration, and the result is cached on disk and reused by later runs and
    later processes.

    Attributes:
        samples (int): Number of runs a plain call performs.
        maxBins (int): Number of centroids each returned RanVar maintains.
    """

    def __init__(self, fn, samples=DEFAULT_SAMPLES, maxBins=DEFAULT_MAXBINS):
        """Wrap a function in a compiled monte carlo executor.

        Args:
            fn (callable): The model to compile.
            samples (int, optional): Runs per call. Defaults to DEFAULT_SAMPLES.
            maxBins (int, optional): Centroids in each returned RanVar.
                                   Defaults to DEFAULT_MAXBINS.

        Raises:
            TypeError: If fn is not callable.
        """
        if not callable(fn):
            raise TypeError(
                f'cfunc expected a callable, got {type(fn).__name__!r}. '
                f'To set options use the keyword form, @cfunc(samples=...)'
            )

        functools.update_wrapper(self, fn)

        self.samples = _checkSamples(samples)
        self.maxBins = maxBins

        self.module     = None
        self.translator = None

    def compile(self):
        """Compile the model now instead of on the first call.

        Returns:
            CompiledMonteCarlo: This wrapper, so the call can be chained.

        Raises:
            RanVarCompileError: If the model cannot be compiled.
        """
        if self.module is not None:
            return self

        translator = Translator(self.__wrapped__)
        source     = translator.generate()

        self.module     = build(source, cacheKey(source))
        self.translator = translator

        return self

    def __call__(self, *args, **kwargs):
        """Run the compiled simulation with the default sample count.

        Args:
            *args: Numeric arguments for the model.
            **kwargs: Numeric arguments for the model.

        Returns:
            RanVar: The distribution of the results, or a tuple of them for a
                  model returning several values.
        """
        return self._run(self.samples, args, kwargs)

    def withSamples(self, samples):
        """Get an equivalent wrapper that runs a different number of samples.

        The compiled module is shared, so overriding the count does not compile
        the model again.

        Args:
            samples (int): Number of runs the returned wrapper performs.

        Returns:
            CompiledMonteCarlo: A wrapper around the same model.
        """
        other = CompiledMonteCarlo(
            self.__wrapped__, samples=samples, maxBins=self.maxBins
        )

        other.module     = self.module
        other.translator = self.translator

        return other

    def _bind(self, args, kwargs):
        """Work out the arguments the compiled loop takes for this call.

        The model's own parameters come from the call, and the names it reads
        from the surrounding scope are looked up again now rather than being
        fixed when it was compiled, so rebinding one between calls is seen.

        Args:
            args (tuple): Positional arguments for the model.
            kwargs (dict): Keyword arguments for the model.

        Returns:
            list: The arguments to hand to the compiled loop.

        Raises:
            TypeError: If a name has changed into something the compiled module
                     cannot take.
        """
        translator = self.translator
        bound      = inspect.signature(self.__wrapped__).bind(*args, **kwargs)

        bound.apply_defaults()

        values = [float(bound.arguments[name]) for name in translator.params]

        for name in translator.free:
            value = translator.resolve(name)

            if name in translator.draws:
                if not isinstance(value, RanVar):
                    raise TypeError(
                        f'{self.__name__}(): {name!r} was a RanVar when the model '
                        f'was compiled and is now a {type(value).__name__!r}'
                    )

                values.append(value)
            else:
                values.append(float(value))

        return values

    def _run(self, samples, args, kwargs):
        """Compile if needed, then run the simulation.

        Args:
            samples (int): Number of runs to perform.
            args (tuple): Positional arguments for the model.
            kwargs (dict): Keyword arguments for the model.

        Returns:
            RanVar: The distribution of the results, or a tuple of them for a
                  model returning several values.
        """
        self.compile()

        values = self._bind(args, kwargs)

        if self.translator.width == 0:
            out = RanVar(maxBins=self.maxBins)
            self.module.run(out, samples, *values)

            return out

        outs = [RanVar(maxBins=self.maxBins) for _ in range(self.translator.width)]
        self.module.run(outs, samples, *values)

        return tuple(outs)

    def __repr__(self):
        state = 'compiled' if self.module is not None else 'not compiled yet'

        return (
            f'<CompiledMonteCarlo {self.__name__} '
            f'samples={self.samples} maxBins={self.maxBins} {state}>'
        )


def cfunc(fn=None, *, samples=DEFAULT_SAMPLES, maxBins=DEFAULT_MAXBINS):
    """Compile a function into a monte carlo simulation that runs in C.

    The same idea as func, but the model is translated into Cython and the loop
    around it compiled, so the whole simulation runs without returning to the
    interpreter. In exchange the model has to stay within what can be turned
    into C: arithmetic over numbers, draws from RanVars with ~, assignments,
    if and else, and a return of one value or a tuple of them.

    Compiling takes several seconds and happens on the first call, not at
    decoration. The result is cached on disk under ~/.cache/ranvar and reused
    by later calls and later processes.

    Args:
        fn (callable, optional): The model to compile, supplied when the
                               decorator is used bare.
        samples (int, optional): Runs per call. Defaults to DEFAULT_SAMPLES.
        maxBins (int, optional): Centroids in each returned RanVar.
                               Defaults to DEFAULT_MAXBINS.

    Returns:
        CompiledMonteCarlo: The wrapped model, or a decorator returning one when
                          options are given.

    Raises:
        RanVarCompileError: On the first call, if the model cannot be compiled.

    Example:
        >>> @cfunc
        ... def profit():
        ...     return ~revenue - ~cost
        >>> profit()
    """
    _checkSamples(samples)

    if fn is None:
        def decorator(fn):
            return CompiledMonteCarlo(fn, samples=samples, maxBins=maxBins)

        return decorator

    return CompiledMonteCarlo(fn, samples=samples, maxBins=maxBins)
