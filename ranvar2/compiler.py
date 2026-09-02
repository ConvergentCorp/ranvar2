import ast
import functools
import hashlib
import importlib.machinery
import importlib.util
import inspect
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap

from ranvar2.decorators import Bound, DEFAULT_MAXBINS, DEFAULT_SAMPLES, _checkSamples
from ranvar2.ranvar import RanVar, RanVarArray


class RanVarCompileError(Exception):
    """Raised when a model cannot be compiled into a C simulation."""


# The generated module names its own machinery with this prefix, so a model
# using a name of its own that starts with it would be shadowed.
RESERVED = '_rv_'

# Return annotations that mean the model hands back a C number rather than a
# Python object, and so returns exactly one value.
# Distinguishes a name bound to None from one that is not bound at all.
MISSING = object()

SCALARS = {
    'cython.double', 'cython.float', 'cython.int', 'cython.long',
    'cython.longlong', 'cython.size_t', 'cython.Py_ssize_t',
    'double', 'float', 'int',
}


class Model():
    """A function prepared for compilation.

    The model's own source is what gets compiled, in Cython's pure Python mode,
    so its types come from ordinary annotations rather than from anything this
    module works out. A parameter annotated as a RanVar is reached through the
    extension type's method table rather than through Python, which is what
    makes a draw a C call.

    The single thing not carried over verbatim is ~. The invert slot is defined
    to return a Python object, so ~a boxes a float however the argument is
    annotated; a draw from a RanVar parameter is rewritten to a.sample(), which
    does compile to a direct call.

    Attributes:
        name (str): The model's name.
        params (list): The model's parameter names, in order.
        digests (list): Which of those are annotated as RanVars.
        arrays (list): Which of those are annotated as RanVarArrays.
        globals (list): Names the model reads from its defining module.
        scalar (bool): Whether the model is annotated as returning a C number.
        source (str): The rewritten model, ready to paste into a module.
    """

    def __init__(self, fn):
        """Prepare a function for compilation.

        Args:
            fn (callable): The model to compile.

        Raises:
            RanVarCompileError: If the model cannot be prepared.
        """
        self.fn      = fn
        self.name    = getattr(fn, '__name__', '<model>')
        self.params  = []
        self.digests = []
        self.arrays  = []
        self.globals = []

        node = self.parse()

        # A model annotated as returning a C number is collected by a loop that
        # never boxes it, and cannot be returning several values. Without such
        # an annotation the result is a Python object, which might be a tuple,
        # so both loops are generated and the model is asked once which it is.
        self.scalar = ast.unparse(node.returns) in SCALARS if node.returns else False

        self.signature(node)
        self.scope(node)

        self.attributes(node)
        self.arrayDraws(node)

        node.decorator_list = []

        loops = ArrayLoops(self.arrays, self.fail)
        node  = loops.visit(node)

        node = Draws(self.digests + loops.bound, self.arrays).visit(node)

        ast.fix_missing_locations(node)

        self.source = ast.unparse(node)

    def fail(self, message):
        """Report that the model cannot be compiled.

        Args:
            message (str): What is wrong.

        Raises:
            RanVarCompileError: Always.
        """
        raise RanVarCompileError(f'{self.name}(): {message}')

    def parse(self):
        """Parse the model's own source.

        Returns:
            ast.FunctionDef: The model's definition.

        Raises:
            RanVarCompileError: If the source cannot be read or is not a plain
                              function definition.
        """
        try:
            source = inspect.getsource(self.fn)
        except (OSError, TypeError):
            self.fail(
                'cannot read the source of this function, so it cannot be '
                'compiled. cfunc needs a function defined in a file'
            )

        node = ast.parse(textwrap.dedent(source)).body[0]

        if not isinstance(node, ast.FunctionDef):
            self.fail('only a plain function can be compiled')

        return node

    def signature(self, node):
        """Record the model's parameters and which of them are digests.

        Args:
            node (ast.FunctionDef): The model's definition.

        Raises:
            RanVarCompileError: If the signature cannot be mirrored by the
                              generated driver.
        """
        args = node.args

        if args.vararg is not None or args.kwarg is not None:
            self.fail('*args and **kwargs cannot be compiled')

        for arg in args.posonlyargs + args.args + args.kwonlyargs:
            if arg.arg.startswith(RESERVED):
                self.fail(f'{arg.arg!r} is reserved for the generated module')

            self.params.append(arg.arg)

            # Cython can only resolve this annotation to the bare name the
            # generated module cimports, not to however the model happened to
            # qualify it (mc.RanVar, after `import ranvar2 as mc`, say): an
            # annotation it cannot resolve is accepted anyway, silently, as an
            # ordinary (untyped) parameter rather than raising -- so a
            # qualified name compiles and runs, just without the direct C
            # call annotating it as RanVar was for. Normalising it here, on
            # the node this function's own signature will later be unparsed
            # from, is what makes every way of spelling it compile the same.
            if isRanVar(arg.annotation):
                self.digests.append(arg.arg)
                arg.annotation = ast.Name(id='RanVar', ctx=ast.Load())
            elif isRanVarArray(arg.annotation):
                self.arrays.append(arg.arg)
                arg.annotation = ast.Name(id='RanVarArray', ctx=ast.Load())

        self.annotations = {
            arg.arg: ast.unparse(arg.annotation) if arg.annotation else None
            for arg in args.posonlyargs + args.args + args.kwonlyargs
        }

    def scope(self, node):
        """Find the names the model reads from the module it was defined in.

        Those are carried across into the generated module so that a model may
        use whatever it likes. A digest read this way is refused, though: a
        module level name is a Python object to Cython whatever it holds, so
        drawing from one would silently be slow, which is the whole thing cfunc
        exists to avoid.

        Args:
            node (ast.FunctionDef): The model's definition.

        Raises:
            RanVarCompileError: If the model draws from a digest that is not a
                              parameter.
        """
        assigned = set(self.params)
        body     = list(self.body(node))

        for child in body:
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
                assigned.add(child.id)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                assigned.add(child.name)
            elif isinstance(child, ast.arg):
                assigned.add(child.arg)

        for child in body:
            if not isinstance(child, ast.Name) or not isinstance(child.ctx, ast.Load):
                continue

            if child.id in assigned or child.id in self.globals:
                continue

            if self.lookup(child.id) is MISSING:
                continue

            self.globals.append(child.id)

        for name in self.globals:
            value = self.lookup(name)
            kind  = 'RanVar' if isinstance(value, RanVar) else (
                'RanVarArray' if isinstance(value, RanVarArray) else None
            )

            if kind is not None:
                self.fail(
                    f'{name!r} is a {kind} read from the surrounding module. A '
                    f'compiled model has to take its digests as parameters '
                    f'annotated RanVar or RanVarArray, so that drawing from them '
                    f'compiles to a C call'
                )

    def body(self, node):
        """Visit the model's statements, without their annotations.

        The decorator list and the signature are part of the definition rather
        than of what runs, and an annotation names types the generated module
        supplies itself, so a name used in one is not something the model reads
        from its own module.

        Args:
            node (ast.FunctionDef): The model's definition.

        Yields:
            ast.AST: Each node under one of the body's statements.
        """
        stack = list(node.body)

        while stack:
            current = stack.pop()

            yield current

            annotation = getattr(current, 'annotation', None)

            for child in ast.iter_child_nodes(current):
                if annotation is not None and child is annotation:
                    continue

                stack.append(child)

    def attributes(self, node):
        """Refuse a draw from something reached through an attribute.

        ~self.digest is the method shaped version of drawing from a module level
        name: it compiles, but the attribute is a Python object to Cython, so it
        is not the C call the decorator exists to give. Unlike a module level
        name there is no value to inspect at compile time, so the shape of the
        expression is what it is judged on.

        Args:
            node (ast.FunctionDef): The model's definition.

        Raises:
            RanVarCompileError: If ~ is applied to an attribute.
        """
        for child in self.body(node):
            if not isinstance(child, ast.UnaryOp) or not isinstance(child.op, ast.Invert):
                continue

            if not isinstance(child.operand, ast.Attribute):
                continue

            drawn = ast.unparse(child.operand)

            self.fail(
                f'~{drawn} draws from an attribute. A compiled model has to take '
                f'its digests as parameters annotated RanVar, so that drawing '
                f'from them compiles to a C call. If {drawn} is a number rather '
                f'than a digest, assign it to a local first'
            )

    def arrayDraws(self, node):
        """Refuse a direct draw from a RanVarArray parameter.

        A RanVarArray holds several digests, not one, so it has no single value
        for ~ to return: an element has to be picked out first, by indexing or
        by iterating.

        Args:
            node (ast.FunctionDef): The model's definition.

        Raises:
            RanVarCompileError: If ~ is applied directly to a RanVarArray
                              parameter.
        """
        for child in self.body(node):
            if not isinstance(child, ast.UnaryOp) or not isinstance(child.op, ast.Invert):
                continue

            operand = child.operand

            if isinstance(operand, ast.Name) and operand.id in self.arrays:
                self.fail(
                    f'~{operand.id} draws from a RanVarArray directly. It '
                    f'holds several digests, not one: iterate over it instead, '
                    f'e.g. `for x in {operand.id}: ... ~x`'
                )

            if (
                isinstance(operand, ast.Subscript)
                and isinstance(operand.value, ast.Name)
                and operand.value.id in self.arrays
                and isinstance(operand.slice, ast.Slice)
            ):
                self.fail(
                    f'~{ast.unparse(operand)} draws from a slice of a '
                    f'RanVarArray, which is still several digests rather than '
                    f'one. Index a single element instead, e.g. '
                    f'~{operand.value.id}[0]'
                )

    def lookup(self, name):
        """Look a name up in the scope the model was defined in.

        Builtins are deliberately not searched: Cython resolves those itself,
        and declaring one in the generated module would shadow it.

        Args:
            name (str): The name to resolve.

        Returns:
            object: What the name is bound to, or MISSING if it is not bound in
                  the model's closure or module.
        """
        code = self.fn.__code__

        if name in code.co_freevars and self.fn.__closure__ is not None:
            cell = self.fn.__closure__[code.co_freevars.index(name)]

            try:
                return cell.cell_contents
            except ValueError:
                return MISSING

        if name in self.fn.__globals__:
            return self.fn.__globals__[name]

        return MISSING

    def bindings(self):
        """Collect the current values of the names the model reads.

        Read afresh rather than kept from compile time, so rebinding one between
        calls is seen the way it would be in the interpreter.

        Returns:
            dict: Name to value, for the names bound anywhere reachable.
        """
        found = {}

        for name in self.globals:
            value = self.lookup(name)

            if value is not MISSING:
                found[name] = value

        return found

    def declaration(self, name):
        """Render one parameter as the generated driver should declare it.

        Args:
            name (str): The parameter name.

        Returns:
            str: The parameter, with its annotation if the model gave one.
        """
        annotation = self.annotations.get(name)

        return f'{name}: {annotation}' if annotation else name

    def generate(self):
        """Build the source of the module that runs this model.

        Returns:
            str: The module source, for Cython's pure Python mode.
        """
        declared = [self.declaration(name) for name in self.params]
        passed   = ', '.join(self.params)
        comma    = ', ' if declared else ''
        joined   = ', '.join(declared)

        # Cython resolves module level names when it compiles, not when the
        # module runs, so anything the model reads has to exist here even though
        # the value is only put in place at call time.
        declarations = ''.join(f'{name} = None\n' for name in self.globals)

        source = MODULE.format(
            name=self.name,
            model=self.source,
            declarations=declarations,
            declared=joined,
            passed=passed,
            comma=comma,
            prefix=RESERVED,
        )

        if not self.scalar:
            source += MANY.format(
                name=self.name,
                declared=joined,
                passed=passed,
                comma=comma,
                prefix=RESERVED,
            )

        return source


MODULE = '''# Generated by ranvar2.compiler. Edits here are overwritten.
# Model: {name}

import cython
from cython.cimports.ranvar2.ranvar import RanVar, RanVarArray

{declarations}

@cython.ccall
{model}


def {prefix}run({prefix}out: RanVar, {prefix}samples: cython.int{comma}{declared}):
    {prefix}i: cython.int

    for {prefix}i in range({prefix}samples):
        {prefix}out._add({name}({passed}), 1.0)


'''


MANY = '''

def {prefix}runMany({prefix}outs: list, {prefix}samples: cython.int{comma}{declared}):
    {prefix}i: cython.int
    {prefix}k: cython.int
    {prefix}width: cython.int = len({prefix}outs)
    {prefix}digest: RanVar

    for {prefix}i in range({prefix}samples):
        {prefix}values = {name}({passed})

        for {prefix}k in range({prefix}width):
            {prefix}digest = {prefix}outs[{prefix}k]
            {prefix}digest._add({prefix}values[{prefix}k], 1.0)
'''


class Draws(ast.NodeTransformer):
    """Rewrites a draw from a digest into a call to sample().

    ~a and a.sample() mean the same thing, but the invert slot has to return a
    Python object, so only the second compiles to a C call. Only parameters
    annotated as RanVars (directly, or as an element of a RanVarArray reached
    by indexing) are rewritten, which leaves ~ on an integer meaning what it
    always meant.

    arr[i], arr a RanVarArray parameter, is also rewritten on its own -- into
    arr._at(i) -- independently of whether it sits under a ~, since arr._at()
    is a proper index into the array (bounds checked, negative indices
    wraparound like a list's) rather than the generic __getitem__ a plain
    subscript on arr would otherwise call.
    """

    def __init__(self, digests, arrays):
        """Prepare to rewrite draws.

        Args:
            digests (list): Names of the parameters annotated as RanVars, or
                          bound as a RanVarArray loop's element.
            arrays (list): Names of the parameters annotated RanVarArray.
        """
        self.digests = set(digests)
        self.arrays  = set(arrays)

    def visit_Subscript(self, node):
        """Rewrite arr[i] into arr._at(i), for arr a RanVarArray parameter.

        Args:
            node (ast.Subscript): The subscript expression.

        Returns:
            ast.AST: The rewritten node, or the original.
        """
        self.generic_visit(node)

        if not isinstance(node.ctx, ast.Load):
            return node

        if not isinstance(node.value, ast.Name) or node.value.id not in self.arrays:
            return node

        # A slice (arr[1:3]) still needs RanVarArray's own __getitem__, since
        # _at() only fetches one element; left alone, so ~arr[1:3] fails the
        # same way ~arr does rather than being rewritten into nonsense.
        if isinstance(node.slice, ast.Slice):
            return node

        return ast.Call(
            func=ast.Attribute(value=node.value, attr='_at', ctx=ast.Load()),
            args=[node.slice],
            keywords=[],
        )

    def visit_UnaryOp(self, node):
        """Rewrite ~digest (or ~arr[i]) into a call to sample().

        Args:
            node (ast.UnaryOp): The expression node.

        Returns:
            ast.AST: The rewritten node, or the original.
        """
        self.generic_visit(node)

        if not isinstance(node.op, ast.Invert):
            return node

        operand = node.operand

        # generic_visit() above has already run visit_Subscript() on operand,
        # so arr[i] is by this point arr._at(i) rather than a Subscript node.
        isDigest = isinstance(operand, ast.Name) and operand.id in self.digests
        isElement = (
            isinstance(operand, ast.Call)
            and isinstance(operand.func, ast.Attribute)
            and operand.func.attr == '_at'
            and isinstance(operand.func.value, ast.Name)
            and operand.func.value.id in self.arrays
        )

        if not (isDigest or isElement):
            return node

        return ast.Call(
            func=ast.Attribute(value=operand, attr='sample', ctx=ast.Load()),
            args=[],
            keywords=[],
        )


class ArrayLoops(ast.NodeTransformer):
    """Rewrites `for x in arr:` into an indexed loop, for arr a RanVarArray.

    Cython's generic iteration protocol boxes every element it yields as a
    Python object, so a plain `for x in arr:` would pay that cost per element
    no matter how the loop body is typed. Rewritten as an indexed loop over
    `range(len(arr))` that fetches each element with arr._at() -- whose
    declared return type is RanVar -- the fetch is a direct C call instead,
    and so is everything the body does with x afterwards, ~x included, once
    Draws also treats x as a digest. That is why the loop variable's name is
    collected here rather than rewritten directly: Draws is what turns ~x into
    x.sample(), the same as it does for a RanVar parameter.

    Only `for x in arr:`, arr a bare parameter reference and x a single name,
    is rewritten here; a fixed number of elements reached by indexing (`for i
    in range(5): ~arr[i]`) gets its own fast path in Draws instead, since
    there is no single loop variable there for this pass to name. Unpacking or
    slicing before iterating is left as ordinary Python, which still runs,
    just without the same speed.

    Attributes:
        arrays (set): Names of the parameters annotated RanVarArray.
        bound (list): Names introduced as loop targets over one of those, so
                    Draws also treats them as digests.
    """

    def __init__(self, arrays, fail):
        """Prepare to rewrite loops over RanVarArray parameters.

        Args:
            arrays (list): Names of the parameters annotated RanVarArray.
            fail (callable): Raises RanVarCompileError, given a message.
        """
        self.arrays  = set(arrays)
        self.fail    = fail
        self.bound   = []
        self.counter = 0

    def visit_For(self, node):
        """Rewrite one for loop, if it iterates a RanVarArray parameter directly.

        Args:
            node (ast.For): The loop to inspect.

        Returns:
            ast.AST: The original node, unless it iterates a RanVarArray
                    parameter directly, in which case a [counter declaration,
                    indexed loop] pair replacing it.

        Raises:
            RanVarCompileError: If such a loop tries to unpack its element into
                              more than one name.
        """
        self.generic_visit(node)

        if not isinstance(node.iter, ast.Name) or node.iter.id not in self.arrays:
            return node

        if not isinstance(node.target, ast.Name):
            self.fail(
                f'for ... in {node.iter.id} cannot unpack into more than one '
                f'name; a RanVarArray yields one RanVar at a time'
            )

        array  = node.iter.id
        target = node.target.id
        index  = f'{RESERVED}i{self.counter}'

        self.counter += 1
        self.bound.append(target)

        counterDecl = ast.AnnAssign(
            target=ast.Name(id=index, ctx=ast.Store()),
            annotation=ast.Attribute(
                value=ast.Name(id='cython', ctx=ast.Load()), attr='int', ctx=ast.Load()
            ),
            value=None,
            simple=1,
        )

        itemDecl = ast.AnnAssign(
            target=ast.Name(id=target, ctx=ast.Store()),
            annotation=ast.Name(id='RanVar', ctx=ast.Load()),
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id=array, ctx=ast.Load()), attr='_at', ctx=ast.Load()
                ),
                args=[ast.Name(id=index, ctx=ast.Load())],
                keywords=[],
            ),
            simple=1,
        )

        indexed = ast.For(
            target=ast.Name(id=index, ctx=ast.Store()),
            iter=ast.Call(
                func=ast.Name(id='range', ctx=ast.Load()),
                args=[ast.Call(
                    func=ast.Name(id='len', ctx=ast.Load()),
                    args=[ast.Name(id=array, ctx=ast.Load())],
                    keywords=[],
                )],
                keywords=[],
            ),
            body=[itemDecl] + node.body,
            orelse=node.orelse,
        )

        return [counterDecl, indexed]


def isRanVar(annotation):
    """Report whether an annotation names the RanVar type.

    Args:
        annotation (ast.AST): The annotation node, or None.

    Returns:
        bool: True if the annotation is RanVar or something.RanVar.
    """
    if isinstance(annotation, ast.Name):
        return annotation.id == 'RanVar'

    if isinstance(annotation, ast.Attribute):
        return annotation.attr == 'RanVar'

    return False


def isRanVarArray(annotation):
    """Report whether an annotation names the RanVarArray type.

    Args:
        annotation (ast.AST): The annotation node, or None.

    Returns:
        bool: True if the annotation is RanVarArray or something.RanVarArray.
    """
    if isinstance(annotation, ast.Name):
        return annotation.id == 'RanVarArray'

    if isinstance(annotation, ast.Attribute):
        return annotation.attr == 'RanVarArray'

    return False


# Building. --------------------------------------------------------------------

def optimiseFlags():
    """Compiler flags a generated model is built with.

    setuptools otherwise hands the model Python's own CFLAGS, which stop at
    -O2. These are the same flags setup.py builds ranvar2.ranvar itself with,
    so a model is optimised to the same level as the digest it calls into.

    -march=native is opt-in for the same reason it is in setup.py: it bakes in
    the build machine's instruction set. A generated model is cached under
    ~/.cache/ranvar2 and reused by later processes, so the flags are part of
    the cache key -- changing them rebuilds rather than reusing a module built
    with the old ones.

    Returns:
        list: Flags to pass as extra_compile_args.
    """
    if sys.platform == 'win32':
        return ['/O2']

    flags = ['-O3', '-fno-math-errno']

    if os.environ.get('RANVAR2_NATIVE'):
        flags.append('-march=native')

    return flags


# Run in a subprocess so a build failure cannot leave setuptools' global state
# behind in the interpreter that asked for it.
BUILD = '''
import json
import os
import sys

from setuptools import Extension, setup
from Cython.Build import cythonize

name, source, target, include, flags = sys.argv[1:6]

setup(
    name=name,
    ext_modules=cythonize(
        [Extension(name, [source], extra_compile_args=json.loads(flags))],
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

    return os.path.join(base, 'ranvar2')


def includeDir():
    """Locate the directory that has to be on Cython's include path.

    The generated module cimports ranvar2.ranvar, so Cython needs the directory
    holding the ranvar2 package rather than the package itself.

    Returns:
        str: Path to put on the include path.
    """
    import ranvar2

    return os.path.dirname(os.path.dirname(os.path.abspath(ranvar2.__file__)))


def cacheKey(source):
    """Build the name a generated module is cached under.

    Keyed on the generated source, the interpreter, the compiled RanVar the
    module links against and the flags it is built with, so neither rebuilding
    ranvar2 nor changing the optimisation level leaves a stale module behind
    that was built against a different layout.

    Args:
        source (str): The generated module source.

    Returns:
        str: A hex digest identifying this build.
    """
    from ranvar2 import ranvar as compiled

    digest = hashlib.sha256()
    digest.update(source.encode('utf-8'))
    digest.update(sys.version.encode('utf-8'))
    digest.update(importlib.machinery.EXTENSION_SUFFIXES[0].encode('utf-8'))
    digest.update(repr(optimiseFlags()).encode('utf-8'))

    status = os.stat(compiled.__file__)
    digest.update(f'{compiled.__file__}:{status.st_mtime_ns}:{status.st_size}'.encode('utf-8'))

    return digest.hexdigest()[:32]


def build(source, key):
    """Compile generated source into an importable module, or reuse the cache.

    The module is built in a private directory and moved into place in one step,
    so a half built module is never importable and two processes compiling the
    same model cannot see each other's work in progress.

    Args:
        source (str): The generated module source.
        key (str): The cache key from cacheKey().

    Returns:
        module: The imported module, exposing the run functions.

    Raises:
        RanVarCompileError: If the build fails.
    """
    directory = cacheDir()
    os.makedirs(directory, exist_ok=True)

    name   = f'ranvar_gen_{key}'
    suffix = importlib.machinery.EXTENSION_SUFFIXES[0]
    target = os.path.join(directory, name + suffix)

    if not os.path.exists(target):
        work = tempfile.mkdtemp(prefix=f'{name}.', dir=directory)

        try:
            # A .py rather than a .pyx: pure Python mode is what lets the
            # model's own annotations carry its types.
            module = os.path.join(work, name + '.py')

            with open(module, 'w') as handle:
                handle.write(source)

            result = subprocess.run(
                [
                    sys.executable, '-c', BUILD, name, module, work,
                    includeDir(), json.dumps(optimiseFlags()),
                ],
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
            shutil.copyfile(module, os.path.join(directory, name + '.py'))
            os.replace(os.path.join(work, name + suffix), target)
        finally:
            shutil.rmtree(work, ignore_errors=True)

    spec     = importlib.util.spec_from_file_location(name, target)
    imported = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(imported)

    return imported


# The decorator. ---------------------------------------------------------------

class CompiledMonteCarlo():
    """A model compiled into a monte carlo simulation that runs in C.

    Like MonteCarlo, calling the wrapper runs the model repeatedly and collects
    the results into a RanVar. Unlike it, the model's own source is compiled by
    Cython and the loop around it with it, so a call crosses into the
    interpreter once rather than once per sample.

    The model is compiled in Cython's pure Python mode, which takes its types
    from ordinary annotations. Any Python compiles; how much of it becomes C
    depends on how much is annotated. Digests have to arrive as parameters
    annotated RanVar, since a module level name is a Python object to Cython
    whatever it holds and drawing from one would quietly be slow.

    Compiling takes several seconds, so it happens on the first call rather than
    at decoration, and the result is cached on disk and reused by later runs and
    later processes.

    Attributes:
        samples (int): Number of runs a plain call performs.
        maxBins (int): Number of centroids each returned RanVar maintains.

    Example:
        >>> @cfunc
        ... def profit(revenue: RanVar, cost: RanVar) -> cython.double:
        ...     return ~revenue - ~cost
        >>> profit(revenueDigest, costDigest)
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

        self.module = None
        self.model  = None
        self.width  = None

    def compile(self):
        """Compile the model now instead of on the first call.

        Returns:
            CompiledMonteCarlo: This wrapper, so the call can be chained.

        Raises:
            RanVarCompileError: If the model cannot be compiled.
        """
        if self.module is not None:
            return self

        model  = Model(self.__wrapped__)
        source = model.generate()

        self.module = build(source, cacheKey(source))
        self.model  = model

        return self

    def __call__(self, *args, **kwargs):
        """Run the compiled simulation with the default sample count.

        Args:
            *args: Arguments for the model, digests included.
            **kwargs: Arguments for the model, digests included.

        Returns:
            RanVar: The distribution of the results, or a tuple of them for a
                  model returning several values.
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

        other.module = self.module
        other.model  = self.model
        other.width  = self.width

        return other

    def _prepare(self, args, kwargs):
        """Get the model's arguments in order, and refresh what it reads.

        Args:
            args (tuple): Positional arguments for the model.
            kwargs (dict): Keyword arguments for the model.

        Returns:
            list: The arguments to hand to the compiled driver.
        """
        bound = inspect.signature(self.__wrapped__).bind(*args, **kwargs)

        bound.apply_defaults()

        # The generated module has a namespace of its own, so whatever the model
        # reads from the module it was written in is copied across before it
        # runs, and copied again each call in case any of it has been rebound.
        for name, value in self.model.bindings().items():
            setattr(self.module, name, value)

        return [bound.arguments[name] for name in self.model.params]

    def _shape(self, values):
        """Work out how many values the model returns, by running it once.

        A model annotated as returning a C number is settled by the annotation
        alone. Otherwise it is compiled with @cython.ccall, so it stays callable
        from Python and can be asked directly rather than guessed at, at the
        cost of one extra evaluation the first time.

        Args:
            values (list): The model's arguments.

        Returns:
            int: The number of values returned, 0 for a single one.
        """
        if self.width is not None:
            return self.width

        if self.model.scalar:
            self.width = 0

            return self.width

        probe = getattr(self.module, self.model.name)(*values)

        self.width = len(probe) if isinstance(probe, tuple) else 0

        return self.width

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

        values = self._prepare(args, kwargs)
        width  = self._shape(values)

        if width == 0:
            out = RanVar(maxBins=self.maxBins)
            getattr(self.module, f'{RESERVED}run')(out, samples, *values)

            return out

        outs = [RanVar(maxBins=self.maxBins) for _ in range(width)]
        getattr(self.module, f'{RESERVED}runMany')(outs, samples, *values)

        return tuple(outs)

    def __repr__(self):
        """Describe this wrapper, for debugging.

        Returns:
            str: The wrapped model's name, the wrapper's settings, and whether
               it has been compiled yet.
        """
        state = 'compiled' if self.module is not None else 'not compiled yet'

        return (
            f'<CompiledMonteCarlo {self.__name__} '
            f'samples={self.samples} maxBins={self.maxBins} {state}>'
        )


def cfunc(fn=None, *, samples=DEFAULT_SAMPLES, maxBins=DEFAULT_MAXBINS):
    """Compile a model into a monte carlo simulation that runs in C.

    The same idea as func, but the model's own source is compiled by Cython in
    pure Python mode and the loop around it compiled with it, so the simulation
    runs without returning to the interpreter.

    The model stays ordinary Python. Its types come from ordinary annotations,
    and how much of it becomes C rather than object code depends on how much of
    it is annotated. Digests have to arrive as parameters annotated RanVar: a
    module level name is a Python object to Cython whatever it holds, so a model
    drawing from one would compile but not go fast, and that is refused rather
    than allowed to look like a win.

    ~ is the one thing not carried across verbatim. The invert slot has to
    return a Python object, so a draw from a digest parameter is rewritten to
    sample(), which is the same thing and does compile to a direct call.

    Compiling takes several seconds and happens on the first call, not at
    decoration. The result is cached under ~/.cache/ranvar2 and reused by later
    calls and later processes.

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
        >>> import cython
        >>> from ranvar2 import RanVar, cfunc

        >>> @cfunc
        ... def profit(revenue: RanVar, cost: RanVar) -> cython.double:
        ...     margin: cython.double = ~revenue - ~cost
        ...     return margin

        >>> profit(revenueDigest, costDigest)
    """
    _checkSamples(samples)

    if fn is None:
        def decorator(fn):
            return CompiledMonteCarlo(fn, samples=samples, maxBins=maxBins)

        return decorator

    return CompiledMonteCarlo(fn, samples=samples, maxBins=maxBins)
