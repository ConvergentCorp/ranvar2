from .ranvar import RanVar, Normal, NegBinom, seed
from .decorators import func
from .compiler import cfunc, RanVarCompileError

# The digest is the representation backing a random variable. ``Digest`` is kept
# as an alias because that is the name the tests and the older code use.
Digest = RanVar

__all__ = [
    "RanVar", "Digest", "Normal", "NegBinom", "func", "cfunc", "seed",
    "RanVarCompileError",
]
