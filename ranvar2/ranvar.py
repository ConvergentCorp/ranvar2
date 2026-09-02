import numbers

import numpy  as np
from cython import (
    cclass, cfunc, boundscheck, initializedcheck, wraparound, ccall, cdivision,
    cast, final, address, sizeof, declare, exceptval,
)

from cython import p_double

from cython import double as cdouble
from cython import int as cint
from cython import void as cvoid

from cython.cimports.libc.string import memmove
from cython.cimports.libc.string import memcpy

from cython.cimports.libc.stdint import uint64_t

from cython.cimports.libc.math import log as clog
from cython.cimports.libc.math import sqrt as csqrt
from cython.cimports.libc.math import cos as ccos
from cython.cimports.libc.math import sin as csin
from cython.cimports.libc.math import exp as cexp
from cython.cimports.libc.math import floor as cfloor
from cython.cimports.libc.math import fabs as cfabs
from cython.cimports.libc.math import lgamma as clgamma
from cython.cimports.libc.math import pow as cpow
from cython.cimports.libc.math import erf as cerf
from cython.cimports.libc.math import erfc as cerfc
from cython.cimports.libc.math import M_PI
from cython.cimports.libc.math import M_SQRT2


# Box-Muller turns one pair of uniforms into two independent standard normal
# variates, and _standardNormal() used to compute both and return one. The
# spare is kept here instead, so every second draw is a load rather than a log,
# a square root and a pair of trig calls. Module level, like the generator it
# comes from, and cleared by seed() so that seeding still resets the stream.
# declare() rather than an annotation: a module level annotation in pure Python
# mode leaves the name a Python global, so every draw would go through a dict
# lookup and a box. These are C variables.
_spare     = declare(cdouble, 0.0)
_haveSpare = declare(cint, 0)


# xoshiro256++, the generator every draw in this module comes from. Four words
# of state, process global the way the C library generator it replaced was, so
# seed() still seeds every digest at once.
#
# rand() cost 18ns a uniform here against this one's 1.6ns, and gave 31 bits
# where a double holds 53 -- so a draw was both slow and coarser than the type
# it was returned in.
_s0 = declare(uint64_t, 0)
_s1 = declare(uint64_t, 0)
_s2 = declare(uint64_t, 0)
_s3 = declare(uint64_t, 0)

# 2**64 / phi, SplitMix64's increment. Named because it does not fit in a
# signed 64 bit integer, and giving it a declared type keeps the arithmetic
# below in C rather than promoting it to Python.
_GOLDEN = declare(uint64_t, 0x9E3779B97F4A7C15)
_MIX_A = declare(uint64_t, 0xBF58476D1CE4E5B9)
_MIX_B = declare(uint64_t, 0x94D049BB133111EB)


@cfunc
@exceptval(check=False)
def _rotl(x: uint64_t, k: cint) -> uint64_t:
    """Rotate a 64 bit word left.

    Args:
        x (uint64_t): The word to rotate.
        k (int): How far, strictly between 0 and 64.

    Returns:
        uint64_t: The rotated word.
    """
    return (x << k) | (x >> (64 - k))


@cfunc
@exceptval(check=False)
def _splitMix(z: uint64_t) -> uint64_t:
    """One SplitMix64 output, used to expand a seed into xoshiro's state.

    xoshiro needs 256 bits of state that are not mostly zeros, and a caller
    passes one number. SplitMix64 is the mixer its authors recommend for
    exactly this: it scatters a small counter across the whole word, so a seed
    of 1 and a seed of 2 give unrelated states rather than adjacent ones.

    Args:
        z (uint64_t): The counter value to mix.

    Returns:
        uint64_t: The mixed word.
    """
    z = (z ^ (z >> 30)) * _MIX_A
    z = (z ^ (z >> 27)) * _MIX_B

    return z ^ (z >> 31)


@cfunc
@exceptval(check=False)
def _random() -> uint64_t:
    """Draw the next 64 bit word from xoshiro256++.

    Returns:
        uint64_t: A uniformly distributed word.
    """
    global _s0, _s1, _s2, _s3

    result: uint64_t = _rotl(_s0 + _s3, 23) + _s0
    t: uint64_t = _s1 << 17

    _s2 ^= _s0
    _s3 ^= _s1
    _s1 ^= _s2
    _s0 ^= _s3
    _s2 ^= t
    _s3 = _rotl(_s3, 45)

    return result


def seed(value):
    """Seed the random number generator that sample() draws from.

    The generator is global to the process, so this seeds every digest at once
    rather than any single instance.

    Args:
        value (int): Seed for the generator. Any integer: it is taken modulo
                   2**64, so a negative or very large one is accepted rather
                   than refused.
    """
    global _s0, _s1, _s2, _s3, _haveSpare

    z: uint64_t = cast(uint64_t, value & 0xFFFFFFFFFFFFFFFF)

    z = z + _GOLDEN
    _s0 = _splitMix(z)
    z = z + _GOLDEN
    _s1 = _splitMix(z)
    z = z + _GOLDEN
    _s2 = _splitMix(z)
    z = z + _GOLDEN
    _s3 = _splitMix(z)

    # xoshiro has no way out of an all zero state. SplitMix64 never produces
    # four zeros in a row, but the generator is silently useless if it ever
    # does, which is worth one comparison at seeding time to rule out.
    if (_s0 | _s1 | _s2 | _s3) == 0:
        _s0 = 1

    # Otherwise the first normal draw after seeding would be the one left over
    # from before it, and seeding would not fully determine the sequence.
    _haveSpare = 0



# The generator starts from a fixed state, so an unseeded process is
# reproducible: the C library generator this replaced behaved as though seeded
# with 1, and this keeps that.
seed(1)

@cclass
class RanVar():
    """
    Implementation of the t-digest algorithm for computing quantiles as class RanVar

    The t-digest algorithm is described in 'Computing Extremely Accurate Quantiles Using t-Digests'
    by Ted Dunning. This data structure provides approximate quantile computation with bounded
    memory usage and high accuracy, particularly at the tails of the distribution.

    The digest maintains a set of centroids (weighted points) that summarize the
    distribution, at most maxBins of them.

    Points are not merged in as they arrive. They wait in a buffer, and when it
    fills, the whole batch is sorted once and merged with the centroids in a
    single pass -- which is why adding a point costs a store rather than a walk
    of the centroids, and why the cost of adding one barely grows with maxBins.
    Every read folds the buffer in first, so nothing outside the class can
    observe it; the one thing that follows from it is that reading a digest is
    part of how its centroids come out, so the same points added in the same
    order but read at different times give centroids that are close rather than
    identical.

    Which centroids survive a merge is decided by Dunning's K_1 scale function:
    the weight is cut into maxBins buckets of equal k, which are narrow at the
    tails and wide in the middle, so the digest keeps its resolution where a
    quantile estimate needs it. The smallest and largest points are always kept
    whole, so lower() and upper() are exactly the minimum and maximum of
    everything added.

    quantile() walks a ladder of cumulative weights, one rung per pair of
    adjacent centroids. That ladder only changes when the centroids do, so it is
    cached and searched rather than rebuilt on every call, which is what makes
    repeated sampling from a fitted digest cheap.

    Attributes:
        bins (np.ndarray): Array storing centroid values (x-coordinates)
        cnts (np.ndarray): Array storing centroid weights (counts)
        cumw (np.ndarray): Cached cumulative weight at the start of each segment
        gaps (np.ndarray): Cached weight spanned by each segment
        buf (np.ndarray): Points waiting to be merged in
        bufw (np.ndarray): Their weights, aligned with buf
        maxBins (int): Maximum number of centroids to maintain
        nActive (int): Current number of active centroids
        nBuf (int): How many points are waiting in the buffer

    Example:
        >>> ranvar = RanVar(maxBins=100)
        >>> ranvar.fit([1, 2, 3, 4, 5])
        >>> digest.quantile(0.5)  # Median
        3.0
    """    
    _bins: cdouble[::1]
    _cnts: cdouble[::1]
    _cumw: cdouble[::1]
    _gaps: cdouble[::1]

    bins: np.ndarray
    cnts: np.ndarray
    cumw: np.ndarray
    gaps: np.ndarray

    # Points wait here until there are enough of them to be worth merging.
    # _tb/_tc are the merge sort's scratch space, nothing more.
    _buf: cdouble[::1]
    _bufw: cdouble[::1]
    _tb: cdouble[::1]
    _tc: cdouble[::1]

    # The quantile each of the maxBins buckets closes at. See _allocate().
    _bnds: cdouble[::1]

    buf: np.ndarray
    bufw: np.ndarray
    tb: np.ndarray
    tc: np.ndarray
    bnds: np.ndarray

    maxBins: cint
    nActive: cint

    bufCap: cint
    nBuf: cint

    # Total weight as _sumWeights() would compute it, cached alongside the
    # ladder, and a flag saying whether either still reflects the centroids.
    _total: cdouble
    _stale: cint

    # Whether the four buffers above have been allocated yet. See _allocate().
    _ready: cint

    def __init__(self, maxBins=32):
        """Initialize a new t-digest.

        Args:
            maxBins (int, optional): Maximum number of centroids to maintain.
                                   Must be at least 3. Defaults to 32.

        Raises:
            ValueError: If maxBins is less than 3.
        """
        # The merge step reserves the first and last centroid for the tails, so
        # it needs an interior pair left to merge. An overflowing digest holds
        # maxBins + 1 centroids, which leaves _findMinimumDifference() an empty
        # range to scan below three bins: it would return its -1 sentinel and
        # _add() would index one slot before the arrays.
        if maxBins < 3:
            raise ValueError(
                f'maxBins must be at least 3, got {maxBins}: a digest needs an '
                f'interior pair to merge once it is full, on top of the first '
                f'and last centroid it keeps for the tails'
            )

        self.maxBins = maxBins
        self.nActive = 0

        # Two buffers' worth per bin ceiling. Larger amortises the merge over
        # more points but makes each one longer to sort; measured flat from
        # roughly two to ten, so this is the small end of the plateau. The
        # floor keeps a tiny digest from flushing every few points.
        self.bufCap = maxBins * 2 if maxBins * 2 > 32 else 32
        self.nBuf   = 0

        # The buffers are not allocated here. Four numpy arrays cost more than
        # everything else about constructing a digest put together, and a good
        # many digests never need them: Normal, NegBinom and Constant draw from
        # parameters and never touch a centroid, and asRanVar() wraps every
        # plain number a caller passes in a Constant. _allocate() puts them in
        # place the first time something actually reads or writes one.
        self.bins = None
        self.cnts = None
        self.cumw = None
        self.gaps = None
        self.buf  = None
        self.bufw = None
        self.tb   = None
        self.tc   = None
        self.bnds = None

        self._total = 0
        self._stale = 1
        self._ready = 0

    @cfunc
    @final
    def _allocate(self) -> cvoid:
        """Put the centroid buffers in place, if they are not there already.

        Called from wherever one of them is about to be read or written, which
        is _add() and _rebuildLadder() on the hot paths and a handful of
        accessors elsewhere. Idempotent, and cheap enough to call every time:
        it is a flag test once the buffers exist.

        The centroid arrays carry a whole buffer beyond the bin ceiling. A
        flush merges the buffer into them before compressing back under it, and
        merging in place from the top is what saves a second pair of arrays to
        merge into.
        """
        if self._ready:
            return

        self.bins = np.zeros(self.maxBins + self.bufCap + 1, dtype=np.float64)
        self.cnts = np.zeros(self.maxBins + self.bufCap + 1, dtype=np.float64)
        self.cumw = np.zeros(self.maxBins + 1, dtype=np.float64)
        self.gaps = np.zeros(self.maxBins + 1, dtype=np.float64)

        self.buf  = np.zeros(self.bufCap, dtype=np.float64)
        self.bufw = np.zeros(self.bufCap, dtype=np.float64)
        self.tb   = np.zeros(self.bufCap, dtype=np.float64)
        self.tc   = np.zeros(self.bufCap, dtype=np.float64)

        # Dunning's K_1 scale function, k(q) = asin(2q - 1), cut into maxBins
        # equal steps of k and inverted back into quantiles. Steep at both ends,
        # so the buckets are narrow at the tails and wide in the middle: that is
        # what puts the digest's resolution where a quantile estimate needs it.
        # The last entry is exactly 1, so the last bucket takes whatever is left.
        # Fixed by maxBins, so built once here rather than recomputed per flush.
        steps     = np.arange(1, self.maxBins + 1, dtype=np.float64)
        self.bnds = (np.sin(-np.pi/2 + steps * np.pi / self.maxBins) + 1.0) * 0.5
        self.bnds[self.maxBins - 1] = 1.0

        self._bins = self.bins
        self._cnts = self.cnts
        self._cumw = self.cumw
        self._gaps = self.gaps

        self._buf  = self.buf
        self._bufw = self.bufw
        self._tb   = self.tb
        self._tc   = self.tc
        self._bnds = self.bnds

        self._ready = 1

    @cfunc
    @final
    @boundscheck(False)
    @wraparound(False)
    @initializedcheck(False)
    def _sortBuffer(self) -> cvoid:
        """Sort the pending buffer by value, carrying the weights with it.

        A bottom-up merge sort, ping-ponging between the buffer and the scratch
        pair and copying back only if it finishes on the wrong side. Stable, so
        equal values keep the order they arrived in.
        """
        n: cint = self.nBuf

        srcV: p_double = address(self._buf[0])
        srcW: p_double = address(self._bufw[0])
        dstV: p_double = address(self._tb[0])
        dstW: p_double = address(self._tc[0])
        swap: p_double

        width: cint = 1
        i: cint
        l: cint
        r: cint
        lEnd: cint
        rEnd: cint
        k: cint

        while width < n:
            i = 0

            while i < n:
                l    = i
                lEnd = i + width
                rEnd = i + 2*width

                if lEnd > n:
                    lEnd = n

                if rEnd > n:
                    rEnd = n

                r = lEnd
                k = i

                while (l < lEnd) and (r < rEnd):
                    if srcV[l] <= srcV[r]:
                        dstV[k] = srcV[l]
                        dstW[k] = srcW[l]
                        l += 1
                    else:
                        dstV[k] = srcV[r]
                        dstW[k] = srcW[r]
                        r += 1

                    k += 1

                while l < lEnd:
                    dstV[k] = srcV[l]
                    dstW[k] = srcW[l]
                    l += 1
                    k += 1

                while r < rEnd:
                    dstV[k] = srcV[r]
                    dstW[k] = srcW[r]
                    r += 1
                    k += 1

                i += 2*width

            swap = srcV; srcV = dstV; dstV = swap
            swap = srcW; srcW = dstW; dstW = swap

            width *= 2

        if srcV != address(self._buf[0]):
            memcpy(address(self._buf[0]),  srcV, n * sizeof(cdouble))
            memcpy(address(self._bufw[0]), srcW, n * sizeof(cdouble))

    @cfunc
    @final
    @boundscheck(False)
    @wraparound(False)
    @cdivision(True)
    @initializedcheck(False)
    def _compress(self, m: cint) -> cvoid:
        """Reduce m sorted weighted points, held in the centroid arrays, to at
        most maxBins centroids.

        One pass. Each point is placed by the quantile of its own midpoint, and
        stays in the centroid being built while that quantile is inside the
        current bucket; the first point past the bucket closes it and opens the
        next. Since a centroid is only opened when a bucket is, and there are
        maxBins buckets, the result cannot exceed the ceiling.

        The lowest and highest points are kept as centroids of their own, so
        lower() and upper() stay the exact minimum and maximum of everything
        added, as they were when centroids were merged one pair at a time.

        Writing is always behind reading -- the centroid being closed sits at
        an index below the point being read -- so this compresses in place.

        Args:
            m (int): How many sorted points the arrays hold.
        """
        bins: p_double = address(self._bins[0])
        cnts: p_double = address(self._cnts[0])
        bnds: p_double = address(self._bnds[0])

        i: cint
        j: cint
        out: cint
        w: cint
        r: cint

        total: cdouble
        before: cdouble
        q: cdouble
        qEnd: cdouble
        sumX: cdouble
        sumW: cdouble

        if m <= 2:
            self.nActive = m

            return

        total = 0
        for i in range(m):
            total = total + cnts[i]

        # The minimum, kept whole.
        out    = 0
        before = cnts[0]

        # The first interior point opens the second centroid, in whichever
        # bucket its own quantile falls.
        out  = 1
        sumX = bins[1] * cnts[1]
        sumW = cnts[1]

        j    = 0
        qEnd = bnds[0]
        q    = (before + cnts[1] * 0.5) / total

        while (q > qEnd) and (j < self.maxBins - 1):
            j += 1
            qEnd = bnds[j]

        before = before + cnts[1]

        for i in range(2, m - 1):
            q = (before + cnts[i] * 0.5) / total

            # The ceiling is enforced here rather than trusted to the bucket
            # count: a quantile that rounds a hair past 1 would otherwise open
            # a centroid the arrays have no room for. One slot is left for the
            # maximum below.
            if (q <= qEnd) or (out >= self.maxBins - 2):
                sumX = sumX + bins[i] * cnts[i]
                sumW = sumW + cnts[i]
            else:
                bins[out] = sumX / sumW
                cnts[out] = sumW

                out += 1

                while (q > qEnd) and (j < self.maxBins - 1):
                    j += 1
                    qEnd = bnds[j]

                sumX = bins[i] * cnts[i]
                sumW = cnts[i]

            before = before + cnts[i]

        bins[out] = sumX / sumW
        cnts[out] = sumW

        # The maximum, kept whole.
        out += 1
        bins[out] = bins[m-1]
        cnts[out] = cnts[m-1]

        self.nActive = out + 1

        # Points that landed in different buckets can still share a value --
        # a digest fed one number many times over is the plain case -- and
        # centroids sitting on top of each other span no interval for a
        # quantile to interpolate across, so they are folded together.
        w = 0
        for r in range(1, self.nActive):
            if bins[r] == bins[w]:
                cnts[w] = cnts[w] + cnts[r]
            else:
                w += 1
                bins[w] = bins[r]
                cnts[w] = cnts[r]

        self.nActive = w + 1

    @cfunc
    @final
    @boundscheck(False)
    @wraparound(False)
    @initializedcheck(False)
    def _flush(self) -> cvoid:
        """Fold the pending buffer into the centroids.

        Sorts the buffer, merges it with the centroids -- both runs are sorted,
        so that is one linear pass, walked from the top because the centroid
        arrays have the buffer's worth of room at the end -- and compresses the
        result back under the bin ceiling.

        This is where the digest does its work. Buffering is what makes it
        worth doing: the cost of a flush is shared across a whole buffer's
        worth of points, where inserting each point into the centroids as it
        arrived cost a walk of them per point.
        """
        if self.nBuf == 0:
            return

        self._sortBuffer()

        bins: p_double = address(self._bins[0])
        cnts: p_double = address(self._cnts[0])
        buf: p_double  = address(self._buf[0])
        bufw: p_double = address(self._bufw[0])

        i: cint = self.nActive - 1
        k: cint = self.nBuf - 1
        w: cint = self.nActive + self.nBuf - 1

        m: cint = w + 1

        # Merged from the top down, so the write head stays above both read
        # heads and nothing is overwritten before it is read.
        while (i >= 0) and (k >= 0):
            if bins[i] > buf[k]:
                bins[w] = bins[i]
                cnts[w] = cnts[i]
                i -= 1
            else:
                bins[w] = buf[k]
                cnts[w] = bufw[k]
                k -= 1

            w -= 1

        while k >= 0:
            bins[w] = buf[k]
            cnts[w] = bufw[k]
            k -= 1
            w -= 1

        # Whatever centroids are left already sit where they belong.

        self.nBuf = 0

        self._compress(m)

        self._stale = 1

    @cfunc
    @boundscheck(False)
    @wraparound(False)
    @initializedcheck(False)
    def _add(self, point:cdouble, count:cdouble) -> cvoid:
        """Add a weighted point to the digest (internal implementation).

        The point is written to a buffer rather than into the centroids. When
        the buffer fills, _flush() sorts it and merges the whole batch in one
        pass. That is what a simulation's inner loop calls, once per sample, so
        it is kept to a bounds-free store and a counter.

        Every read of the digest flushes first, so nothing outside can observe
        the buffer.

        Args:
            point (float): The value to add.
            count (float): The weight/count of the value.
        """
        if not self._ready:
            self._allocate()

        if self.nBuf == self.bufCap:
            self._flush()

        self._buf[self.nBuf]  = point
        self._bufw[self.nBuf] = count

        self.nBuf += 1

        self._stale = 1


    @cfunc
    @final
    @boundscheck(False)
    @cdivision(True)
    @wraparound(False)
    @initializedcheck(False)
    def _lower(self) -> cdouble:
        """Get the minimum value in the digest.

        Returns:
            float: The smallest centroid value.

        Raises:
            ValueError: If the digest is empty.
        """
        if self.nActive == 0:
            raise ValueError('cannot take the lower bound of an empty digest')

        return self._bins[0]

    @cfunc
    @final
    @boundscheck(False)
    @cdivision(True)
    @wraparound(False)
    @initializedcheck(False)
    def _upper(self) -> cdouble:
        """Get the maximum value in the digest.

        Returns:
            float: The largest centroid value.

        Raises:
            ValueError: If the digest is empty. The index below is nActive-1,
                      which reads off the front of the buffer when nothing has
                      been added, and bounds checking is off.
        """
        if self.nActive == 0:
            raise ValueError('cannot take the upper bound of an empty digest')

        return self._bins[self.nActive - 1]

    @cfunc
    @final
    @boundscheck(False)
    @wraparound(False)
    @initializedcheck(False)
    def _sumWeights(self) -> cdouble:
        """Calculate the total weight of all active centroids.

        Returns:
            float: Sum of all centroid weights.
        """
        som: cdouble = 0
        i: cint

        for i in range(self.nActive):
            som = som + self._cnts[i]

        return som    
    
    @cfunc
    @final
    @boundscheck(False)
    @wraparound(False)
    @cdivision(True)
    @initializedcheck(False)
    def _interpolationBounds(self, i: cint, som: cdouble,
                             yi: p_double, yi_n: p_double) -> cvoid:
        """Cumulative weights bracketing the segment between centroid i and i+1.

        We use the approach of Dunning here to improve interpolation when we have
        single weighted points: a centroid carrying a weight of one sits at an
        exact observation, so the segment interpolates from the centroid itself
        rather than from its midpoint.

        The two results are written through pointers rather than returned as a
        pair. Cython types an unannotated return as an object, so returning them
        built and threw away a Python tuple on every call, which is once per
        cdf() or dcdf().

        Args:
            i (int): Index of the left centroid of the segment.
            som (float): Total weight of all centroids before i.
            yi (double*): Receives the cumulative weight at c[i].
            yi_n (double*): Receives the cumulative weight at c[i+1].
        """
        m: p_double = address(self._cnts[0])

        if (m[i] > 1) & (m[i+1] > 1):
            # Case I: Both points greater than one, normal interpolation.
            yi[0]   = som + m[i]/2
            yi_n[0] = yi[0] + (m[i+1] + m[i]) / 2

        elif (m[i] == 1) & (m[i+1] > 1):
            # Case II: Left point is a single observation.
            yi[0]   = som
            yi_n[0] = yi[0] + (m[i+1]) / 2

        elif (m[i] > 1) & (m[i+1] == 1):
            # Case III: Right point is a single observation.
            yi[0]   = som + m[i]/2
            yi_n[0] = yi[0] + (m[i]) / 2
        else:
            # Case IV: Both are single observations, nothing to interpolate over.
            yi[0]   = som
            yi_n[0] = yi[0]

    @cfunc
    @final
    @boundscheck(False)
    @wraparound(False)
    @cdivision(True)
    @initializedcheck(False)
    def _rebuildLadder(self) -> cvoid:
        """Recompute the cumulative weight ladder quantile() searches.

        Rung i is the cumulative weight at centroid i, and gap i is the weight
        the segment from centroid i to i+1 spans. The gaps sum to the total
        weight, which is what lets a probability be located in the ladder.

        The total is summed in the same order as _sumWeights() so the two agree
        to the last bit.
        """
        i: cint
        som: cdouble
        wGap: cdouble

        if not self._ready:
            self._allocate()

        m: p_double = address(self._cnts[0])
        cumw: p_double = address(self._cumw[0])
        gaps: p_double = address(self._gaps[0])

        som = 0
        for i in range(self.nActive):
            som = som + m[i]

        self._total = som

        som = 0
        cumw[0] = 0

        for i in range(self.nActive - 1):
            # An outer centroid contributes its full weight to its only segment,
            # an interior one half to each side. The two ends are tested
            # separately so the gaps still sum to the total when there are only
            # two centroids and segment 0 is both the first and the last.
            if i == 0:
                wGap = m[i]
            else:
                wGap = m[i]/2

            if i == self.nActive - 2:
                wGap = wGap + m[i+1]
            else:
                wGap = wGap + m[i+1]/2

            gaps[i] = wGap

            som = som + wGap
            cumw[i+1] = som

        self._stale = 0

    # Public API. --------------------------------------------------------------

    def add(self, point: cdouble, count: cdouble = 1.0):
        """Add a single weighted observation to the digest.

        Both arguments are annotated, so Cython converts them straight to
        doubles rather than parsing them as objects and unboxing them inside.

        Args:
            point (float): The value to add.
            count (float, optional): The weight of the value. Defaults to 1.0.
        """
        self._add(point, count)

    def getActiveBinCount(self):
        """Get the number of centroids currently in use.

        Returns:
            int: Count of active centroids, at most maxBins.
        """
        self._flush()

        return self.nActive

    def getBins(self):
        """Get the active centroid values.

        Returns:
            np.ndarray: View of the centroid values, in ascending order.
        """
        self._flush()

        if not self._ready:
            return np.zeros(0, dtype=np.float64)

        return self.bins[:self.nActive]

    def getWeights(self):
        """Get the active centroid weights.

        Returns:
            np.ndarray: View of the centroid weights, aligned with getBins().
        """
        self._flush()

        if not self._ready:
            return np.zeros(0, dtype=np.float64)

        return self.cnts[:self.nActive]

    def lower(self):
        """Get the minimum value in the digest.

        Returns:
            float: The smallest centroid value.

        Raises:
            ValueError: If the digest is empty.
        """
        self._flush()

        return self._lower()

    def upper(self):
        """Get the maximum value in the digest.

        Returns:
            float: The largest centroid value.

        Raises:
            ValueError: If the digest is empty.
        """
        self._flush()

        return self._upper()

    @boundscheck(False)
    @wraparound(False)
    @cdivision(True)
    @initializedcheck(False)
    def cdf(self, k: cdouble):
        """Compute the cumulative distribution function at a given point.

        Implements the CDF estimation algorithm from Ted Dunning's paper
        'Computing Extremely Accurate Quantiles Using t-Digests'. Uses
        different interpolation strategies depending on whether centroids
        represent single points or aggregated ranges.

        The total weight comes from the cache the ladder keeps rather than from
        _sumWeights(), which walked every centroid again on each call, on top of
        the walk that finds the segment. _rebuildLadder() accumulates it in the
        same order _sumWeights() did, so it is the same total to the last bit.

        Args:
            k (float): The point at which to evaluate the CDF.

        Returns:
            float: Estimated CDF value between 0 and 1.
        """
        som:cdouble = 0
        i:cint
        yi:cdouble
        yi_n:cdouble
        g:cdouble

        self._flush()

        if k <= self._lower():
            return 0.
        elif k >= self._upper():
            return 1.

        c: p_double = address(self._bins[0])
        m: p_double = address(self._cnts[0])

        if self._stale:
            self._rebuildLadder()

        # k is strictly inside the support, so one of these segments holds it
        # and the loop always returns. The last centroid starts no segment,
        # which is why this stops one short of it rather than reading the slot
        # past the live prefix the way it used to.
        for i in range(self.nActive - 1):
            if c[i] <= k < c[i+1]:
                self._interpolationBounds(i, som, address(yi), address(yi_n))

                g = (yi_n - yi) / (c[i+1] - c[i])

                return (g*(k - c[i]) + yi) / self._total

            else:
                som += m[i]

        return None

    def ccdf(self, x):
        """Compute the complementary cumulative distribution function at a point.

        Args:
            x (float): The point at which to evaluate the CCDF.

        Returns:
            float: 1 minus the CDF at x.

        Raises:
            ValueError: If the digest is empty.
        """
        return 1 - self.cdf(x)

    @boundscheck(False)
    @wraparound(False)
    @cdivision(True)
    @initializedcheck(False)
    def dcdf(self, k: cdouble):
        """Compute the derivative of the CDF at a given point.

        The CDF is piecewise linear between centroids, so its derivative is the
        constant gradient of the segment containing k, normalised by the total
        weight. This is the density estimate at k. Outside the support the CDF is
        flat, so the derivative is zero.

        Takes the total from the ladder's cache rather than recomputing it, for
        the same reason cdf() does.

        Args:
            k (float): The point at which to evaluate the derivative.

        Returns:
            float: Estimated density at k.
        """
        som:cdouble = 0
        i:cint
        yi:cdouble
        yi_n:cdouble
        g:cdouble

        self._flush()

        if (k <= self._lower()) or (k >= self._upper()):
            return 0.

        c: p_double = address(self._bins[0])
        m: p_double = address(self._cnts[0])

        if self._stale:
            self._rebuildLadder()

        for i in range(self.nActive - 1):
            if c[i] <= k < c[i+1]:
                self._interpolationBounds(i, som, address(yi), address(yi_n))

                g = (yi_n - yi) / (c[i+1] - c[i])

                return g / self._total

            else:
                som += m[i]

        return 0.

    def dccdf(self, k):
        """Compute the derivative of the complementary CDF at a given point.

        Args:
            k (float): The point at which to evaluate the derivative.

        Returns:
            float: The negated density at k.
        """
        return -self.dcdf(k)

    @ccall
    @boundscheck(False)
    @wraparound(False)
    @cdivision(True)
    @initializedcheck(False)
    def quantile(self, p: cdouble) -> cdouble:
        """Compute the quantile for a given probability.

        Locates the probability in the cumulative weight ladder by binary
        search, then interpolates linearly between the two centroids of the
        segment it lands in. The ladder is rebuilt only when a point has been
        added since it was last built, so drawing repeatedly from a digest that
        is no longer changing costs a search rather than a scan.

        Args:
            p (float): Probability value between 0 and 1. Values outside that
                     range clamp to the ends of the support.

        Returns:
            float: Estimated quantile value.

        Raises:
            ValueError: If the digest is empty.
        """
        W: cdouble
        w_: cdouble
        fraction: cdouble
        lo: cint
        hi: cint
        mid: cint

        # Tested here rather than left to _flush(): sampling calls this once
        # per draw with nothing pending, and the test is a field read where the
        # call is not always one the C compiler inlines away.
        if self.nBuf != 0:
            self._flush()

        if p <= 0:
            return self._lower()
        elif p >= 1:
            return self._upper()

        if self._stale:
            self._rebuildLadder()

        c   = self._bins
        cum = self._cumw

        W  = self._total
        w_ = p*W

        # The rungs are non decreasing, so the segment holding w_ is the last
        # one starting at or below it. Searching for the last rather than the
        # first also steps over any segment of zero weight, which has no
        # interior to interpolate across.
        lo = 0
        hi = self.nActive - 1

        while lo < hi:
            mid = (lo + hi + 1) // 2

            if cum[mid] <= w_:
                lo = mid
            else:
                hi = mid - 1

        # Past the last segment: one centroid only, or p*W rounded up to the
        # total. Either way the answer is the top of the support.
        if lo > self.nActive - 2:
            return self._upper()

        fraction = (w_ - cum[lo]) / self._gaps[lo]

        return fraction * (c[lo+1] - c[lo]) + c[lo]


    @ccall
    @boundscheck(False)
    @wraparound(False)
    @cdivision(True)
    @initializedcheck(False)
    def sample(self) -> cdouble:
        """Sample a single value from the distribution represented by the digest.

        Draws a uniform probability and returns the quantile at that
        probability, so the draw follows the distribution the digest models.
        Use seed() to make a sequence of draws reproducible.

        Returns:
            float: A value drawn from the distribution.
        """

        p: cdouble = _uniform01()

        return self.quantile(p)


    def mean(self):
        """Compute the mean of the distribution.

        Merging two centroids replaces them with their weighted average, which
        leaves the total weighted sum untouched, so this is the exact mean of
        the data added to the digest rather than an approximation of it.

        Returns:
            float: The weighted mean of the centroids.

        Raises:
            ValueError: If the digest is empty.
        """
        som: cdouble = 0
        i: cint

        self._flush()

        if self.nActive == 0:
            raise ValueError('mean() is undefined for an empty digest')

        for i in range(self.nActive):
            som = som + self._bins[i]*self._cnts[i]

        return som / self._sumWeights()

    @cfunc
    @final
    @boundscheck(False)
    @wraparound(False)
    @initializedcheck(False)
    def _fitBuffer(self, data: cdouble[::1]) -> cvoid:
        """Add every value in a contiguous buffer, without leaving C.

        Args:
            data (double[::1]): The values to add, each with weight 1.
        """
        i: cint

        for i in range(data.shape[0]):
            self._add(data[i], 1.0)

    def fit(self, x):
        """Fit the digest to a collection of data points.

        This is a convenience method that adds all points in the collection
        to the digest with equal weight.

        A numeric numpy array goes through a C loop over its buffer. Iterating
        it in Python instead would hand back a fresh numpy scalar per element
        for _add() to unbox, which costs more per point than a stored float
        does -- so the array, the obvious thing to fit, used to be the slower
        thing to fit. Everything else still iterates, so any sequence,
        generator or iterator works as it did.

        Args:
            x (iterable): Collection of numeric values to add to the digest.
        """
        # Real numbers only: a complex, object or datetime array has no
        # meaningful double buffer, and falling through leaves it raising out
        # of _add() the way it always has.
        if isinstance(x, np.ndarray) and x.ndim == 1 and x.dtype.kind in 'fiub':
            self._fitBuffer(np.ascontiguousarray(x, dtype=np.float64))

            return

        for xx in x:
            self._add(xx, 1.0)


    def toDict(self):
        """Serialise this digest to plain JSON types.

        A t-digest is fully described by its centroid positions, their
        weights, the bin ceiling and how many centroids are live -- the same
        values __reduce__ hands the unpickler -- so this is a faithful
        representation rather than a summary of one.

        The payload names the class that wrote it, which is what lets
        fromDict() rebuild a subclass rather than a bare RanVar. Subclasses
        that carry parameters instead of centroids override this, but they
        keep the type key.

        Returns:
            dict: The digest, as JSON types only.
        """
        # Before nActive is read, not after: getBins() below would flush the
        # buffer and hand back centroids the count no longer described.
        self._flush()

        return {
            'type':    type(self).__name__,
            'maxBins': self.maxBins,
            'nActive': self.nActive,
            'bins':    self.getBins().tolist(),
            'weights': self.getWeights().tolist(),
        }

    @classmethod
    def fromDict(cls, data):
        """Rebuild whatever wrote a payload, as the class that wrote it.

        Dispatches on the payload's type key: a Normal comes back a Normal,
        a NegBinom a NegBinom, and so on for any RanVar subclass, including
        one defined outside this module. Reached through a subclass
        (Normal.fromDict(...)) it additionally refuses a payload written by
        something that is not that subclass, rather than quietly returning
        another type.

        Args:
            data (dict): A payload from toDict().

        Returns:
            RanVar: The reconstructed distribution.

        Raises:
            ValueError: If the payload has no type key, names a class that is
                      not a RanVar (or not a cls, when reached through a
                      subclass), or is missing a field that class needs.
        """
        if not isinstance(data, dict) or 'type' not in data:
            raise ValueError(
                "a RanVar payload needs a 'type' key naming the class that "
                "wrote it, as toDict() writes one"
            )

        name   = data['type']
        target = RanVar._named(name)

        if target is None:
            raise ValueError(f'{name!r} is not a known RanVar class')

        if not issubclass(target, cls):
            raise ValueError(
                f'{name!r} is not a {cls.__name__}, so {cls.__name__}.'
                f'fromDict() will not rebuild it. Use RanVar.fromDict() to '
                f'accept any of them'
            )

        try:
            return target._fromDict(data)
        except KeyError as missing:
            raise ValueError(
                f'{name} payload is missing {missing.args[0]!r}'
            ) from missing

    @classmethod
    def _named(cls, name):
        """Find the RanVar subclass a payload's type key names.

        Walks the live subclass tree rather than a registry, so a subclass
        defined outside this module is found as well, with no registration
        step for a caller to forget.

        Args:
            name (str): The class name to look for.

        Returns:
            type: The class, or None if no RanVar goes by that name.
        """
        pending = [RanVar]

        while pending:
            found = pending.pop()

            if found.__name__ == name:
                return found

            pending.extend(found.__subclasses__())

        return None

    @classmethod
    def _fromDict(cls, data):
        """Rebuild a digest from its own payload.

        The hook fromDict() calls once it has resolved the class. Every
        subclass that overrides toDict() overrides this to match, which is
        what keeps the dispatch in one place rather than in every class.

        Args:
            data (dict): A payload from toDict().

        Returns:
            RanVar: The reconstructed digest.

        Raises:
            ValueError: If the payload's centroids do not describe a digest
                      of the size it claims.
        """
        maxBins = int(data['maxBins'])
        nActive = int(data['nActive'])
        bins    = data['bins']
        weights = data['weights']

        if len(bins) != nActive or len(weights) != nActive:
            raise ValueError(
                f'payload claims {nActive} centroids but carries '
                f'{len(bins)} of them and {len(weights)} weights'
            )

        if nActive > maxBins:
            raise ValueError(
                f'payload claims {nActive} centroids, more than the {maxBins} '
                f'its bin ceiling allows'
            )

        digest = cls(maxBins=maxBins)

        # The backing arrays are one longer than the ceiling (_add() writes
        # into the spare slot before merging), and only the live prefix is
        # serialised, so the rest is left at the zeros __init__ made.
        centroids = np.zeros(maxBins + 1, dtype=np.float64)
        counts    = np.zeros(maxBins + 1, dtype=np.float64)

        centroids[:nActive] = bins
        counts[:nActive]    = weights

        digest.__setstate__((nActive, centroids, counts))

        return digest

    def __invert__(self):
        """Syntactic sugar for the sample method.

        Returns:
            float: A value drawn from the distribution, as sample() would give.

        Raises:
            ValueError: If the digest is empty.
        """
        return self.sample()

    def _state(self):
        """The centroid state __reduce__ hands the unpickler.

        Every subclass's __reduce__ carries the same triple, so allocating the
        buffers a lazily built digest has not needed yet happens here rather
        than in each of them.

        Returns:
            tuple: The (nActive, bins, cnts) triple __setstate__ expects.
        """
        self._flush()
        self._allocate()

        # Only the live prefix: the arrays carry a buffer's worth of slack
        # past it that means nothing to a reader, and __setstate__ sizes what
        # it restores into itself.
        return (self.nActive, self.getBins(), self.getWeights())

    def __reduce__(self):
        """Support pickling and copying.

        Extension types holding memoryviews get no default __reduce__, so we
        serialise the backing arrays and rebind the views in __setstate__.

        Returns:
            tuple: The (callable, args, state) triple pickle expects.
        """
        return (type(self), (self.maxBins,), self._state())

    def __setstate__(self, state):
        """Restore the state produced by __reduce__.

        Allocates outright rather than through _allocate(): the centroids
        arrive here, so a restored digest has buffers whatever the constructor
        left behind. The ladder is not serialised, so it is sized to match and
        left at zeros for the first query to rebuild.

        Args:
            state (tuple): The (nActive, bins, cnts) triple from __reduce__.
        """
        self.nActive = state[0]
        self.nBuf    = 0

        # Sized as _allocate() would, not as the payload happens to be: a
        # restored digest has to have the same room to merge into that a fresh
        # one does, whatever length of prefix it was handed.
        self.bins = np.zeros(self.maxBins + self.bufCap + 1, dtype=np.float64)
        self.cnts = np.zeros(self.maxBins + self.bufCap + 1, dtype=np.float64)
        self.cumw = np.zeros(self.maxBins + 1, dtype=np.float64)
        self.gaps = np.zeros(self.maxBins + 1, dtype=np.float64)

        self.bins[:self.nActive] = np.asarray(state[1], dtype=np.float64)[:self.nActive]
        self.cnts[:self.nActive] = np.asarray(state[2], dtype=np.float64)[:self.nActive]

        self.buf  = np.zeros(self.bufCap, dtype=np.float64)
        self.bufw = np.zeros(self.bufCap, dtype=np.float64)
        self.tb   = np.zeros(self.bufCap, dtype=np.float64)
        self.tc   = np.zeros(self.bufCap, dtype=np.float64)

        # Dunning's K_1 scale function, k(q) = asin(2q - 1), cut into maxBins
        # equal steps of k and inverted back into quantiles. Steep at both ends,
        # so the buckets are narrow at the tails and wide in the middle: that is
        # what puts the digest's resolution where a quantile estimate needs it.
        # The last entry is exactly 1, so the last bucket takes whatever is left.
        # Fixed by maxBins, so built once here rather than recomputed per flush.
        steps     = np.arange(1, self.maxBins + 1, dtype=np.float64)
        self.bnds = (np.sin(-np.pi/2 + steps * np.pi / self.maxBins) + 1.0) * 0.5
        self.bnds[self.maxBins - 1] = 1.0

        self._bins = self.bins
        self._cnts = self.cnts
        self._cumw = self.cumw
        self._gaps = self.gaps

        self._buf  = self.buf
        self._bufw = self.bufw
        self._tb   = self.tb
        self._tc   = self.tc
        self._bnds = self.bnds

        self._ready = 1

        # __init__ has already marked it stale, since __reduce__ rebuilds through
        # the constructor. Repeated here so the rule that any change to the
        # centroids invalidates the ladder holds wherever they are assigned.
        self._stale = 1


# Sampling helpers for the parametric subclasses. ------------------------------
#
# Free functions rather than methods, since Normal and NegBinom both need a
# standard normal draw (NegBinom by way of its gamma sampler) and there is
# nothing instance specific about generating one. All draw from the same
# process-global C generator seed() controls, consistent with RanVar.sample().

@cfunc
@exceptval(check=False)
@cdivision(True)
def _uniform() -> cdouble:
    """Draw from the open interval (0, 1), never exactly 0 or 1.

    Box-Muller and the gamma sampler below both take a log of this, so the
    closed end at 0 that _uniform01() can in principle return would blow up to
    -inf. It comes up about once in every 2**53 draws.

    Returns:
        float: A uniform draw strictly between 0 and 1.
    """
    u: cdouble = _uniform01()

    while u <= 0.0:
        u = _uniform01()

    return u


@cfunc
@exceptval(check=False)
@cdivision(True)
def _uniform01() -> cdouble:
    """Draw from [0, 1), with a full 53 bits of resolution.

    The top 53 bits of a generator word scaled by 2**-53, which is the
    standard way to fill a double: every representable value in [0, 1) with
    that spacing is reachable, and 1 is not.

    Returns:
        float: A uniform draw in [0, 1).
    """
    return cast(cdouble, _random() >> 11) * (1.0 / 9007199254740992.0)


@cfunc
@exceptval(check=False)
@cdivision(True)
def _standardNormal() -> cdouble:
    """Draw one N(0, 1) value using the Box-Muller transform.

    The transform yields two independent variates per pair of uniforms, at the
    cost of one logarithm, one square root and one pair of trig calls. The
    second is held rather than discarded, so alternate calls cost a load.

    Returns:
        float: A value drawn from the standard normal distribution.
    """
    global _spare, _haveSpare

    u1: cdouble
    u2: cdouble
    r: cdouble
    theta: cdouble

    if _haveSpare:
        _haveSpare = 0

        return _spare

    u1 = _uniform()
    u2 = _uniform()

    r     = csqrt(-2.0 * clog(u1))
    theta = 2.0 * M_PI * u2

    _spare     = r * csin(theta)
    _haveSpare = 1

    return r * ccos(theta)


@cfunc
@exceptval(check=False)
@cdivision(True)
def _sampleGamma(shape: cdouble, scale: cdouble) -> cdouble:
    """Draw from Gamma(shape, scale) by Marsaglia and Tsang (2000).

    The squeeze below needs shape >= 1; a shape under 1 is boosted by one,
    sampled, and then shrunk back down by an independent uniform variate
    raised to 1/shape, the standard way to extend the method to the full
    positive range.

    Args:
        shape (float): The gamma shape parameter, must be positive.
        scale (float): The gamma scale parameter, must not be negative.

    Returns:
        float: A value drawn from Gamma(shape, scale).
    """
    boost: cdouble = 1.0
    d: cdouble
    c: cdouble
    x: cdouble
    v: cdouble
    u: cdouble

    if shape < 1.0:
        boost = cpow(_uniform(), 1.0 / shape)
        shape = shape + 1.0

    d = shape - 1.0 / 3.0
    c = 1.0 / csqrt(9.0 * d)

    while True:
        v = -1.0
        while v <= 0.0:
            x = _standardNormal()
            v = 1.0 + c * x

        v = v * v * v
        u = _uniform()

        if u < 1.0 - 0.0331 * x * x * x * x:
            return boost * d * v * scale

        if clog(u) < 0.5 * x * x + d * (1.0 - v + clog(v)):
            return boost * d * v * scale


@cfunc
@exceptval(check=False)
@cdivision(True)
def _samplePoissonSmall(lam: cdouble) -> cdouble:
    """Draw from Poisson(lam) following Knuth (1969).

    Counts uniforms until their running product underflows exp(-lam). Exact,
    and cheap for a small lam, but the expected number of iterations is lam,
    so _samplePoisson() only routes here below its threshold.

    Args:
        lam (float): The Poisson rate, must not be negative.

    Returns:
        float: A value drawn from Poisson(lam), a non-negative integer held
             as a float to match sample()'s return type.
    """
    enlam: cdouble = cexp(-lam)
    prod: cdouble = 1.0
    k: cdouble = 0.0

    while True:
        prod = prod * _uniform()

        if prod <= enlam:
            return k

        k = k + 1.0


@cfunc
@exceptval(check=False)
@cdivision(True)
def _samplePoissonLarge(lam: cdouble) -> cdouble:
    """Draw from Poisson(lam) by Hormann's transformed rejection method
    (1993), used above the threshold where Knuth's method would need too
    many iterations. The log-gamma term is evaluated with lgamma() from
    math.h rather than tabulated, since k is unbounded.

    Args:
        lam (float): The Poisson rate, must be at least the threshold
                   _samplePoisson() switches to this method at.

    Returns:
        float: A value drawn from Poisson(lam), a non-negative integer held
             as a float to match sample()'s return type.
    """
    slam: cdouble = csqrt(lam)
    loglam: cdouble = clog(lam)
    b: cdouble = 0.931 + 2.53 * slam
    a: cdouble = -0.059 + 0.02483 * b
    invalpha: cdouble = 1.1239 + 1.1328 / (b - 3.4)
    vr: cdouble = 0.9277 - 3.6224 / (b - 2.0)

    U: cdouble
    V: cdouble
    us: cdouble
    k: cdouble

    while True:
        U = _uniform() - 0.5
        V = _uniform()
        us = 0.5 - cfabs(U)
        k = cfloor((2.0 * a / us + b) * U + lam + 0.43)

        if (us >= 0.07) and (V <= vr):
            return k

        if (k < 0.0) or ((us < 0.013) and (V > us)):
            continue

        if (clog(V) + clog(invalpha) - clog(a / (us * us) + b)) <= (-lam + k * loglam - clgamma(k + 1.0)):
            return k


@cfunc
@exceptval(check=False)
def _samplePoisson(lam: cdouble) -> cdouble:
    """Draw from Poisson(lam), routing to whichever method is cheap for lam.

    Args:
        lam (float): The Poisson rate, must not be negative.

    Returns:
        float: A value drawn from Poisson(lam).
    """
    if lam < 10.0:
        return _samplePoissonSmall(lam)

    return _samplePoissonLarge(lam)


# Parametric distributions. -----------------------------------------------------
#
# All subclass RanVar so a compiled model can still take them through a
# parameter annotated RanVar: sample() is declared cpdef in ranvar.pxd, so
# overriding it here gets dispatched through the vtable at the C level, the
# same as any other virtual method, with no Python call involved.

@cclass
class Normal(RanVar):
    """A digest that draws from a fixed Normal(mean, std) instead of data.

    sample() overrides the base t-digest lookup with a direct Box-Muller
    draw, so using a Normal as a compiled model's input costs one C call
    rather than a binary search through fitted centroids. fit() is
    overridden to match: it re-estimates mean and std analytically by the
    method of moments rather than accumulating points into a digest, and
    add() is refused outright, since a single point cannot estimate a
    spread.
    """
    _mean: cdouble
    _std: cdouble

    def __init__(self, mean=0.0, std=1.0, maxBins=32):
        """Create a fixed Normal(mean, std) digest.

        Args:
            mean (float, optional): The distribution's mean. Defaults to 0.0.
            std (float, optional): The distribution's standard deviation,
                                 must be positive. Defaults to 1.0.
            maxBins (int, optional): Passed through to RanVar. Unused by
                                   Normal itself, since neither add() nor
                                   fit() builds a digest here. Defaults to 32.

        Raises:
            ValueError: If std is not positive.
        """
        if std <= 0:
            raise ValueError('std must be positive')

        super().__init__(maxBins=maxBins)

        self._mean = mean
        self._std  = std

    @ccall
    @cdivision(True)
    def sample(self) -> cdouble:
        """Draw a value from Normal(mean, std).

        Returns:
            float: A value drawn from the distribution.
        """
        return self._mean + self._std * _standardNormal()

    def mean(self):
        """Return this distribution's fixed mean.

        Unlike RanVar.mean(), which averages the fitted centroids, this
        returns the parameter Normal currently holds, from construction or
        the last fit(), since sample() draws from that parameter rather than
        from any digest.

        Returns:
            float: The distribution's mean.
        """
        return self._mean

    def std(self):
        """Return this distribution's fixed standard deviation.

        Returns:
            float: The distribution's standard deviation.
        """
        return self._std

    def add(self, point, count=1.0):
        """Not implemented: a single point cannot estimate a spread.

        RanVar.add() appends one point to a t-digest. Normal instead only
        supports re-estimating both of its parameters at once from a full
        collection, via fit(), since mean and std cannot be estimated
        analytically from a single point.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            'Normal.add() is not supported: a single point cannot estimate '
            'a standard deviation. Use fit() with a full collection of data '
            'instead.'
        )

    def fit(self, x):
        """Re-estimate mean and std from data by the method of moments.

        Unlike RanVar.fit(), which adds every point to a t-digest, this
        replaces Normal's own parameters outright with the sample mean and
        the sample standard deviation of x (Bessel's correction, dividing by
        n - 1), the closed-form estimator for a normal distribution's
        parameters.

        Args:
            x (iterable): Data to estimate mean and std from. Needs at least
                         2 points, since a single point cannot estimate a
                         spread.

        Raises:
            ValueError: If x has fewer than 2 points.
        """
        values = list(x)
        n      = len(values)

        if n < 2:
            raise ValueError('fit() needs at least 2 points to estimate a standard deviation')

        mean = sum(values) / n
        var  = sum((v - mean) ** 2 for v in values) / (n - 1)

        self._mean = mean
        self._std  = var ** 0.5

    def lower(self):
        """Return this distribution's lower support bound.

        Unlike RanVar.lower(), which reads the smallest fitted centroid, this
        is the bound of Normal's own support and does not depend on its
        current mean or std.

        Returns:
            float: -inf, since a normal distribution has unbounded support.
        """
        return float('-inf')

    def upper(self):
        """Return this distribution's upper support bound.

        Returns:
            float: +inf, since a normal distribution has unbounded support.
        """
        return float('inf')

    def cdf(self, k):
        """Compute the CDF of Normal(mean, std) at a point.

        Uses erf() from math.h directly, since the normal CDF has a closed
        form, rather than RanVar.cdf()'s piecewise-linear read of a fitted
        digest.

        Args:
            k (float): The point to evaluate the CDF at.

        Returns:
            float: The probability of drawing at most k.
        """
        z = (k - self._mean) / (self._std * M_SQRT2)

        return 0.5 * (1.0 + cerf(z))

    def ccdf(self, x):
        """Compute the complementary CDF of Normal(mean, std) at a point.

        Uses erfc() directly rather than 1 - cdf(x), which loses precision
        far into the tail where cdf(x) is close to 1.

        Args:
            x (float): The point to evaluate the CCDF at.

        Returns:
            float: The probability of drawing more than x.
        """
        z = (x - self._mean) / (self._std * M_SQRT2)

        return 0.5 * cerfc(z)

    def dcdf(self, k):
        """Compute the density of Normal(mean, std) at a point.

        The normal density has a closed form, so this is exact rather than
        RanVar.dcdf()'s gradient of the piecewise-linear digest CDF.

        Args:
            k (float): The point to evaluate the density at.

        Returns:
            float: The density at k.
        """
        z = (k - self._mean) / self._std

        return cexp(-0.5 * z * z) / (self._std * csqrt(2.0 * M_PI))

    def dccdf(self, k):
        """Compute the derivative of the complementary CDF at a point.

        Args:
            k (float): The point to evaluate the derivative at.

        Returns:
            float: The negated density at k.
        """
        return -self.dcdf(k)

    @ccall
    def quantile(self, p: cdouble) -> cdouble:
        """Not implemented: the normal quantile function has no closed form.

        Its inverse needs the inverse error function, erfinv, which is not
        among the C99 math.h functions this library restricts itself to, so
        there is no analytical expression to evaluate here the way sample()
        and cdf() have.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            'Normal.quantile() has no closed form using only standard C '
            'math functions: it would need erfinv, which math.h does not '
            'provide.'
        )

    def toDict(self):
        """Serialise this distribution to plain JSON types.

        Unlike RanVar.toDict(), which writes the centroids, this writes the
        two parameters sample() draws from: neither add() nor fit() builds a
        digest here, so there are no centroids to carry.

        Returns:
            dict: The distribution, as JSON types only.
        """
        return {
            'type':    type(self).__name__,
            'mean':    self._mean,
            'std':     self._std,
            'maxBins': self.maxBins,
        }

    @classmethod
    def _fromDict(cls, data):
        """Rebuild a Normal from its own payload.

        Args:
            data (dict): A payload from toDict().

        Returns:
            Normal: The reconstructed distribution.
        """
        return cls(
            mean=float(data['mean']),
            std=float(data['std']),
            maxBins=int(data['maxBins']),
        )

    def __reduce__(self):
        """Support pickling and copying.

        Returns:
            tuple: The (callable, args, state) triple pickle expects.
        """
        return (
            type(self), (self._mean, self._std, self.maxBins),
            self._state(),
        )


@cclass
class NegBinom(RanVar):
    """A digest that draws from a fixed NegBinom(mean, dispersion) instead of
    data.

    Parameterized the NB2 way: variance = mean + dispersion * mean**2, the
    form regression and overdispersed count models usually fit. Internally
    this is drawn as a Gamma-Poisson mixture, so dispersion may be any
    positive real rather than only 1/integer.

    The textbook r/p parameterization -- the number of failures before the
    r-th success, at success probability p -- is accepted as an alternative
    at construction, and converted to mean and dispersion on the spot:
    everything past __init__ works in mean/dispersion only.

    fit() re-estimates mean and dispersion analytically by inverting that
    same variance formula from a sample's moments, rather than accumulating
    points into a digest, and add() is refused outright, since a single
    point cannot estimate a variance.
    """
    _mean: cdouble
    _dispersion: cdouble

    _shape: cdouble
    _scale: cdouble

    def __init__(self, mean=None, dispersion=None, maxBins=32, r=None, p=None):
        """Create a fixed NegBinom digest, in either parameterization.

        Give either mean and dispersion (the NB2 form this class works in
        throughout) or r and p (the textbook form), not a mix of the two.
        r/p is converted here and stored as mean and dispersion, so nothing
        downstream -- fit(), the accessors, pickling -- has to know which
        form it was built from.

        Args:
            mean (float, optional): The distribution's mean count, must not
                                  be negative. Defaults to 1.0.
            dispersion (float, optional): Overdispersion relative to a
                                        same-mean Poisson, must be positive.
                                        Defaults to 1.0.
            maxBins (int, optional): Passed through to RanVar. Unused by
                                   NegBinom itself, since neither add() nor
                                   fit() builds a digest here. Kept ahead of
                                   r/p in the signature so that positional
                                   calls (__reduce__ below among them) still
                                   reach it. Defaults to 32.
            r (float, optional): Number of successes to wait for, must be
                               positive. Needs p alongside it. Any positive
                               real, not only an integer, since the
                               Gamma-Poisson mixture sample() draws from
                               extends the distribution to real r.
            p (float, optional): Probability of success on each trial, in
                               (0, 1]. Needs r alongside it. p = 1 gives the
                               degenerate distribution at 0.

        Raises:
            ValueError: If both parameterizations are given, if only one of
                      r/p is, if mean is negative, if dispersion is not
                      positive, if r is not positive, or if p is outside
                      (0, 1].
        """
        if r is not None or p is not None:
            if mean is not None or dispersion is not None:
                raise ValueError(
                    'NegBinom takes either mean/dispersion or r/p, not both'
                )

            if r is None or p is None:
                raise ValueError('the r/p parameterization needs both r and p')

            if r <= 0:
                raise ValueError('r must be positive')

            if p <= 0 or p > 1:
                raise ValueError('p must be in (0, 1]')

            # mean = r(1 - p)/p and variance = r(1 - p)/p**2, which is
            # mean + mean**2/r, so dispersion (the NB2 coefficient on
            # mean**2) is exactly 1/r.
            mean       = r * (1.0 - p) / p
            dispersion = 1.0 / r
        else:
            mean       = 1.0 if mean is None else mean
            dispersion = 1.0 if dispersion is None else dispersion

        if mean < 0:
            raise ValueError('mean must not be negative')

        if dispersion <= 0:
            raise ValueError('dispersion must be positive')

        super().__init__(maxBins=maxBins)

        self._setParams(mean, dispersion)

    def _setParams(self, mean, dispersion):
        """Set mean and dispersion, and the cache sample() draws from.

        Args:
            mean (float): The distribution's mean count.
            dispersion (float): Overdispersion relative to a same-mean
                              Poisson.
        """
        self._mean       = mean
        self._dispersion = dispersion

        # shape = 1/dispersion, scale = mean/shape; cached so sample() only
        # multiplies rather than converting on every draw.
        self._shape = 1.0 / dispersion
        self._scale = mean * dispersion

    @ccall
    @cdivision(True)
    def sample(self) -> cdouble:
        """Draw a value from NegBinom(mean, dispersion).

        Draws a Gamma(shape, scale) rate and then a Poisson at that rate,
        the standard Gamma-Poisson mixture representation of the negative
        binomial, and what lets dispersion be a real number rather than only
        1/integer.

        Returns:
            float: A value drawn from the distribution (a non-negative
                 integer held as a float, consistent with quantile()
                 elsewhere in this hierarchy).
        """
        rate: cdouble = _sampleGamma(self._shape, self._scale)

        return _samplePoisson(rate)

    def mean(self):
        """Return this distribution's fixed mean.

        Unlike RanVar.mean(), which averages the fitted centroids, this
        returns the parameter NegBinom currently holds, from construction or
        the last fit(), since sample() draws from that parameter rather than
        from any digest.

        Returns:
            float: The distribution's mean.
        """
        return self._mean

    def dispersion(self):
        """Return this distribution's fixed dispersion parameter.

        Returns:
            float: How much the variance exceeds a same-mean Poisson's.
        """
        return self._dispersion

    def add(self, point, count=1.0):
        """Not implemented: a single point cannot estimate a variance.

        RanVar.add() appends one point to a t-digest. NegBinom instead only
        supports re-estimating both of its parameters at once from a full
        collection, via fit(), since mean and dispersion cannot be estimated
        analytically from a single point.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            'NegBinom.add() is not supported: a single point cannot '
            'estimate a variance. Use fit() with a full collection of data '
            'instead.'
        )

    def fit(self, x):
        """Re-estimate mean and dispersion from data by the method of
        moments.

        Unlike RanVar.fit(), which adds every point to a t-digest, this
        replaces NegBinom's own parameters outright by inverting its own
        variance = mean + dispersion * mean**2 parameterization with the
        sample mean and sample variance of x (Bessel's correction, dividing
        by n - 1): dispersion = (variance - mean) / mean**2.

        Args:
            x (iterable): Data to estimate mean and dispersion from. Needs at
                         least 2 points, a positive sample mean, and to be
                         overdispersed relative to a same-mean Poisson
                         (sample variance > sample mean), or there is no
                         positive dispersion that fits it.

        Raises:
            ValueError: If x has fewer than 2 points, a non-positive sample
                      mean, or is not overdispersed.
        """
        values = list(x)
        n      = len(values)

        if n < 2:
            raise ValueError('fit() needs at least 2 points to estimate a variance')

        mean = sum(values) / n

        if mean <= 0:
            raise ValueError('fit() needs a positive sample mean to estimate dispersion')

        var        = sum((v - mean) ** 2 for v in values) / (n - 1)
        dispersion = (var - mean) / (mean * mean)

        if dispersion <= 0:
            raise ValueError(
                'fit() needs overdispersed data (sample variance > sample '
                'mean) to estimate a positive dispersion'
            )

        self._setParams(mean, dispersion)

    def lower(self):
        """Return this distribution's lower support bound.

        Unlike RanVar.lower(), which reads the smallest fitted centroid, this
        is the bound of NegBinom's own support (a count, so it starts at 0)
        and does not depend on its current mean or dispersion.

        Returns:
            float: 0.0, the smallest count NegBinom can draw.
        """
        return 0.0

    def upper(self):
        """Return this distribution's upper support bound.

        Returns:
            float: +inf, since a count has no fixed largest value.
        """
        return float('inf')

    def cdf(self, k):
        """Not implemented: no closed form using only standard C math
        functions.

        The negative binomial CDF is a regularized incomplete beta function,
        which is not among the functions in math.h this library restricts
        itself to, and summing the PMF up to k would be an unbounded loop
        rather than an analytical expression.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            'NegBinom.cdf() has no closed form using only standard C math '
            'functions: it would need the regularized incomplete beta '
            'function, which math.h does not provide.'
        )

    def ccdf(self, x):
        """Not implemented, for the same reason as cdf().

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            'NegBinom.ccdf() has no closed form using only standard C math '
            'functions, for the same reason as NegBinom.cdf().'
        )

    def dcdf(self, k):
        """Not implemented: NegBinom is discrete, so it has no density.

        RanVar.dcdf() estimates a density from the piecewise-linear digest
        CDF, which is a meaningful thing to do for the continuous
        distribution a fitted digest approximates. NegBinom's CDF is a step
        function instead, whose derivative is zero almost everywhere and
        undefined at the steps, so there is nothing analogous to return.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            'NegBinom.dcdf() is not defined: NegBinom is a discrete '
            'distribution, so its CDF has no derivative.'
        )

    def dccdf(self, k):
        """Not implemented, for the same reason as dcdf().

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            'NegBinom.dccdf() is not defined, for the same reason as '
            'NegBinom.dcdf().'
        )

    @ccall
    def quantile(self, p: cdouble) -> cdouble:
        """Not implemented: no closed form using only standard C math
        functions.

        The negative binomial quantile function is the inverse of the
        regularized incomplete beta function, which math.h does not provide,
        so there is no analytical expression to evaluate here the way
        sample() has.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            'NegBinom.quantile() has no closed form using only standard C '
            'math functions, for the same reason as NegBinom.cdf().'
        )

    def toDict(self):
        """Serialise this distribution to plain JSON types.

        Written in the mean/dispersion form the class works in throughout,
        whichever parameterization it was built from, since r/p is converted
        away in __init__.

        Returns:
            dict: The distribution, as JSON types only.
        """
        return {
            'type':       type(self).__name__,
            'mean':       self._mean,
            'dispersion': self._dispersion,
            'maxBins':    self.maxBins,
        }

    @classmethod
    def _fromDict(cls, data):
        """Rebuild a NegBinom from its own payload.

        Args:
            data (dict): A payload from toDict().

        Returns:
            NegBinom: The reconstructed distribution.
        """
        return cls(
            mean=float(data['mean']),
            dispersion=float(data['dispersion']),
            maxBins=int(data['maxBins']),
        )

    def __reduce__(self):
        """Support pickling and copying.

        Returns:
            tuple: The (callable, args, state) triple pickle expects.
        """
        return (
            type(self), (self._mean, self._dispersion, self.maxBins),
            self._state(),
        )


@cclass
class Constant(RanVar):
    """A digest that always returns a fixed value instead of data.

    The degenerate distribution: all of its weight sits on a single point,
    so sample() returns that point rather than drawing anything, and every
    quantile of it is that same point. Useful wherever a model wants a plain
    number in a slot that has to be a RanVar -- a scenario held fixed, an
    assumption not yet given a spread -- without special casing the model.

    Unlike Normal and NegBinom, add() is supported here: a single point does
    fully determine a constant, so adding one just replaces the value.
    """
    _value: cdouble

    def __init__(self, value=0.0, maxBins=32):
        """Create a digest fixed at one value.

        Args:
            value (float, optional): The value every draw returns. Defaults
                                   to 0.0.
            maxBins (int, optional): Passed through to RanVar. Unused by
                                   Constant itself, since neither add() nor
                                   fit() builds a digest here. Defaults to 32.
        """
        super().__init__(maxBins=maxBins)

        self._value = value

    @ccall
    def sample(self) -> cdouble:
        """Return the fixed value.

        Returns:
            float: The value, the only one this distribution takes.
        """
        return self._value

    def value(self):
        """Return the fixed value.

        Returns:
            float: The value, from construction or the last add()/fit().
        """
        return self._value

    def mean(self):
        """Return this distribution's mean.

        Unlike RanVar.mean(), which averages the fitted centroids, this
        returns the value Constant currently holds, from construction or the
        last add()/fit(), since sample() returns that value rather than
        drawing from any digest.

        Returns:
            float: The value, which is also its mean.
        """
        return self._value

    def add(self, point, count=1.0):
        """Replace the value with a single point.

        RanVar.add() appends one weighted observation to a t-digest. A
        constant has room for only one point, so this replaces the value
        outright instead. count is ignored: a degenerate distribution puts
        all of its weight on that one point however it is weighted.

        Args:
            point (float): The new value.
            count (float, optional): Ignored. Accepted only to match
                                   RanVar.add()'s signature. Defaults to 1.0.
        """
        self._value = point

    def fit(self, x):
        """Re-estimate the value from data as its sample mean.

        Unlike RanVar.fit(), which adds every point to a t-digest, this
        replaces Constant's own value outright with the mean of x, the
        method of moments estimator for a degenerate distribution's single
        parameter. Unlike Normal.fit() and NegBinom.fit(), one point is
        enough, since there is no spread to estimate.

        Args:
            x (iterable): Data to estimate the value from. Needs at least 1
                         point.

        Raises:
            ValueError: If x is empty.
        """
        values = list(x)
        n      = len(values)

        if n < 1:
            raise ValueError('fit() needs at least 1 point to estimate a value')

        self._value = sum(values) / n

    def lower(self):
        """Return this distribution's lower support bound.

        Unlike RanVar.lower(), which reads the smallest fitted centroid, this
        is the bound of Constant's own support, which is the single point it
        sits on.

        Returns:
            float: The value.
        """
        return self._value

    def upper(self):
        """Return this distribution's upper support bound.

        Returns:
            float: The value, the same as lower(), since the support is a
                 single point.
        """
        return self._value

    def cdf(self, k):
        """Compute the CDF of the degenerate distribution at a point.

        A step: nothing is below the value, everything is at or below it
        from the value onwards.

        Args:
            k (float): The point to evaluate the CDF at.

        Returns:
            float: 0.0 below the value, 1.0 at or above it.
        """
        return 1.0 if k >= self._value else 0.0

    def ccdf(self, x):
        """Compute the complementary CDF at a point.

        Args:
            x (float): The point to evaluate the CCDF at.

        Returns:
            float: 1.0 below the value, 0.0 at or above it, the probability
                 of drawing more than x.
        """
        return 1.0 if x < self._value else 0.0

    def dcdf(self, k):
        """Not implemented: a degenerate distribution has no density.

        RanVar.dcdf() estimates a density from the piecewise-linear digest
        CDF. Constant's CDF is a single step instead, whose derivative is
        zero everywhere except at the value, where it is a Dirac delta rather
        than a number, so there is nothing analogous to return.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            'Constant.dcdf() is not defined: a constant is degenerate, so '
            'its CDF is a step with no derivative at the value it steps at.'
        )

    def dccdf(self, k):
        """Not implemented, for the same reason as dcdf().

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            'Constant.dccdf() is not defined, for the same reason as '
            'Constant.dcdf().'
        )

    @ccall
    def quantile(self, p: cdouble) -> cdouble:
        """Return the quantile for a given probability.

        Every quantile of a degenerate distribution is the point it sits on,
        so p is ignored, rather than RanVar.quantile()'s search through a
        fitted digest's cumulative weight ladder.

        Args:
            p (float): Probability value. Ignored.

        Returns:
            float: The value.
        """
        return self._value

    def toDict(self):
        """Serialise this distribution to plain JSON types.

        Only the point it sits on: a constant has no centroids, and add()
        and fit() replace that point rather than building a digest.

        Returns:
            dict: The distribution, as JSON types only.
        """
        return {
            'type':    type(self).__name__,
            'value':   self._value,
            'maxBins': self.maxBins,
        }

    @classmethod
    def _fromDict(cls, data):
        """Rebuild a Constant from its own payload.

        Args:
            data (dict): A payload from toDict().

        Returns:
            Constant: The reconstructed distribution.
        """
        return cls(value=float(data['value']), maxBins=int(data['maxBins']))

    def __reduce__(self):
        """Support pickling and copying.

        Returns:
            tuple: The (callable, args, state) triple pickle expects.
        """
        return (
            type(self), (self._value, self.maxBins),
            self._state(),
        )

# A collection of digests. -----------------------------------------------------

@cclass
class RanVarArray():
    """A mutable, list-like sequence of RanVar instances.

    Some models depend on a whole collection of digests rather than a single
    one -- one per region, one per product line -- and RanVarArray is what
    lets a model take that as one parameter instead of *args. It behaves like
    an ordinary list (indexing, slicing, append, len, +, iteration, ...), and
    every element has to be a RanVar (Normal, NegBinom and Constant included,
    since they all subclass it), checked on insertion.

    Its other job is letting ranvar2.compiler compile a loop over it: `for x
    in arr:` inside a @cfunc model, where arr is a parameter annotated
    RanVarArray, is rewritten into an indexed loop calling _at() below, so
    that fetching each element -- and then drawing from it with ~x or
    x.sample() -- is a direct C call rather than the generic (Python object
    producing) iteration protocol. See ranvar2.compiler.ArrayLoops.

    Attributes:
        _items (list): The wrapped RanVar instances, in order.

    Example:
        >>> regions = RanVarArray.fromList([Normal(100, 10), Normal(200, 20)])
        >>> len(regions)
        2
        >>> regions.append(Normal(50, 5))
        >>> [r.mean() for r in regions]
        [100.0, 200.0, 50.0]
    """

    _items: list

    def __init__(self, items=None):
        """Wrap a collection of RanVar instances.

        Args:
            items (iterable, optional): RanVar instances to start with.
                                       Defaults to an empty array.

        Raises:
            TypeError: If any item is not a RanVar instance.
        """
        self._items = []

        if items is not None:
            for item in items:
                self._items.append(self._check(item))

    @classmethod
    def fromList(cls, items):
        """Build a RanVarArray from a plain Python list (or any iterable).

        Args:
            items (iterable): RanVar instances to wrap.

        Returns:
            RanVarArray: The new array.

        Raises:
            TypeError: If any item is not a RanVar instance.
        """
        return cls(items)

    def _check(self, item):
        """Validate that an item may go into the array.

        Args:
            item (object): The candidate item.

        Returns:
            RanVar: item, unchanged.

        Raises:
            TypeError: If item is not a RanVar instance.
        """
        if not isinstance(item, RanVar):
            raise TypeError(
                f'RanVarArray only holds RanVar instances (Normal, NegBinom, '
                f'Constant, ... included, since they all subclass it), got '
                f'{type(item).__name__!r}'
            )

        return item

    @ccall
    def _at(self, i: cint) -> RanVar:
        """Fetch an element through a C signature, for compiled code to call.

        Reached both from the compiler's own loop counters (always in range)
        and from a user's own index expression rewritten from arr[i] (not
        guaranteed to be), so bounds checking and negative-index wraparound
        stay on here, unlike the internal methods elsewhere in this module
        that only ever see indices their own algorithm already guarantees are
        in range.

        Args:
            i (int): Index of the element to fetch. May be negative, Python
                   style, to count from the end.

        Returns:
            RanVar: The element at that index.

        Raises:
            IndexError: If i is out of range.
        """
        return self._items[i]

    def __len__(self) -> cint:
        """Number of RanVar instances in the array.

        Typed so a compiled model's `len(arr)` reads the count directly rather
        than boxing it into a Python int first.

        Returns:
            int: The element count.
        """
        return len(self._items)

    def __getitem__(self, key):
        """Get an item or a slice.

        Args:
            key (int | slice): Position, or range of positions, to fetch.

        Returns:
            RanVar: The element at key, if key is an int.
            RanVarArray: A new array over the slice, if key is a slice.
        """
        if isinstance(key, slice):
            return RanVarArray(self._items[key])

        return self._items[key]

    def __setitem__(self, key, value):
        """Set an item or a slice.

        Args:
            key (int | slice): Position, or range of positions, to replace.
            value (RanVar | iterable): The replacement, matching key's shape.

        Raises:
            TypeError: If value (or any of its elements) is not a RanVar.
        """
        if isinstance(key, slice):
            self._items[key] = [self._check(v) for v in value]
        else:
            self._items[key] = self._check(value)

    def __delitem__(self, key):
        """Remove an item or a slice.

        Args:
            key (int | slice): Position, or range of positions, to remove.
        """
        del self._items[key]

    def __iter__(self):
        """Iterate over the wrapped RanVar instances.

        Returns:
            iterator: Yields each RanVar in order.
        """
        return iter(self._items)

    def __reversed__(self):
        """Iterate over the wrapped RanVar instances in reverse.

        Returns:
            iterator: Yields each RanVar in reverse order.
        """
        return reversed(self._items)

    def __contains__(self, item):
        """Report whether an item is in the array.

        Args:
            item (object): The item to look for.

        Returns:
            bool: True if item is present.
        """
        return item in self._items

    def __bool__(self):
        """Report whether the array holds any elements.

        Returns:
            bool: True if the array is non empty.
        """
        return bool(self._items)

    def __eq__(self, other):
        """Compare against another RanVarArray or a plain list.

        Args:
            other (object): What to compare against.

        Returns:
            bool: True if other holds the same elements in the same order.
            NotImplemented: If other is neither a RanVarArray nor a list.
        """
        # A cdef attribute reached through a plainly typed parameter is a
        # Python-level lookup rather than the direct struct access it would be
        # through self, and cdef attributes aren't visible that way at all, so
        # `other` is narrowed to its static type first.
        if isinstance(other, RanVarArray):
            typed: RanVarArray = other

            return self._items == typed._items

        if isinstance(other, list):
            return self._items == other

        return NotImplemented

    def index(self, item, start=0, stop=None):
        """Find the position of an item.

        Args:
            item (RanVar): The item to look for.
            start (int, optional): Position to start searching from.
            stop (int, optional): Position to stop searching before.

        Returns:
            int: The position of the first matching item.

        Raises:
            ValueError: If item is not present.
        """
        if stop is None:
            stop = len(self._items)

        return self._items.index(item, start, stop)

    def count(self, item):
        """Count how many times an item appears.

        Args:
            item (RanVar): The item to count.

        Returns:
            int: How many elements equal item.
        """
        return self._items.count(item)

    def append(self, item):
        """Add an item to the end.

        Args:
            item (RanVar): The item to add.

        Raises:
            TypeError: If item is not a RanVar instance.
        """
        self._items.append(self._check(item))

    def insert(self, index, item):
        """Insert an item before a position.

        Args:
            index (int): Position to insert before.
            item (RanVar): The item to insert.

        Raises:
            TypeError: If item is not a RanVar instance.
        """
        self._items.insert(index, self._check(item))

    def extend(self, items):
        """Add every item from an iterable to the end.

        Args:
            items (iterable): RanVar instances to add.

        Raises:
            TypeError: If any item is not a RanVar instance.
        """
        self._items.extend(self._check(item) for item in items)

    def pop(self, index=-1):
        """Remove and return an item.

        Args:
            index (int, optional): Position to remove. Defaults to the last.

        Returns:
            RanVar: The removed item.

        Raises:
            IndexError: If the array is empty.
        """
        return self._items.pop(index)

    def remove(self, item):
        """Remove the first occurrence of an item.

        Args:
            item (RanVar): The item to remove.

        Raises:
            ValueError: If item is not present.
        """
        self._items.remove(item)

    def clear(self):
        """Remove every item."""
        self._items.clear()

    def copy(self):
        """Make a shallow copy.

        Returns:
            RanVarArray: A new array wrapping the same RanVar instances.
        """
        return RanVarArray(self._items)

    def reverse(self):
        """Reverse the array in place."""
        self._items.reverse()

    def sort(self, key=None, reverse=False):
        """Sort the array in place.

        RanVar instances have no natural order, so a key function is usually
        needed, e.g. `arr.sort(key=lambda r: r.mean())`.

        Args:
            key (callable, optional): Function computing a sort key per item.
            reverse (bool, optional): Sort descending instead of ascending.
        """
        self._items.sort(key=key, reverse=reverse)

    def __add__(self, other):
        """Concatenate with another array or list.

        Args:
            other (RanVarArray | list): What to append.

        Returns:
            RanVarArray: A new array holding both sides' elements.
            NotImplemented: If other is neither a RanVarArray nor a list.
        """
        if isinstance(other, RanVarArray):
            typed: RanVarArray = other

            return RanVarArray(self._items + typed._items)

        if isinstance(other, list):
            return RanVarArray(self._items + other)

        return NotImplemented

    def __radd__(self, other):
        """Concatenate when this array is on the right of +.

        Args:
            other (list): What this array is being appended to.

        Returns:
            RanVarArray: A new array holding both sides' elements.
            NotImplemented: If other is not a list.
        """
        if isinstance(other, list):
            return RanVarArray(other + self._items)

        return NotImplemented

    def __iadd__(self, other):
        """Extend in place with +=.

        Args:
            other (iterable): RanVar instances to add.

        Returns:
            RanVarArray: This array.

        Raises:
            TypeError: If any item is not a RanVar instance.
        """
        self.extend(other)

        return self

    def __mul__(self, n):
        """Repeat the array's elements n times.

        Args:
            n (int): Number of repetitions.

        Returns:
            RanVarArray: A new array with the elements repeated.
        """
        return RanVarArray(self._items * n)

    def __rmul__(self, n):
        """Repeat the array's elements n times (n * arr).

        Args:
            n (int): Number of repetitions.

        Returns:
            RanVarArray: A new array with the elements repeated.
        """
        return RanVarArray(self._items * n)

    def __imul__(self, n):
        """Repeat the array's elements n times, in place.

        Args:
            n (int): Number of repetitions.

        Returns:
            RanVarArray: This array.
        """
        self._items *= n

        return self

    def __repr__(self):
        """Describe this array, for debugging.

        Returns:
            str: The wrapped elements, in list form.
        """
        return f'RanVarArray({self._items!r})'

    def __reduce__(self):
        """Support pickling and copying.

        Returns:
            tuple: The (callable, args) pair pickle expects.
        """
        return (RanVarArray, (list(self._items),))


# Coercion. --------------------------------------------------------------------

def asRanVar(x):
    """Coerce a value into the RanVar (or RanVarArray) a model takes.

    A model's parameters are digests, but the values a caller has on hand are
    often plain numbers -- a lead time of 7, a price held fixed -- or a plain
    list of digests rather than a RanVarArray. This wraps those, so a caller
    can pass whatever it holds and a model that is stochastic in a parameter
    today still works when that parameter is a constant tomorrow.

    Args:
        x (object): The value to coerce. A RanVar (Normal, NegBinom, Constant
                  and any other subclass included) or a RanVarArray is
                  returned unchanged; a real number becomes a Constant; a list
                  or tuple becomes a RanVarArray, with every element coerced
                  the same way, so a list of plain numbers works as well as a
                  list of digests.

    Returns:
        RanVar | RanVarArray: x itself, or the digest wrapping it.

    Raises:
        TypeError: If x is none of those. A nested list reports the same way,
                 from the RanVarArray it would have had to hold.

    Example:
        >>> asRanVar(7).sample()
        7.0
        >>> len(asRanVar([1.0, Normal(mean=2.0, std=1.0)]))
        2
    """
    if isinstance(x, (RanVar, RanVarArray)):
        return x

    if isinstance(x, (list, tuple)):
        return RanVarArray([asRanVar(item) for item in x])

    # numbers.Real rather than (int, float), so that the numpy scalars a
    # dataframe hands out (np.int64 among them, which is not an int subclass)
    # are wrapped rather than refused.
    if isinstance(x, numbers.Real):
        return Constant(float(x))

    raise TypeError(
        f'asRanVar() cannot convert {type(x).__name__!r}: it takes a RanVar, '
        f'a RanVarArray, a real number (wrapped in a Constant), or a list of '
        f'those (wrapped in a RanVarArray)'
    )
