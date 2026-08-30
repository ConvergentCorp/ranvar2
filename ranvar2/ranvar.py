import numpy  as np
from cython import cclass, cfunc, boundscheck, initializedcheck, wraparound, ccall, cdivision, cast

from cython import double as cdouble
from cython import int as cint
from cython import void as cvoid

from cython.cimports.libc.stdlib import rand as crand
from cython.cimports.libc.stdlib import srand as csrand
from cython.cimports.libc.stdlib import RAND_MAX as C_RAND_MAX

from cython.cimports.libc.math import log as clog
from cython.cimports.libc.math import sqrt as csqrt
from cython.cimports.libc.math import cos as ccos
from cython.cimports.libc.math import exp as cexp
from cython.cimports.libc.math import floor as cfloor
from cython.cimports.libc.math import fabs as cfabs
from cython.cimports.libc.math import lgamma as clgamma
from cython.cimports.libc.math import pow as cpow
from cython.cimports.libc.math import erf as cerf
from cython.cimports.libc.math import erfc as cerfc
from cython.cimports.libc.math import M_PI
from cython.cimports.libc.math import M_SQRT2


def seed(value):
    """Seed the random number generator that sample() draws from.

    Sampling uses the C library generator, which is global to the process, so
    this seeds every digest at once rather than any single instance.

    Args:
        value (int): Seed for the generator.
    """
    csrand(value)


@cclass
class RanVar():
    """
    Implementation of the t-digest algorithm for computing quantiles as class RanVar

    The t-digest algorithm is described in 'Computing Extremely Accurate Quantiles Using t-Digests'
    by Ted Dunning. This data structure provides approximate quantile computation with bounded
    memory usage and high accuracy, particularly at the tails of the distribution.

    The digest maintains a set of centroids (weighted points) that summarize the distribution.
    When the number of centroids exceeds maxBins, nearby centroids are merged to maintain
    the memory bound while preserving accuracy.

    quantile() walks a ladder of cumulative weights, one rung per pair of
    adjacent centroids. That ladder only changes when a point is added, so it is
    cached and searched rather than rebuilt on every call, which is what makes
    repeated sampling from a fitted digest cheap.

    Attributes:
        bins (np.ndarray): Array storing centroid values (x-coordinates)
        cnts (np.ndarray): Array storing centroid weights (counts)
        cumw (np.ndarray): Cached cumulative weight at the start of each segment
        gaps (np.ndarray): Cached weight spanned by each segment
        maxBins (int): Maximum number of centroids to maintain
        nActive (int): Current number of active centroids

    Example:
        >>> ranvar = RanVar(maxBins=100)
        >>> ranvar.fit([1, 2, 3, 4, 5])
        >>> digest.quantile(0.5)  # Median
        3.0
    """    
    _bins: cdouble[:]
    _cnts: cdouble[:]
    _cumw: cdouble[:]
    _gaps: cdouble[:]

    bins: np.ndarray
    cnts: np.ndarray
    cumw: np.ndarray
    gaps: np.ndarray

    maxBins: cint
    nActive: cint

    # Total weight as _sumWeights() would compute it, cached alongside the
    # ladder, and a flag saying whether either still reflects the centroids.
    _total: cdouble
    _stale: cint

    def __init__(self, maxBins=32):
        """Initialize a new t-digest.

        Args:
            maxBins (int, optional): Maximum number of centroids to maintain.
                                   Defaults to 32.
        """
        self.maxBins = maxBins
        self.nActive = 0

        self.bins = np.zeros(self.maxBins + 1, dtype=np.float64)
        self.cnts = np.zeros(self.maxBins + 1, dtype=np.float64)
        self.cumw = np.zeros(self.maxBins + 1, dtype=np.float64)
        self.gaps = np.zeros(self.maxBins + 1, dtype=np.float64)

        self._bins = self.bins
        self._cnts = self.cnts
        self._cumw = self.cumw
        self._gaps = self.gaps

        self._total = 0
        self._stale = 1

    @ccall
    @boundscheck(False)
    @wraparound(False)
    @initializedcheck(False)
    def _findLastLesserOrEqualIndex(self, point: cdouble) -> cint:
        """Find the index of the last centroid that is <= the given point.

        This method performs a linear search through the sorted centroids to find
        the insertion point for a new value.

        Args:
            point (float): The value to search for.

        Returns:
            int: Index of the last centroid <= point, or -1 if point is smaller
                than all centroids.
        """
        idx: cint = -1
        while True:
            if (self._bins[idx + 1] > point) or (idx + 1 == self.nActive):
                break
            else:
                idx += 1

        return idx

    @ccall
    @boundscheck(False)
    @wraparound(False)
    @initializedcheck(False)
    def _shiftRightAndInsert(self, idx:cint, point:cdouble, count:cdouble) -> cvoid:
        """Insert a new centroid at the specified position by shifting elements right.

        This method maintains the sorted order of centroids by shifting all centroids
        to the right of the insertion point and inserting the new centroid.

        Args:
            idx (int): Index after which to insert the new centroid.
            point (float): Value of the new centroid.
            count (float): Weight of the new centroid.
        """
        j: cint

        for j in range(self.nActive - 1, idx, -1):
            self._bins[j+1] = self._bins[j]
            self._cnts[j+1] = self._cnts[j]

        self._bins[idx+1] = point
        self._cnts[idx+1] = count
        self.nActive += 1

    @cfunc
    @boundscheck(False)
    @wraparound(False)
    @initializedcheck(False)
    def _findMinimumDifference(self) -> cint:
        """Find the pair of adjacent centroids with minimum distance.

        This method is used when the digest exceeds maxBins to identify which
        centroids should be merged. It avoids merging the first and last centroids
        to preserve the tails of the distribution.

        Returns:
            int: Index of the first centroid in the pair with minimum distance,
                or -1 if no suitable pair is found.
        """
        k: cint
        dB: cdouble
        minK: cint
        minDiff: cdouble

        minK    = -1
        minDiff = 9e9
        # We don't want to merge the first or last bin because we want to maintain the
        # tails. It also solves the problem where we try and sample a point that is before
        # the first centroid.
        for k in range(1, self.nActive - 2):
            dB = self._bins[k+1] - self._bins[k]
            if dB < minDiff:
                minDiff = dB
                minK    = k

        return minK

    @ccall
    @boundscheck(False)
    @wraparound(False)
    @initializedcheck(False)
    def _shiftLeftAndOverride(self, idx: cint) -> cvoid:
        """Remove a centroid by shifting all subsequent centroids left.

        This method maintains the sorted order and compactness of the centroid
        arrays by removing the centroid at the specified index.

        Args:
            idx (int): Index of the centroid to remove.
        """
        j: cint
        for j in range(idx, self.nActive-1):
            self._bins[j] = self._bins[j+1]
            self._cnts[j] = self._cnts[j+1]

        self._bins[self.nActive-1] = 0
        self._cnts[self.nActive-1] = 0

        self.nActive -= 1

    @ccall
    @boundscheck(False)
    @wraparound(False)
    @cdivision(True)
    @initializedcheck(False)
    def _add(self, point:cdouble, count:cdouble) -> cvoid:
        """Add a weighted point to the digest (internal implementation).

        This is the core method that implements the t-digest algorithm. It either
        updates an existing centroid if the point matches exactly, or inserts a new
        centroid. If the number of centroids exceeds maxBins, it merges the two
        closest centroids.

        Args:
            point (float): The value to add.
            count (float): The weight/count of the value.
        """

        idx:cint = self._findLastLesserOrEqualIndex(point)

        if (idx >= 0) and self._bins[idx] == point:
            self._cnts[idx] += count
        else:
            self._shiftRightAndInsert(idx, point, count)

        if self.nActive > self.maxBins:
            k:cint = self._findMinimumDifference()

            sumC:cdouble = self._cnts[k+1] + self._cnts[k]

            self._bins[k] = (self._bins[k]*self._cnts[k] + self._bins[k+1]*self._cnts[k+1])
            self._bins[k] = self._bins[k] / sumC
            self._cnts[k] = sumC

            self._shiftLeftAndOverride(k+1)

        self._stale = 1


    @ccall
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

    @ccall
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
    @boundscheck(False)
    @wraparound(False)
    @cdivision(True)
    @initializedcheck(False)
    def _interpolationBounds(self, i: cint, som: cdouble):
        """Cumulative weights bracketing the segment between centroid i and i+1.

        We use the approach of Dunning here to improve interpolation when we have
        single weighted points: a centroid carrying a weight of one sits at an
        exact observation, so the segment interpolates from the centroid itself
        rather than from its midpoint.

        Args:
            i (int): Index of the left centroid of the segment.
            som (float): Total weight of all centroids before i.

        Returns:
            tuple: (yi, yi_n), the cumulative weights at c[i] and c[i+1].
        """
        m = self._cnts

        if (m[i] > 1) & (m[i+1] > 1):
            # Case I: Both points greater than one, normal interpolation.
            yi   = som + m[i]/2
            yi_n = yi + (m[i+1] + m[i]) / 2

        elif (m[i] == 1) & (m[i+1] > 1):
            # Case II: Left point is a single observation.
            yi   = som
            yi_n = yi + (m[i+1]) / 2

        elif (m[i] > 1) & (m[i+1] == 1):
            # Case III: Right point is a single observation.
            yi   = som + m[i]/2
            yi_n = yi + (m[i]) / 2
        else:
            # Case IV: Both are single observations, nothing to interpolate over.
            yi   = som
            yi_n = yi

        return yi, yi_n

    @cfunc
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

        m = self._cnts

        som = 0
        for i in range(self.nActive):
            som = som + m[i]

        self._total = som

        som = 0
        self._cumw[0] = 0

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

            self._gaps[i] = wGap

            som = som + wGap
            self._cumw[i+1] = som

        self._stale = 0

    # Public API. --------------------------------------------------------------

    def add(self, point, count=1.0):
        """Add a single weighted observation to the digest.

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
        return self.nActive

    def getBins(self):
        """Get the active centroid values.

        Returns:
            np.ndarray: View of the centroid values, in ascending order.
        """
        return self.bins[:self.nActive]

    def getWeights(self):
        """Get the active centroid weights.

        Returns:
            np.ndarray: View of the centroid weights, aligned with getBins().
        """
        return self.cnts[:self.nActive]

    def lower(self):
        """Get the minimum value in the digest.

        Returns:
            float: The smallest centroid value.

        Raises:
            ValueError: If the digest is empty.
        """
        return self._lower()

    def upper(self):
        """Get the maximum value in the digest.

        Returns:
            float: The largest centroid value.

        Raises:
            ValueError: If the digest is empty.
        """
        return self._upper()

    def cdf(self, k):
        """Compute the cumulative distribution function at a given point.

        Implements the CDF estimation algorithm from Ted Dunning's paper
        'Computing Extremely Accurate Quantiles Using t-Digests'. Uses
        different interpolation strategies depending on whether centroids
        represent single points or aggregated ranges.

        Args:
            k (float): The point at which to evaluate the CDF.

        Returns:
            float: Estimated CDF value between 0 and 1.
        """
        som:cdouble = 0
        i:cint

        c = self._bins
        m = self._cnts


        if k <= self._lower():
            return 0.
        elif k >= self._upper():
            return 1.
        else:
            for i in range(self.nActive):
                if c[i] <= k < c[i+1]:
                    yi, yi_n = self._interpolationBounds(i, som)

                    g    = (yi_n - yi) / (c[i+1] - c[i])
                    yk   = g*(k - c[i]) + yi

                    return yk / self._sumWeights()

                else:
                    som += m[i]
        
        pass

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

    def dcdf(self, k):
        """Compute the derivative of the CDF at a given point.

        The CDF is piecewise linear between centroids, so its derivative is the
        constant gradient of the segment containing k, normalised by the total
        weight. This is the density estimate at k. Outside the support the CDF is
        flat, so the derivative is zero.

        Args:
            k (float): The point at which to evaluate the derivative.

        Returns:
            float: Estimated density at k.
        """
        som:cdouble = 0
        i:cint

        c = self._bins
        m = self._cnts

        if (k <= self._lower()) or (k >= self._upper()):
            return 0.

        for i in range(self.nActive - 1):
            if c[i] <= k < c[i+1]:
                yi, yi_n = self._interpolationBounds(i, som)

                g = (yi_n - yi) / (c[i+1] - c[i])

                return g / self._sumWeights()

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

        c   = self._bins
        cum = self._cumw

        if p <= 0:
            return self._lower()
        elif p >= 1:
            return self._upper()

        if self._stale:
            self._rebuildLadder()

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

        p: cdouble = cast(cdouble, crand()) / cast(cdouble, C_RAND_MAX)
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

        if self.nActive == 0:
            raise ValueError('mean() is undefined for an empty digest')

        for i in range(self.nActive):
            som = som + self._bins[i]*self._cnts[i]

        return som / self._sumWeights()

    def fit(self, x):
        """Fit the digest to a collection of data points.

        This is a convenience method that adds all points in the collection
        to the digest with equal weight.

        Args:
            x (iterable): Collection of numeric values to add to the digest.
        """
        for xx in x:
            self._add(xx, 1.0)


    def __invert__(self):
        """Syntactic sugar for the sample method.

        Returns:
            float: A value drawn from the distribution, as sample() would give.

        Raises:
            ValueError: If the digest is empty.
        """
        return self.sample()

    def __reduce__(self):
        """Support pickling and copying.

        Extension types holding memoryviews get no default __reduce__, so we
        serialise the backing arrays and rebind the views in __setstate__.

        Returns:
            tuple: The (callable, args, state) triple pickle expects.
        """
        return (type(self), (self.maxBins,), (self.nActive, self.bins, self.cnts))

    def __setstate__(self, state):
        """Restore the state produced by __reduce__.

        Args:
            state (tuple): The (nActive, bins, cnts) triple from __reduce__.
        """
        self.nActive = state[0]
        self.bins = np.array(state[1], dtype=np.float64)
        self.cnts = np.array(state[2], dtype=np.float64)

        self._bins = self.bins
        self._cnts = self.cnts

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
@cdivision(True)
def _uniform() -> cdouble:
    """Draw from the open interval (0, 1), never exactly 0 or 1.

    Box-Muller and the gamma sampler below both take a log of this, so the
    closed end at 0 that crand() can in principle return would blow up to
    -inf.

    Returns:
        float: A uniform draw strictly between 0 and 1.
    """
    u: cdouble = cast(cdouble, crand()) / cast(cdouble, C_RAND_MAX)

    while u <= 0.0 or u >= 1.0:
        u = cast(cdouble, crand()) / cast(cdouble, C_RAND_MAX)

    return u


@cfunc
@cdivision(True)
def _standardNormal() -> cdouble:
    """Draw one N(0, 1) value using the Box-Muller transform.

    Returns:
        float: A value drawn from the standard normal distribution.
    """
    u1: cdouble = _uniform()
    u2: cdouble = _uniform()

    return csqrt(-2.0 * clog(u1)) * ccos(2.0 * M_PI * u2)


@cfunc
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
# Both subclass RanVar so a compiled model can still take them through a
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

    def __reduce__(self):
        """Support pickling and copying.

        Returns:
            tuple: The (callable, args, state) triple pickle expects.
        """
        return (
            type(self), (self._mean, self._std, self.maxBins),
            (self.nActive, self.bins, self.cnts),
        )


@cclass
class NegBinom(RanVar):
    """A digest that draws from a fixed NegBinom(mean, dispersion) instead of
    data.

    Parameterized the NB2 way: variance = mean + dispersion * mean**2, the
    form regression and overdispersed count models usually fit. Internally
    this is drawn as a Gamma-Poisson mixture, so dispersion may be any
    positive real rather than only 1/integer.

    fit() re-estimates mean and dispersion analytically by inverting that
    same variance formula from a sample's moments, rather than accumulating
    points into a digest, and add() is refused outright, since a single
    point cannot estimate a variance.
    """
    _mean: cdouble
    _dispersion: cdouble

    _shape: cdouble
    _scale: cdouble

    def __init__(self, mean=1.0, dispersion=1.0, maxBins=32):
        """Create a fixed NegBinom(mean, dispersion) digest.

        Args:
            mean (float, optional): The distribution's mean count, must not
                                  be negative. Defaults to 1.0.
            dispersion (float, optional): Overdispersion relative to a
                                        same-mean Poisson, must be positive.
                                        Defaults to 1.0.
            maxBins (int, optional): Passed through to RanVar. Unused by
                                   NegBinom itself, since neither add() nor
                                   fit() builds a digest here. Defaults to 32.

        Raises:
            ValueError: If mean is negative or dispersion is not positive.
        """
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

    def __reduce__(self):
        """Support pickling and copying.

        Returns:
            tuple: The (callable, args, state) triple pickle expects.
        """
        return (
            type(self), (self._mean, self._dispersion, self.maxBins),
            (self.nActive, self.bins, self.cnts),
        )

