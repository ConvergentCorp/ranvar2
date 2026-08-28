import numpy  as np
from cython import cclass, cfunc, boundscheck, initializedcheck, wraparound, ccall, cdivision, cast

from cython import double as cdouble
from cython import int as cint
from cython import void as cvoid

from cython.cimports.libc.stdlib import rand as crand
from cython.cimports.libc.stdlib import srand as csrand
from cython.cimports.libc.stdlib import RAND_MAX as C_RAND_MAX


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


    def sample(self):
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
        """
        Syntactic sugar for the sample method.
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
    
