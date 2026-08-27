import numpy  as np
from cython import cclass, cfunc, boundscheck, initializedcheck, wraparound, ccall, cdivision, cast

import cython.double as cdouble
import cython.int as cint
import cython.void as cvoid

from cython.cimports.libc.stdlib import rand as crand
from cython.cimports.libc.stdlib import RAND_MAX as C_RAND_MAX


class RanVar():
    """
    Implementation of the t-digest algorithm for computing quantiles as class RanVar

    The t-digest algorithm is described in 'Computing Extremely Accurate Quantiles Using t-Digests'
    by Ted Dunning. This data structure provides approximate quantile computation with bounded
    memory usage and high accuracy, particularly at the tails of the distribution.

    The digest maintains a set of centroids (weighted points) that summarize the distribution.
    When the number of centroids exceeds maxBins, nearby centroids are merged to maintain
    the memory bound while preserving accuracy.

    Attributes:
        bins (np.ndarray): Array storing centroid values (x-coordinates)
        cnts (np.ndarray): Array storing centroid weights (counts)
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

    bins: np.ndarray
    cnts: np.ndarray

    maxBins: cint
    nActive: cint

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

        self._bins = self.bins
        self._cnts = self.cnts

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


    @ccall
    @boundscheck(False)
    @cdivision(True)
    @wraparound(False)
    @initializedcheck(False)
    def _lower(self) -> cint:
        """Get the minimum value in the digest.

        Returns:
            int: The smallest centroid value, cast to integer.
        """
        return int(self._bins[0])

    @ccall
    @boundscheck(False)
    @cdivision(True)
    @wraparound(False)
    @initializedcheck(False)
    def _upper(self) -> cint:
        """Get the maximum value in the digest.

        Returns:
            int: The largest centroid value, cast to integer.
        """
        return int(self._bins[self.nActive - 1])

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
    
    # Public API. --------------------------------------------------------------

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
                    # We use the approach of Dunning here to improve interpolation when
                    # we have single weighted points.
                    if (m[i] > 1) & (m[i+1] > 1):
                        # Case I: Both points greater than one, normal interpolation.
                        yi   = som + m[i]/2
                        yi_n = yi + (m[i+1] + m[i]) / 2

                    elif (m[i] == 1) & (m[i+1] > 1):
                        # Case I: Both points greater than one, normal interpolation.
                        yi   = som
                        yi_n = yi + (m[i+1]) / 2

                    elif (m[i] > 1) & (m[i+1] == 1):
                        # Case I: Both points greater than one, normal interpolation.
                        yi   = som + m[i]/2
                        yi_n = yi + (m[i]) / 2
                    else:
                        yi   = som
                        yi_n = yi

                    g    = (yi_n - yi) / (c[i+1] - c[i])
                    yk   = g*(k - c[i]) + yi

                    return yk / self._sumWeights()

                else:
                    som += m[i]
        
        pass

    def ccdf(self, x):
        return 1 - self.cdf(x)

    def quantile(self, p):
        """Compute the quantile for a given probability.

        Uses linear interpolation between centroids to estimate the quantile.
        The interpolation accounts for the different weighting schemes at
        boundaries and interior points.

        Args:
            p (float): Probability value between 0 and 1.

        Returns:
            float: Estimated quantile value.

        Raises:
            No explicit validation, but p should be in [0, 1] for meaningful results.
        """
        if p <= 0:
            return self._lower()
        elif p >= 1:
            return self._upper()
        else:
            W  = self._sumWeights()
            m = self.cnts
            c = self.bins
            wi = 0
            w_ = p*W

            for i in range(self.nActive-1):
                if i == 0:
                    wGap = m[i] + m[i+1]/2
                elif i == self.nActive - 1:
                    wGap = m[i]/2 + m[i+1]
                else:
                    wGap = m[i]/2 + m[i+1]/2

                wi_n = wi + wGap

                if wi <= w_ < wi_n:
                    fraction = (w_ - wi) / wGap
                    c_ = fraction * (c[i+1] - c[i]) + c[i]
                    return c_

                wi = wi_n

            else:
                return self._upper()
        


    def sample(self):
        """Sample a single value from the distribution represented by the digest.

        Generates a random quantile and returns the corresponding value,
        rounded to the nearest integer.

        Args:
            size (int, optional): Currently unused, always returns a single sample.
                                Defaults to 1.

        Returns:
            int: A sampled integer value from the distribution.

        Note:
            Despite the size parameter, this method currently only returns
            a single sample.
        """

        out:cdouble = cast(cdouble, crand()) / cast(cdouble, C_RAND_MAX)
        
        p: cdouble = crand()
        #p = np.random.rand()
        return int(round(self.quantile(p)))        


    def mean(self):
        pass

    def fit(self, x):
        """Fit the digest to a collection of data points.

        This is a convenience method that adds all points in the collection
        to the digest with equal weight.

        Args:
            x (iterable): Collection of numeric values to add to the digest.
        """
        for xx in x:
            self._add(xx)        


    def __invert__(self):
        """
        Syntactic sugar for the sample method.
        """
        return self.sample()
    
