import numpy as np

def local_extrema_1d(y, x=None, tau=0.0, return_plateaus=False):
    """
    Find local maxima/minima indices for 1D sampled data y[i]=f(x[i]).

    Parameters
    ----------
    y : array-like, shape (n,)
        Sample values on grid.
    x : array-like, shape (n,), optional
        Grid points (not used for computation here; only returned if provided).
    tau : float, default 0.0
        Threshold for near-zero differences to suppress tiny noise.
        Differences with abs(d) <= tau are treated as 0.
    return_plateaus : bool, default False
        If True, return plateau ranges (start,end) for flat extrema;
        otherwise return a single representative index (midpoint) for each plateau.

    Returns
    -------
    maxima, minima : np.ndarray
        Indices of local maxima and minima (midpoints if plateau and return_plateaus=False).
    (optional) max_plateaus, min_plateaus : list[tuple[int,int]]
        Plateau ranges [start,end] (inclusive), if return_plateaus=True.
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    if n < 3:
        return np.array([], dtype=int), np.array([], dtype=int)

    # First difference
    d = np.diff(y)

    # Apply threshold: small diffs -> 0
    if tau > 0:
        d[np.abs(d) <= tau] = 0.0

    # Convert to sign: -1, 0, +1
    s = np.sign(d).astype(int)

    maxima = []
    minima = []
    max_plateaus = []
    min_plateaus = []

    i = 0
    # We analyze sign changes of d around points/plateaus.
    # A plateau corresponds to a run of s==0.
    while i < len(s):
        if s[i] != 0:
            i += 1
            continue

        # Find run of zeros: [i, j] in s
        j = i
        while j < len(s) and s[j] == 0:
            j += 1
        # Now s[i:j] are zeros. This means y is flat from index i to j in terms of differences.
        # The plateau in y is from p_start to p_end:
        p_start = i
        p_end = j  # because diff index k corresponds to y[k]->y[k+1]
        # plateau y indices are [p_start, p_end]
        # Need signs just before and after the plateau run:
        left_sign = s[i-1] if i-1 >= 0 else 0
        right_sign = s[j] if j < len(s) else 0

        # Plateau maximum: + then -
        if left_sign > 0 and right_sign < 0:
            if return_plateaus:
                max_plateaus.append((p_start, p_end))
            else:
                maxima.append((p_start + p_end) // 2)

        # Plateau minimum: - then +
        if left_sign < 0 and right_sign > 0:
            if return_plateaus:
                min_plateaus.append((p_start, p_end))
            else:
                minima.append((p_start + p_end) // 2)

        i = j

    # Non-plateau extrema: sign change at a single point
    # maxima at i if s[i-1] > 0 and s[i] < 0  (i refers to diff index, point is i)
    # minima at i if s[i-1] < 0 and s[i] > 0
    for k in range(1, len(s)):
        if s[k-1] > 0 and s[k] < 0:
            maxima.append(k)
        elif s[k-1] < 0 and s[k] > 0:
            minima.append(k)

    maxima = np.array(sorted(set(maxima)), dtype=int)
    minima = np.array(sorted(set(minima)), dtype=int)

    # Remove boundary if accidentally included (shouldn't, but safe)
    maxima = maxima[(maxima > 0) & (maxima < n-1)]
    minima = minima[(minima > 0) & (minima < n-1)]

    if return_plateaus:
        return maxima, minima, max_plateaus, min_plateaus
    return maxima, minima

