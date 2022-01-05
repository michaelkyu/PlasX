import heapq
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

import numpy as np
from numba import jit, vectorize, prange, int32, int64, float32, float64

from plasx import utils

"""
Utility functions implemented in numba
"""

# @vectorize(nopython=True)
# def nb_divide(x, y):
#     return x / y

def nb_maximum_py(x, y):
    """Parallelized version of division"""

    z = np.empty(x.size, x.dtype)
    for i in prange(x.size):
        z[i] = max(x[i], y[i])
    return z

def nb_minimum_py(x, y):
    """Parallelized version of division"""

    z = np.empty(x.size, x.dtype)
    for i in prange(x.size):
        z[i] = min(x[i], y[i])
    return z

def nb_divide_py(x, y):
    """Parallelized version of division"""

    z = np.empty(x.size, x.dtype)
    for i in prange(x.size):
        z[i] = x[i] / y[i]
    return z

def nb_add_py(x, y):
    """Parallelized version of add"""

    z = np.empty(x.size, x.dtype)
    for i in prange(x.size):
        z[i] = x[i] + y[i]
    return z

def nb_subtract_py(x, y):
    """Parallelized version of add"""

    z = np.empty(x.size, x.dtype)
    for i in prange(x.size):
        z[i] = x[i] - y[i]
    return z

def nb_cast_py(x, dtype):
    z = np.empty(x.size, dtype)
    for i in prange(x.size):
        z[i] = dtype(x[i])
    return z

def nb_copy_py(x):
    z = np.empty(x.size, x.dtype)
    for i in prange(x.size):
        z[i] = x[i]
    return z

def nb_idx_py(x, y):
    """Parallelized version of indexing"""

    z = np.empty(y.size, x.dtype)
    for i in prange(y.size):
        z[i] = x[y[i]]
    return z

def nb_repeat_py(x,y):
    """Parallelized version of np.repeat"""

    # Cannot use prange to calculating cumsum
    y_sum = 0
    y_cumsum = np.empty(y.size, y.dtype)
    for i in range(y.size):
        y_cumsum[i] = y_sum
        y_sum += y[i]

    z = np.empty(y_sum, x.dtype)
    for i in prange(y.size):
        curr = y_cumsum[i]
        for j in range(y[i]):
            z[curr+j] = x[i]
    return z

for func in ['nb_maximum', 'nb_minimum',
             'nb_divide', 'nb_add', 'nb_subtract',
             'nb_cast',
             'nb_copy',
             'nb_idx',
             'nb_repeat']:
    exec("""{func} = jit({func}_py, nopython=True, nogil=True, fastmath=True, parallel=True)""".format(func=func))
    exec("""{func}_st = jit({func}_py, nopython=True, nogil=True, fastmath=True, parallel=False)""".format(func=func))


# from numba import int32, int64, float32, float64

# #@vectorize(nopython=True, parallel=True)
# @vectorize([float64(int32, int32),
#             float64(int64, int64),
#             float32(float32, float32),
#             float64(float64, float64)], nopython=True, target='parallel')
# def nb_divide2(x, y):
#     """Parallelized version of division"""
#     return x / y
#     # z = np.empty(x.size, x.dtype)
#     # for i in prange(x.size):
#     #     z[i] = x[i] / y[i]


@jit(nopython=True, nogil=True)
def extend_1d_arr(x):
    """Extends a 1d array by twice its size, with random non-empty values
    at the second half

    """
    new_x = np.empty(x.size * 2, x.dtype)
    new_x[ : x.size] = x
    return new_x

@jit(nopython=True, nogil=True)
def extend_2d_arr(x):
    """Extends a 2d array by twice its ROWS, with random non-empty values
    at the second half

    """

    new_x = np.empty((x.shape[0] * 2, x.shape[1]), x.dtype)
    new_x[ : x.shape[0], :] = x
    return new_x

@jit(nopython=True, nogil=True)
def remove_duplicates(x):
    """
    Scans through x and removes duplicates.

    NOTE: Assumes the same x's are grouped consecutively
    """

    x_unique = np.zeros(100000, x.dtype)
    xc = x[0]
    g = 0
    for xx in x:
        if xx != xc:
            if g == x_unique.size:
                x_unique = extend_1d_arr(x_unique)

            x_unique[g] = xc
            xc = xx
            g += 1

    if g == x_unique.size:
        x_unique = extend_1d_arr(x_unique)
    x_unique[g] = xc
    x_unique = x_unique[: g+1]
    return x_unique

@jit(nopython=True, nogil=True)
def remove_duplicates_2d(x):
    """
    Scans through x and removes duplicates.

    NOTE: Assumes the same x's are grouped consecutively
    """

    x_unique = np.zeros((100000, x.shape[1]), x.dtype)
    xc = x[0,:]
    g = 0
    for xi in range(x.shape[0]):
        xx = x[xi,:]
        if (xx != xc).any():
            if g == x_unique.shape[0]:
                x_unique = extend_2d_arr(x_unique)

            x_unique[g, :] = xc
            xc = xx
            g += 1

    if g == x_unique.shape[0]:
        x_unique = extend_2d_arr(x_unique)
    x_unique[g, :] = xc
    x_unique = x_unique[: g+1, :]
    return x_unique

@jit(nopython=True, nogil=True)
def lexsort_nb(a, b):
    """numba implementation of np.lexsort. Taken from https://github.com/numpy/numpy/issues/12755"""
    
    idxs = np.argsort(a, kind="quicksort") 
    return idxs[np.argsort(b[idxs], kind="mergesort")] 

@jit(nopython=True, nogil=True)
def lexsort_nb2(a, first_mergesort=False):
#def lexsort_nb2(a, first_kind='quicksort'):
    """Same as lexsort_nb(), but takes in a k-by-N array (sorting by k
    keys) instead of two keys (a and b) in lexsort_nb()

    first_mergesort : use mergesort for the first key
    """
    
    assert a.shape[0] >= 1
    # For the first sort, you can use either 'quicksort' or 'mergesort'
    # -- quicksort is generally faster if the array is unsorted (default)
    # -- mergesort is generally faster by almost 2x if the array is sorted (I need to do more testing to verify 2x. It seems sometimes that there's no significant speedup.
    if first_mergesort:
        idxs = np.argsort(a[0,:], kind="mergesort")
    else:
        idxs = np.argsort(a[0,:], kind='quicksort')
    
    for i in range(1, a.shape[0]):
        # For subsequent sorts, use mergesort to have stable sorting
        idxs = idxs[np.argsort(a[i,:][idxs], kind="mergesort")]
    return idxs

def lexsort_nb2_py(a, first_mergesort=False):
    """Same as lexsort_nb2() but a python version in order to use numpy's
    native sort

    """
    
    assert a.shape[0] >= 1
    if first_mergesort:
        idxs = np.argsort(a[0,:], kind="mergesort")
    else:
        idxs = np.argsort(a[0,:], kind='quicksort')
    
    for i in range(1, a.shape[0]):
        # For subsequent sorts, use mergesort to have stable sorting
        idxs = idxs[np.argsort(a[i,:][idxs], kind="mergesort")]
    return idxs

def test_lexsort(N=1000000):
    """Test runtime and correctness of numba lexsort implementations"""
    
    a = np.random.rand(N)
    b = np.random.randint(N // 4, size=N)
    ab = np.vstack((a,b))

    from IPython import get_ipython
    ipython = get_ipython()

    print('np.lexsort')
    ipython.magic("timeit np.lexsort([a, b])")
    print('lexsort_nb')
    ipython.magic("timeit lexsort_nb(a, b)")
    print('lexsort_nb2')
    ipython.magic("timeit lexsort_nb2(ab)")

    # %timeit lexsort_nb2(ab)
    # %timeit lexsort_nb(a, b)
    # %timeit np.lexsort([a, b])

    assert (np.lexsort([a, b]) == lexsort_nb(a, b)).all()
    assert (np.lexsort([a, b]) == lexsort_nb2(ab)).all()



@jit(nopython=True, nogil=True)
def get_boundaries(x):
    # Assumes that the same x are grouped consecutively (but not
    # necessarily sorted).  Doesn't assume that y is sorted.
 
    if x.size > 0:
        n = 1000000
        x_unique = np.empty(n, x.dtype)
    
        xc = x[0] # Current x

        boundaries = np.empty(n+1, dtype=np.int32)
        boundaries[0] = 0
        
        g = 0
        for i in range(x.size):
            xx = x[i]
            if xx != xc:
                if g == x_unique.size:
                    x_unique = extend_1d_arr(x_unique)
                    boundaries = extend_1d_arr(boundaries)
                x_unique[g] = xc
                g += 1
                boundaries[g] = i                
                xc = xx            
        if g == x_unique.size:
            x_unique = extend_1d_arr(x_unique)
            boundaries = extend_1d_arr(boundaries)
            
        x_unique[g] = xc
        boundaries[g+1] = x.size

        x_unique = x_unique[: g+1]
        boundaries = boundaries[: g+2]        
    else:
        boundaries = np.empty(0, dtype=np.int32)
        x_unique = np.empty(0, dtype=x.dtype)
        
    return x_unique, boundaries

@jit(nopython=True, nogil=True)
def get_boundaries2(x):
    """Alternative, simpler implementation of get_boundaries()"""
    boundaries = (x[1:] != x[:-1]).nonzero()[0] + 1
    boundaries = np.append(np.append(0, boundaries), x.size)
    #boundaries = np.append(np.append([0], boundaries), [x.size])
    return boundaries

@jit(nopython=True, nogil=True)
def get_boundaries_2d(x):
    """2D version.

    Alternative, simpler implementation of get_boundaries()"""

    # boundaries = np.empty(x.shape[0] + 1, np.int32)
    # for i in range(x.shape[0]):
    #     boundaries[i+1] 

    boundaries = ((x[1:,:] != x[:-1,:]).sum(1) >= 1).nonzero()[0] + 1
    boundaries = np.append(np.append(0, boundaries), x.shape[0])
    return boundaries

def get_boundaries_py(x):
    """Alternative, simpler implementation of get_boundaries()"""
    is_boundary = np.empty(x.size+1, np.bool_)
    n = 2
    for i in prange(1, x.size):
        unequal = x[i] != x[i-1]
        is_boundary[i] = unequal
        n += unequal
    is_boundary[0] = True
    is_boundary[x.size] = True
    
    boundaries = np.empty(n, np.int64)
    j = 0
    for i in range(is_boundary.size):
        if is_boundary[i]:
            boundaries[j] = i
            j += 1

    return boundaries

get_boundaries_mt = jit(get_boundaries_py, nopython=True, nogil=True, parallel=True)
get_boundaries_st = jit(get_boundaries_py, nopython=True, nogil=True, parallel=False)


@jit(nopython=True, nogil=True)
def is_sorted(arr):
    if len(arr.shape)==1:
        return ((arr[1:] - arr[:-1])>=0).all()
    elif len(arr.shape)==2:
        for i in range(arr.shape[0]-1):
            if not lte(arr[i,:], arr[i+1,:]):
                return False
        return True

        # idx = lexsort_nb2(arr.T[::-1,:], first_mergesort=True)
        # return (idx == np.arange(arr.shape[0])).all()
