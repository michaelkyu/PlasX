# Some of the code in this file was modified from the pickle-blosc software at https://github.com/limix/pickle-blosc. The license for pickle-blosc is reproduced below.
#
#
# The MIT License (MIT)
# =====================
#
# Copyright (c) `2018` `Danilo Horta`
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.



import multiprocessing
import pickle as pkl
import blosc
import io
import time
import os

def set_threads(threads=None):
    from plasx import utils

    if threads is None:
        # Get the maximum number of cores on this machine
        threads = utils.get_max_threads()

    blosc.set_nthreads(threads)


def blosc_compress(obj, path_or_buf=None, cname='zstd', shuffle=None, clevel=5, obj_type=None):
    """Compresses a Python object `obj` using the blosc library, and writes to `path_or_buf`
    
    obj_type :

         If 'str', then `obj` is a Python string that is encoded and
         then compressed. If 'pickle', then the object is pickled and
         then compressed. Default: `obj` is assumed to be a binary
         value to be compressed directly

    """
    if shuffle is None:
        shuffle = blosc.NOSHUFFLE

    if obj_type=='str':
        arr = obj.encode()
    elif obj_type=='pickle':
#        arr = pkl.dumps(obj, -1)

        # Set a pickle protocol of 4
        # -- to avoid having to support protocol 3 and below
        # -- and to avoid someone else running a newer version of Python inadvertently pickling with protocol >=5 (which then cannot be read by other users of PlasX)
        arr = pkl.dumps(obj, 4)
    else:
        arr = obj

    start = time.time()
    if path_or_buf:
        with open(path_or_buf, "wb") as f:
            s = 0
            while s < len(arr):
                e = min(s + blosc.MAX_BUFFERSIZE, len(arr))
                carr = blosc.compress(arr[s:e], typesize=8, cname=cname, shuffle=shuffle, clevel=clevel)
                f.write(carr)
                s = e
        pickle_time = time.time() - start
        size = os.path.getsize(path_or_buf)
        return pickle_time, size
    else:
        s = 0
        carr_list = []
        while s < len(arr):
            e = min(s + blosc.MAX_BUFFERSIZE, len(arr))
            carr = blosc.compress(arr[s:e], typesize=8, cname=cname, shuffle=shuffle, clevel=clevel)
            carr_list.append(carr)
            s = e
        carr = b''.join(carr_list)
        return carr        

class BloscReadStream(io.RawIOBase):
    def __init__(self, path_or_buf, verbose=False, bufsize=None, obj_type='file'):
        """
        If path_or_buf is supposed to bytes, and not a file-like object, then there are two options:

        (1) Before calling __init__(), wrap path_or_buf into a file-like object with io.BytesIO(path_or_buf)
        (2) Set obj_type=='bytes'
        """

        if obj_type=='file':
            if isinstance(path_or_buf, io.IOBase):
                self.f = path_or_buf
            else:
                assert os.path.exists(path_or_buf)
                self.f = open(path_or_buf, "rb")
        elif obj_type=='bytes':
            self.f = io.BytesIO(path_or_buf)
        else:
            raise Exception('Unsupported input type: {}'.format(obj_type))
        self.count = 0
        self.read_time = 0
        self.verbose = verbose
        
        if bufsize is None:
            bufsize = int(8e9)
        self.arr = memoryview(bytearray(bufsize))
        self.start = 0
        self.end = 0
        self.cblock = None

    def close(self):
        self.f.close()
        
    def readable(self):
        return True
    
    def read(self, n):
        """TODO: preallocate a buffer for self.f.read(ctbytes), to prevent new
        allocation for every read. The size of this buffer can be
        pre-empted by seeking through all of the headers in the file.

        """

        start = time.time()
        if self.verbose:
            print(time.time(), 'n: %s (%0.2f Gb)' % (n, n / 1e9))
            #print('n: %s (%0.2f Gb)' % (n, n / 1e9))

        if self.verbose:
            print('\tStart:', self.start, 'End:', self.end, 'Buffer size:', len(self.arr))

        if len(self.arr) < n:
            if self.verbose:
                print('\tExtending buffer from %s to %s bytes' % (len(self.arr), n))
                print('\tCurrent start / end:', self.start, self.end)
            newarr = memoryview(bytearray(n))
            newarr[ : self.end - self.start ] = self.arr[self.start : self.end]
            self.arr = newarr
            self.start, self.end = 0, self.end - self.start
        elif (len(self.arr) - self.end) < n:
            if self.verbose:
                print('\tShifting %s bytes' % (self.end - self.start))
            self.arr[ : self.end - self.start] = self.arr[self.start : self.end]
            self.start, self.end = 0, self.end - self.start
        
        while (self.end - self.start) < n:
            # Read the BLOSC 16-byte header. See https://github.com/Blosc/c-blosc/blob/master/README_HEADER.rst
            header = self.f.read(16)
            if len(header) == 0:
                if self.verbose: print('\tEmpty header')
                break
            nbytes = int.from_bytes(header[4:8], "little")
            ctbytes = int.from_bytes(header[12:16], "little")

            if self.cblock is None:
                self.cblock = memoryview(bytearray(ctbytes))
            elif len(self.cblock) < ctbytes:
                self.cblock = memoryview(bytearray(max(ctbytes, 2*len(self.cblock))))

            if len(self.arr) < (self.end + nbytes):
                # Extend the buffer array `self.arr`. Extend by either
                # twice the length of the array or the required length
                # to accomodate the next decompression, whichever is
                # longer
                bufsize = max(2*len(self.arr), self.end+nbytes)
                if self.verbose: print('\tExtending new buffer from %s to %s bytes' % (len(self.arr), bufsize))
                newarr = memoryview(bytearray(bufsize))
                newarr[:self.end] = self.arr[:self.end]
                self.arr = newarr
                
            self.cblock[:16] = header
            self.cblock[16 : ctbytes] = self.f.read(ctbytes - 16)
            self.arr[self.end : self.end + nbytes] = blosc.decompress(self.cblock[: ctbytes])

            self.end += nbytes
            
            self.count += 1
            if self.verbose: print('\tRead iteration:', self.count,
                                   ', compress/decompress block size: %s / %s' % (ctbytes, nbytes))

        new_start = self.start + n
        if self.end < new_start:
            if verbose:
                print('\tRequested {:n} bytes, but only sending back {:n} bytes'.format(n, self.end - self.start))
            
            # I think this should never happen, because pickle should
            # not over-request more bytes than is in the actual object
            # (I assume the pickle format knows about the total data
            # size)
            raise Exception('Check if this is proper behavior and okay, or if it indicates a bug')
            new_start = self.end
        
        retval = self.arr[self.start : new_start]
        self.start = new_start
        
        delta_time = time.time() - start
        if self.verbose: print('\tRead and decompress time:', delta_time)
        self.read_time += delta_time
        
        return retval

    def readinto(self, b):
        raise(NotImplementedError)
        
    def readline(self, size):
        raise(NotImplementedError)


def blosc_decompress(path_or_buf, stream='auto', obj_type=None, verbose=False):
    """
    Decompress the file `path_or_buf`, which is assumed to be created by the function `blosc_compress`

    There was a bug in pickle-blosc on reading the number of bytes in the header. I've fixed it.
    """


    if stream=='auto':
        if not isinstance(path_or_buf, bytes) and os.path.exists(path_or_buf) and (os.path.getsize(path_or_buf) >= 32e9):
            # Stream only if this is a file and is >=32GB in size
            stream = True
        else:
            stream = False

    if isinstance(path_or_buf, bytes):
        path_or_buf = io.BytesIO(path_or_buf)
    
    if stream:
        assert obj_type=='pickle'
        return pkl.load(BloscReadStream(path_or_buf, verbose=verbose))
    else:
        arr = []
        
        try:
            f = open(path_or_buf, "rb")
            opened = True
        except:
            f = path_or_buf
            opened = False
        finally:
            while True:
                # Read the BLOSC 16-byte header. See https://github.com/Blosc/c-blosc/blob/master/README_HEADER.rst
                header = f.read(16)
                if len(header) == 0:
                    break
                ctbytes = int.from_bytes(header[12:16], "little")            
                carr = header + f.read(ctbytes - 16)
                arr.append(blosc.decompress(carr))
            if opened:
                f.close()
        
        if obj_type=='str':
            return (b"".join(arr)).decode()
        elif obj_type=='pickle':
            return pkl.loads(b"".join(arr))
        else:
            return b"".join(arr)

def pickle(obj, path_or_buf, cname='zstd', shuffle=None, clevel=5, compress=True, makedirs=False, threads=None):
    if makedirs:
        os.makedirs(os.path.dirname(path_or_buf), exist_ok=True)

    set_threads(threads)

    if compress:
        return blosc_compress(obj, path_or_buf, cname='zstd', shuffle=None, clevel=clevel, obj_type='pickle')
    else:
        with open(path_or_buf, 'wb') as f:
            pkl.dump(obj, f)

def unpickle(path_or_buf, stream='auto', verbose=False, keys=None, threads=None):

    set_threads(threads)

    ret =  blosc_decompress(path_or_buf, stream=stream, obj_type='pickle', verbose=verbose)

    import gc ; gc.collect()

    def iterable_get(ret, key_list):
        x = ret
        for k in key_list:
            x = x[k]
        return x

    if keys is not None:
        if isinstance(keys, str):
            keys = [keys]

        # If `keys` is specified, then the data is assumed to be a
        # dictionary. Return the values in the dictionary at `keys`
        ret = [iterable_get(ret, k) if isinstance(k, list) else ret[k]  for k in keys]

    import gc ; gc.collect()

    return ret

def write_compressed_txt(obj, path_or_buf, cname='zstd', shuffle=None, clevel=5):
    return blosc_compress(obj, path_or_buf, cname='zstd', shuffle=None, clevel=clevel, obj_type='str')

def read_compressed_txt(path_or_buf):
    return blosc_decompress(path_or_buf, stream='auto', obj_type='str')
