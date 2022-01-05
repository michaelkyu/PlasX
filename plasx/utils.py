import sys
import subprocess
import os
import time
import tempfile
import shutil
import shlex
import multiprocessing
from datetime import datetime
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import numpy as np
import scipy.sparse

from plasx import nb_utils
from plasx.fasta_utils import *
from plasx.pd_utils import *
from plasx.compress_utils import *

def get_max_threads():
    return multiprocessing.cpu_count()

def multiline(df, sep='|', newsep='\n', inplace=False):
    """Attempts to replace a separator string (default: '|') with
whitespace (default: '\n') in every column of a dataframe"""

    if not inplace:
        df = df.copy()

    for c in df.columns:
        try:
            df[c] = df[c].str.replace(sep, newsep)
        except:
            pass

    return df

def pretty_print(df,
                 newlines=True,
                 col_newlines=True,
                 max_rows=None,
                 max_columns=None,
                 max_colwidth=None,
                 transpose=True,
                 precision=3,
                 float_format=None,
                 multi=False,
                 simple=None,
                 vcols=None,
                 vertical_cols=None):
    """More human-viewable display of pd.DataFrame and numpy arrays.

    newlines : 

        If True, then set pandas style to display whitespace as newlines

    transpose :

        If True, then show pd.Series as a 1-by-* DataFrame.  This is
        sometimes easier to view, especially if displaying multiple
        pd.Series

    """

    from IPython.display import display, HTML   

    # Set numpy display config
    np.set_printoptions(precision=2, linewidth=200)

    if max_rows is None:
        max_rows = pd.options.display.max_rows

    if max_columns is None:
        # The default 'pd.options.display.max_columns' is actually 20,
        # but I find that I typically want to view more columns, so I set it 50
        # here
        max_columns = 75

    if max_colwidth is None:
        max_colwidth = 150
    elif max_colwidth=='default':
        max_colwidth = pd.options.display.max_colwidth        

    if isinstance(float_format, str):
        float_format = lambda x: float_format % x

    if isinstance(df, pd.Series) and transpose:
        # Convert pd.Series to a 1-by-* pd.DataFrame
        df = df.to_frame().T

    if multi:
        df = multiline(df)

    if (vertical_cols is None) and (vcols is not None):
        vertical_cols = vcols
        
    with pd.option_context("display.max_rows", max_rows):
        with pd.option_context("max_colwidth", max_colwidth):
            with pd.option_context("display.max_columns", max_columns):
                with pd.option_context("precision", precision):
#                    with pd.option_context('display.float_format', float_format):
                        # newlines==True is only applicable when `df` is a DataFrame (because pd.Series has no property `style`)
                        if newlines and isinstance(df, pd.DataFrame):
                            if col_newlines:
                                df = df.rename(columns=lambda x: '\n'.join(x.replace('_',' ').split()) if isinstance(x, str) else x)

                            if simple:
                                # Filter for columns that contain scalar values, e.g. no tuples, lists, etc.
                                # Also, filter out strings that are >1000 characters
                                df = df.loc[:, [all([not hasattr(x, '__iter__') or (isinstance(x, str) and len(x) < 1000) for x in df[c]]) for c in df.columns]]

                            # Ipython bug: max_rows is not adhered to by
                            # Ipython.display() when dataframe style is set.
                            # Fix: Do a head()
                            df_head = df.head(max_rows)

                            # If index is categorical, then change to the original dtype for faster display
                            if len(df_head.index.names)==1 and df.index.dtype.name=='category':
                                df_head = df_head.copy()
                                df_head.index = np.array(df_head.index)

                            if vertical_cols:
                                df_head = df_head.style.set_table_styles(
                                    [dict(selector="th",props=[('max-width', '80px')]),
#                                    [dict(selector="th",props=[('max-width', '40ch')]),
                                     dict(selector="th.col_heading",
                                          props=[("writing-mode", "vertical-rl"), 
                                                 ('transform', 'rotateZ(180deg)'),
                                                 ('vertical-align', 'top'),
    #                                             ('white-space', 'pre-wrap'),
                                             ])]
                                )

                            else:
#                                df_head = df_head.set_properties(**{'white-space': 'pre-wrap',})
                                df_head = df_head.style.set_properties(**{'white-space': 'pre-wrap',})


                            display(df_head)

                            # df_tail = df.tail(max_rows)
                            # df_tail = df_tail.style.set_properties(**{'white-space': 'pre-wrap',})
                            # display(df_tail)
                        else:
                            display(df)

                        # if newlines:
                        #     # Need to manually do a pd.DataFrame.head() because the
                        #     # pd.option_context doesn't really do much as conversion
                        #     # to HTML is done first
                        #     max_rows = pd.options.display.max_rows
                        #     display( HTML( df.head(max_rows).to_html().replace("\\n","<br>") ) )
                        # else:
                        #     display( df )
                    
    # Reset numpy display config
    np.set_printoptions()

def pprint(df, **kwargs):
    pretty_print(df, **kwargs)

def run_cmd(cmd, tee=True, shell=True, env=None, wait=True, debug=False, verbose=False, **kwargs):
    """Runs a command, and returns the output live.

    If stdout and stderr are not specified, then they are piped by default, and printed live
    """

    if env:
        cmd = utils.make_cmd(cmd, env=env)

    if debug or verbose:
        print('Running command:')
        print(cmd)
        if debug:
            return

#    if ('shell' not in kwargs) or (('shell' in kwargs) and (not kwargs['shell'])):
    if not shell:
        cmd = shlex.split(cmd)

    if 'stdout' not in kwargs:
        kwargs['stdout'] = subprocess.PIPE
    if 'stderr' not in kwargs:
        kwargs['stderr'] = subprocess.STDOUT

    p = subprocess.Popen(cmd, shell=shell, **kwargs)

    if tee:
        # Live stream the output
        # From https://stackoverflow.com/a/18422264
        for line in iter(p.stdout.readline, b''):  # replace '' with b'' for Python 3
#            sys.stdout.write(line)
            sys.stdout.write(line.decode())
            #f.write(line)
    else:
        if wait:
            p.wait()

    if tee or wait:
        p.wait()
        poll = p.poll()
        if poll != 0 :
            print('poll:', poll)
        assert poll==0, 'Did not successfully run command: {}'.format(cmd)
    return p

def make_cmd(*cmd_list, env=None, sep='&&\\\n'):
    """Formats a set of commands that's suitable to run with
    `subprocess`. Handles two tasks:

    (1) Automatically adds commands to initialize a conda virtual
        environment (e.g. anvio-master-zelda) in order to have the
        correct paths.

    (2) Split multiple commands by newlines and the '&&' connector.

    """
    
    if isinstance(cmd_list, str):
        cmd_list = [ cmd_list ]
    else:
        cmd_list = list(cmd_list)

    if env is None:
        pass
    else:
        cmd_list.insert(0, 'conda activate {}'.format(env))

        # Constants for loading conda environments in a subprocess
        conda_local = 'eval "$(/scratch/miniconda/bin/conda shell.bash hook)" && conda deactivate'

        cmd_list.insert(0, conda_local)

    return (' ' + sep).join(cmd_list)

def eprint(i, length, prefix='', suffix='', print=True):
    """
    Prints out an loop enumeration statement, e.g.

        Sample: 1 out of 122: USA0001_01 (02:16:40 PM)
    """

    string = '{prefix}: {i} out of {length}: {suffix}'.format(prefix=prefix, i=i+1, length=length, suffix=suffix)
    if print:
        time_print(string)
    else:
        return string

def tprint(*x, day=False, second=True, end='\n', verbose=None):
    if (verbose is None) or verbose:
        time_print(*x, day=day, second=second, end=end)
    
def time_print(*x, day=False, second=True, end='\n'):
    fmt = []
    if day: fmt.append('%x')
    if second: fmt.append('%X')
    fmt = ' '.join(fmt)

    # if len(x)==1: x = x[0]
    # # If a string, then re-incase it as a list, to prevent individual
    # # characters being passed
    # if isinstance(x, (str, np.str_)): x = [x]

    print(*x, '(%s)' % datetime.now().strftime(fmt), end=end)
    sys.stdout.flush()

@contextmanager
def silence_output(stdout=True, stderr=True):
    """Silences stdout and/or stderr with a context, e.g.

        with silence_stdout():
            print("will not print")

        print("this will print")

    Code inspired from
    https://stackoverflow.com/questions/6735917/redirecting-stdout-to-nothing-in-python

    """
    
    devnull = open(os.devnull, "w")
    if stdout:  old_stdout = sys.stdout
    if stderr:  old_stderr = sys.stderr
        
    try:
        if stdout: sys.stdout = devnull
        if stderr: sys.stderr = devnull
        yield devnull
    finally:
        if stdout: sys.stdout = old_stdout
        if stderr: sys.stderr = old_stderr

@contextmanager
def redirect_output(stdout=None, stderr=None):    
    try:
        old_stdout = sys.stdout
        old_stderr = sys.stderr

        if stdout: sys.stdout = stdout
        if stderr: sys.stderr = stderr
        yield 0
    finally:
        if stdout: sys.stdout = old_stdout
        if stderr: sys.stderr = old_stderr


def rearrange(df, columns, indices, start=None):
    """
    Rearranges columns in a dataframe, so that `columns` occurs at the column indices specified by `indices`
    """

    if isinstance(columns, str):
        columns = [columns]
        indices = [indices]

    # if isinstance(indices, int) or indices=='len':
    #     indices = [indices for _ in range(len(columns))]
    if isinstance(indices, int) and (start is True):
        indices = np.arange(indices, indices + len(columns))

    new_columns = [c for c in df.columns if c not in columns]
    for c, i in zip(columns, indices):
        if i=='len':
            new_columns.append(c)
        else:
            new_columns.insert(i, c)
    return df[new_columns]

def as_flat(x):
    if scipy.sparse.issparse(x) and (x.nnz==0):
        # Need to handle the special case when the input `x` is 1-by-0 scipy sparse matrix.
        # - This happens when you index a scipy matrix with empty indices, e.g. sp[[],[]].
        # - Solution: manually create an empty 1-D array with the same dtype
        return np.array([], dtype=x.dtype)
    else:
        return np.array(x).flatten()



def downcast_uint(x):
    """Convert an array so that 
    Get the smallest uint data type to downcast np.ndarray `x`, without clipping values"""
    
    assert np.issubdtype(x.dtype, np.integer) or np.issubdtype(x.dtype, np.bool_)

    if x.size==0:
        # This is an empty array, so use np.uint8
        return x.astype(np.bool)

    max_val = x.max()
    
    # Check that value is non-negative
    assert max_val >= 0
      
    if max_val <= 1:
        return x.astype(np.bool)
    elif max_val <= (2**8 - 1):
        return x.astype(np.uint8)
    elif max_val <= (2**16 - 1):
        return x.astype(np.uint16)
    elif max_val <= (2**32 - 1):
        return x.astype(np.uint32)
    elif max_val <= (2**64 - 1):
        return x.astype(np.uint64)

    

@contextmanager
def TemporaryDirectory(name=None, post_delete=None, path=True, verbose=False,
                       overwrite=None, overwrite_err_msg=None):
    """Creates a context manager that can be invoked with the paradigm
    
    with TemporaryDirectory() as f:
        <code>

    The advantage of this function over tempfile.TemporaryDirectory()
    is that this function can take in a pre-specified directory
    name. In that case, the directory is not deleted at the end of the
    `with` statement, unless post_delete=True.

    If a pre-specified directory name is given, then that directory is
    created, if it doesn't already exist.

    If path=True, then return a pathlib.Path instance (instead of a
    string)

    overwrite : If False, then check if file already exists, and raise Exception if so. Default: True.

    pre_delete : If True, then first delete the directory if it already exists. Default: False
    """

    try:
        if name is None:
            delete = True
            name = tempfile.mkdtemp()
            if verbose:
                print('Created temporary directory:', name)
        else:
            check_overwrite(name, overwrite, overwrite_err_msg)
            
            os.makedirs(name, exist_ok=True)
            delete = False
            
        if path:
            name = Path(name)

        yield name

    finally:
        if (post_delete is True or delete) and (name is not None):
            shutil.rmtree(name)

def check_overwrite(path, overwrite=None, overwrite_err_msg=None):
    """
    overwrite has three modes (Default: 0)
        
        0 : Don't do anything. Just let the file/directory be overwritten however.
        1 : Check if file at location 'path' already exists, and raise Exception if so.
        2 : Check if file at location 'path' already exists, and delete it if so.
    """

    if overwrite is None:
        overwrite = 0
    assert overwrite in [0,1,2]

    if (overwrite != 0) and os.path.exists(path):
        if overwrite==1:
            if overwrite_err_msg is None:
                overwrite_err_msg = f"Attempting to create file or directory at {path}, but it already exists. Delete it, or set overwrite to True."
            raise Exception(overwrite_err_msg)

        elif overwrite==2:
            if os.path.isfile(path):
                os.remove(path)
            else:
                shutil.rmtree(path)


def subset_dict(d, keys):
    """Return subset of dictionary"""

    return {k : d[k] for k in keys}
