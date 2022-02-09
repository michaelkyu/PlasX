import os
import operator
import functools
import time
import numpy as np
import pandas as pd
import scipy
from pathlib import Path

from plasx import utils

def catin(series, domain, keep_index=False):
    """Alias for categorical_isin()"""
    
    return categorical_isin(series, domain, keep_index=keep_index)
    
def categorical_isin(series, domain, keep_index=False):
    """Does a faster version of pd.Series.isin() for categorical
    dtypes. Does this by only checking the unique categories

    Calculates the equivalent of `series.isin(domain)`

    """

    isin = series.cat.categories.isin(domain)[series.cat.codes.values]
    if keep_index:
        pd.Series(isin, index=series.index)
    else:
        return isin

def int_loc(query, domain, check=True):
    """Fast querying into `domain`, a pd.Series with an integer index. Querying at locations in the numpy array `query`.

    The equivalent of domain.loc[query]

    The implementation trick is to create an array with values located at the domain's index.

    Disadvantage: requires creating a very big array

    If check=True, then do some extra processing to validate the input
    """

    if len(domain.index) > 0:
        min_val = domain.index.min()        
        max_val = domain.index.max()
    else:
        min_val, max_val = 0, 0

    if check:
        assert pd.api.types.is_integer_dtype(query), f"query was dtype {type(query)}, but it needs to be an integer dtype"
        assert pd.api.types.is_integer_dtype(domain.index), f"domain was dtype {type(domain)}, but it needs to be an integer dtype"
        assert min_val >= 0, f"Minimum value of domain was {min_val}, but it needs to be >=0"
        assert isinstance(query, np.ndarray)
        assert (max_val <= 10**9), \
            "The max value in this domain's index is above 1 billion, which will require creating an array with memory >8gb. Make sure you are okay with this and then set check=False"
        assert domain.index.duplicated().sum()==0 # Check that the domain' index does not have duplicate values
        assert np.all(isin_int(query, domain.index, max_val=max_val)) # Check that all of query is in the domain

    arr = np.empty(max_val + 1, domain.dtype)
    arr[domain.index] = domain.values
    return arr[query]

def isin_int(series, domain, max_val=None):
    """Fast np.in1d when series and domain are both integers.

    Equivalent to np.in1d(series, domain)

    The implementation trick is to create a boolean presence/absence array.
    """

    if max_val is None:
        if len(series) > 0:
            max_val = np.max(series)
        else:
            max_val = 0

    mask = np.zeros(max_val + 1, np.bool_)
    mask[domain[domain <= max_val]] = True

    return mask[series]

def better_pd_concat(df_list, sort=True, **kwargs):
    """Does the same as pandas.concat(), but categorical columns will ALWAYS be preserved as categorical

    The (bad) behavior of pandas.concat() is to return an object dtype
    if the list of Series have different 'categories'.

    """

    from pandas.api.types import union_categoricals

    if all(isinstance(x, pd.Series) for x in df_list):
        # Everything is a pd.Series

        df_concat = pd.concat(df_list, ignore_index=True)
        df_concat.index = union_categoricals([x.index for x in df_list])

        assert len(df_concat) == sum(map(len, df_list))
        return df_concat
    else:
        # Check that the columns across the dataframes are all the same (and same order)
        try:
            for i in range(len(df_list) - 1):
                assert (df_list[i].columns == df_list[
                    i + 1].columns).all(), 'Columns are not in the same order across dataframes. Suggest re-running with sort=True.'
        except AssertionError as e:
            if sort:
                df_list = [df[sorted(df.columns)] for df in df_list]
            else:
                raise e

        columns = df_list[0].columns
        dtypes = df_list[0].dtypes

        non_categ_columns = [c for c in columns if dtypes[c].name != 'category']
        categ_columns = [c for c in columns if dtypes[c].name == 'category']

        df_concat = pd.concat([df[non_categ_columns] for df in df_list], **kwargs)

        for c in categ_columns:
            df_concat[c] = union_categoricals([df[c] for df in df_list])

        # Use original ordering of columns
        df_concat = df_concat[columns]

        return df_concat


def sync_categories(to_sync):
    """For an input set of categorical series, set their categories to be
    the same (the union of categories across all series).

    Does modifications inplace.

    to_sync : list-like of pandas Categorical Series

    """

    import functools
    union_categories = functools.reduce(lambda x, y: x.union(y),
                                        [x.cat.categories for x in to_sync])

    # Update 12/19/21: Need to return synced categoricals as new
    # objects, as inplace=True has become deprecated in newer pandas
    # versions
    return [x.cat.set_categories(union_categories) for x in to_sync]

    # for x in to_sync:
    #     x.cat.set_categories(union_categories, inplace=True)

    # return to_sync


def merge(left, right, **kwargs):
    """Alias for better_merge()"""
    return better_merge(left, right, **kwargs)

def better_merge(left, right, stepwise=None, **kwargs):
    """Does the same as pandas.merge(), but categorical columns that are
    joined will be first converted into numerical codes before
    joining. After joining, the codes will be converted back into
    categoricals.

    In this process, the ordering of categories may be messed up.

    The (bad) behavior of pandas.merge() is to return an object dtype
    for the joined categorical columns.

    """

    if stepwise:
        ### If merging on multiple columns, then do a groupby on the
        ### first column, and then merge.

        on = kwargs['on']
        assert 'left_on' not in kwargs and 'right_on' not in kwargs
        key_list = np.asarray(pd.concat([left[on[0]], right[on[0]]], ignore_index=True).drop_duplicates())
        # left_dict = {a:b for a,b in left.groupby(on[0], observed=True)}
        # right_dict = {a:b for a,b in right.groupby(on[0], observed=True)}
        utils.tprint('Creating dictionaries')
        left_dict = {a:b for a,b in utils.subset(left, {on[0]:key_list}, unused=False).groupby(on[0], observed=True)}
        right_dict = {a:b for a,b in utils.subset(right, {on[0]:key_list}, unused=False).groupby(on[0], observed=True)}

        utils.tprint('Creating sub merges')
        to_concat = []
        assert 'how' in kwargs and kwargs['how']=='inner'
        for key in set(left_dict.keys()) & set(right_dict.keys()):
            to_concat.append(left_dict[key].merge(right_dict[key], on=on[1:], **kwargs))

        utils.tprint('Concating')
        ret = pd.concat(to_concat)
        utils.tprint('Done')
        return ret

    left_on, right_on = kwargs.get('left_on', None), kwargs.get('right_on', None)
    if kwargs['on']:
        left_on = kwargs['on']
        right_on = kwargs['on']

    if isinstance(left_on, str):
        left_on = [left_on]
    if isinstance(right_on, str):
        right_on = [right_on]

    if left_on and right_on:
        left = left.copy()
        right = right.copy()

        new_categories_dict = {}

        for left_col, right_col in zip(left_on, right_on):

            if str(left[left_col].dtype.name) == 'category':
#                print(left_col, right_col)
                assert right[right_col].dtype.name == 'category'

                categories_union = left[left_col].cat.categories.union(right[right_col].cat.categories)
                left[left_col] = left[left_col].cat.set_categories(categories_union)
                right[right_col] = right[right_col].cat.set_categories(categories_union)

                left[left_col] = left[left_col].cat.codes
                right[right_col] = right[right_col].cat.codes

                new_categories_dict[left_col] = categories_union
                new_categories_dict[right_col] = categories_union

    elif left_on:
        raise Exception()
    elif right_on:
        raise Exception()

    # return left, right, categories_union

    # Do merge
    start = time.time()
    merge_df = pd.merge(left, right, **kwargs)
    print('Merge_time:', time.time() - start)
    print('Merge shape:', merge_df.shape)

    # Change numerical codes back into categoricals
    for col, new_categories in new_categories_dict.items():
#        merge_df[col] = pd.Categorical.from_codes(merge_df[col], categories=new_categories)
        merge_df[col] = pd.Categorical.from_codes(merge_df[col].values, categories=new_categories)

    return merge_df


def dask_merge(left, right, compute=True, left_splits=None, right_splits=None, **kwargs):
    """Does pandas.merge, but uses dask. Splits pandas dataframes into dataframes automatically"""

    # pandas.merge(left, right, how='inner', on=None, left_on=None, right_on=None, left_index=False, right_index=False, sort=False, suffixes=('_x', '_y'), copy=True, indicator=False, validate=None)[source]

    # TODO: make the default splits be the squareroot of the number of
    # CPUs (since merging is number of left splits * number of right
    # splits)
    if left_splits is None:
        left_splits = 4
    if right_splits is None:
        right_splits = 4

    # Convert into dask dataframes
    # print(type(left), type(right))
    if not isinstance(left, dd.core.DataFrame):
        left = dd.from_pandas(left, npartitions=left_splits)
    if not isinstance(right, dd.core.DataFrame):
        right = dd.from_pandas(right, npartitions=right_splits)

    merge = left.merge(right, **kwargs)
    if compute:
        merge = merge.compute()
    return merge


def df_2_long_df(sp, dic=None, colname='i', rowname='j', dataname='data', how='outer'):
    """Same as spmatrix_2_long_df, but takes in dataframes as input"""

    """TODO : merge this formatting code (converting to dictionary) with spmatrix_2_long"""
    try:
        # Test if `sp` is a dictionary, by seeing if it has .keys() attribute
        sp.keys()
        sp_dict = sp
    except:
        if isinstance(sp, (list, tuple)):
            # 'data_1', 'data_2', etc.
            sp_dict = {'%s_%s' % (dataname, i): x for i, x in enumerate(sp)}
        else:
            # 'data' --> sp
            sp_dict = {dataname: sp}

    if dic is None:  dic = {}
    dic['name'] = list(sp_dict.values())[0].index.values

    # Convert dataframes to COO matrices
    sp_dict = {k: dense_to_sparse_df(v).sparse.to_coo() for k, v in sp_dict.items()}

    return wide_2_long_df(sp_dict,
                          dic=dic,
                          colname=colname,
                          rowname=rowname,
                          dataname=dataname,
                          how=how)


def dense_to_sparse_df(df, fill_value=None):
    """Cast columns in a pd.DataFrame to Sparse dtypes.

    If `fill_value` is not specified, then assume default fill_values
    for each numerical dtype (e.g. nan for np.float)

    """

    def to_sparse_dtype(s, fill_value=None):
        """Changes a pandas dtype into its sparse version. If the dtype is
        already sparse, then don't do anything

        """

        if 'Sparse' in s.name:
            return s
        else:
            if fill_value is not None:
                # Convert the fill_value into the specific numpy
                # dtype. E.g. if fill_value==0 (a Python integer) and
                # dtype is np.float64, then the fill_value gets
                # converted to np.float64(0)
                fill_value = s.type(fill_value)
            return pd.SparseDtype(s, fill_value=fill_value)

    # return df.astype({k : pd.SparseDtype(v, fill_value=fill_value) for k, v in df.dtypes.items()})
    return df.astype({k: to_sparse_dtype(v, fill_value=fill_value) for k, v in df.dtypes.items()})


def wide_2_long_df(sp, dic=None, rowname='i', colname='j', dataname='data', how='outer'):
    """Return a long-format dataframe that shows only the non-zero values
    of a scipy sparse matrix. If multiple matrices are given, then the
    intersection or union of non-zero values are given.  """

    try:
        # Test if `sp` is a dictionary, by seeing if it has .keys() attribute
        sp.keys()
        sp_dict = sp
    except:
        if isinstance(sp, (list, tuple)):
            # 'data_1', 'data_2', etc.
            sp_dict = {'%s_%s' % (dataname, i): x for i, x in enumerate(sp)}
        else:
            # 'data' --> sp
            sp_dict = {dataname: sp}

    # Convert numpy arrays and dataframes into COO matrices
    for k, v in list(sp_dict.items()):
        if isinstance(v, np.ndarray):
            sp_dict[k] = scipy.sparse.coo_matrix(v)
        elif isinstance(v, pd.DataFrame):
            if dic is None:  dic = {}
            if 'name' not in dic:
                dic['name'] = k.index.values

            sp_dict[k] = dense_to_sparse_df(v).sparse.to_coo()

    if how == 'outer':
        i, j = sum([sp > 0 for sp in sp_dict.values()]).nonzero()
    if how == 'inner':
        from functools import reduce
        tmp = reduce(lambda a, b: a.multiply(b), (x > 0 for x in sp in sp.values()))
        tmp.eliminate_zeros()
        i, j = tmp.nonzero()

    df = pd.DataFrame({sp_name: utils.as_flat(utils.sp_as_indexable(sp)[i, j]) \
                       for sp_name, sp in sp_dict.items()})

    if dic is None:
        df[rowname] = i
        df[colname] = j
    else:
        for k, v in dic.items():
            # categorical_from_nonunique is needed, because the same
            # description can be shared by multiple COG accesions,
            # e.g. the description "uncharacterized protein"
            df[rowname + '_' + k] = categorical_from_nonunique(i, v)
            df[colname + '_' + k] = categorical_from_nonunique(j, v)

    return df


def categorical_from_nonunique(codes, categories):
    """If `categories` is an array of non-unique values, then
    pd.Categorical.from_codes() won't work out of the box.

    This function will create a pd.Categorical by collapsing redundant
    categories and represent them with a single code.

    This involves changing the codes to new codes.

    """

    # if categories.dtype == np.dtype('O'):
    #     # If `categories` is an array mix of dtypes, e.g. ints, nans,
    #     # and strings, then its dtype is np.dtype('O'). In this case,
    #     # running np.unique will throw an error because it can't compare strings vs integers for example.
    #     unique_categories = np.array(list(set(categories)))
    # else:
    #     #unique_categories = np.unique(categories)
    #     unique_categories = pd.Series(categories).drop_duplicates().values

    unique_categories = pd.Series(categories).drop_duplicates().values

    # Maps unique categories to consecutive integers: 1,2,3,...
    unique_categories_index = pd.Series(np.arange(unique_categories.size), index=unique_categories)

    # reindex[i] = new code for code `i`
    reindex = unique_categories_index[categories].values
    new_codes = reindex[codes]

    return pd.Categorical.from_codes(new_codes, categories=unique_categories)


def sparse_melt(sp, rownames=None, colnames=None, index_name=None, var_name=None, value_name=None, square=None):
    if index_name is None:
        index_name = 'index'
    if var_name is None:
        var_name = 'column'
    if value_name is None:
        value_name = 'value'
    if square is None:
        square = False

    if square:
        assert colnames is None, "You should only specify `rownames`, if square=True"
        colnames = rownames
        
    edges = sp.nonzero()
    edges_i, edges_j = edges
    df = pd.DataFrame({index_name : edges_i if (rownames is None) else pd.Categorical.from_codes(edges_i, rownames),
                       var_name : edges_j if (colnames is None) else pd.Categorical.from_codes(edges_j, colnames),
                       value_name : utils.as_flat(sp[edges_i, edges_j])})
    return df

def sparse_pivot(df,
                 index=None,
                 columns=None,
                 values=None,
                 index_use_index=False,
                 columns_use_index=False,
                 fill_value=0.0,
                 offset_hack=None,
                 square=False,
                 remove_unused_categories=False,
                 directed=None,
                 binary=None,
                 verbose=False,
                 method=None,
                 default_value=None,
                 aggfunc=None,
                 rettype='sparse'):
    """Turns a long-format DataFrame into a wide-format sparse
    DataFrame. Does same functionality as pd.pivot, but returns a
    sparse (rather than normal) DataFrame
    
    df :
    
        dataframe
    
    index, columns, values:
    
        Columns in `df` to be pivoted. Same as the parameters for pd.DataFrame.pivot(). Cannot be `None`.
        
        `index` and `columns` should be categorical Series. `values` should be numerical Series.

    rettype:

        'sparse' : pd.DataFrame with Sparse series

        'spmatrix' : scipy sparse COO matrix

    square :

        If True, then reorder the category code values of `index` and `columns` to be the same

    """

    assert (index_use_index + columns_use_index) != 2, 'Cannot use index twice'


    if method=='pd':
        # Use pandas.pivot_table, but do some preformatting to make things faster
        
        assert rettype=='dense_df'

        df = df.copy()

        # Change `index` and `columns` into a list
        if isinstance(index, str) or (not hasattr(index, '__iter__') and index in df.columns):
            index = [ index ]
        if isinstance(columns, str) or (not hasattr(index, '__iter__') and columns in df.columns):
            columns = [ columns ]

        # Convert categories into codes (This is the biggest help in making the code run fast)
        def convert_cat_to_codes(columns):
            categories = []
            for c in columns:
                if df[c].dtype.name=='category':
                    if remove_unused_categories:
                        df[c] = df[c].cat.remove_unused_categories()
                    categories.append(df[c].cat.categories)
                    df[c] = df[c].cat.codes
                else:
                    categories.append(None)
            return categories
        index_categories = convert_cat_to_codes(index)
        column_categories = convert_cat_to_codes(columns)

        if default_value is None:
            assert values is not None, "Specify the `values` variables"
        else:
            # Set a default value if the column `values` is not specified
            assert values is None
            assert 'value1234' not in df.columns
            df['value1234'] = default_value
            values = 'value1234'
            
        if aggfunc is None:
            # Default is to sum up values
            # - NOTE**** THIS IS DIFFERENT FROM THE DEFAULT AGGFUNC 'mean' in pd.pivot_table()
            aggfunc = 'sum'

        return pd.pivot_table(df, index=index, columns=columns, values=values,
                              fill_value=fill_value, aggfunc=aggfunc)

    if index_use_index:
        index = pd.Series(df.index)
    else:
        index = df[index]

    if columns_use_index:
        columns = pd.Series(df.index)
    else:
        columns = df[columns]

    # Convert to categories
    try:
        index.cat
    except:
        index = index.astype('category')

    try:
        columns.cat
    except:
        columns = columns.astype('category')


    if remove_unused_categories:
        index = index.cat.remove_unused_categories()
        columns = columns.cat.remove_unused_categories()

    if square:
        if verbose: utils.tprint('Syncing index and column categorical coding')
        index = index.copy()
        columns = columns.copy()
        index, columns = sync_categories([index, columns])
        assert np.all(index.cat.categories == columns.cat.categories)
        if verbose: utils.tprint('Done syncing')

    if values is None:
        # Default value of 1
        values = np.ones(index.size, np.int64)
    else:
        values = df[values].values


    # if len(index.shape)==1:
    #     # A single variable index
    #     pass
    # else:
    #     # A MultiIndex
    #     index.
    
    shape = (index.cat.categories.size, columns.cat.categories.size)

    if fill_value != 0:
        # If you set fill_value to not 0, then maybe it's because your
        # dataframe contains 0's that are significant and should be
        # preserved in the output. In scipy.sparse, those 0's will be
        # removed. To counter this, a hack is to add an offset to all
        # values. For example, if all values are non-negative, then
        # simply adding an offset of 1 will do the trick. However, an
        # appropriate an offset can be tricky to set, e.g. values
        # could be -1 or 0 such that an offset of 1 would be
        # confusing. Thus, to avoid data-specific trickiness, here you
        # must manually specify the offset in the variable
        # `offset_hack`.

        assert rettype != 'spmatrix', 'Only a fill-value of 0 is supported if returning a scipy sparse matrix'

        assert offset_hack is not None
        sp = scipy.sparse.coo_matrix((values + offset_hack, (index.cat.codes, columns.cat.codes)), shape=shape)
    else:
        sp = scipy.sparse.coo_matrix((values, (index.cat.codes, columns.cat.codes)), shape=shape)

    # Need to eliminate zeros, as their presence will lead to a bug
    # when calling pandas' from_spmatrix() function
    # https://github.com/pandas-dev/pandas/issues/29814
    sp.eliminate_zeros()

    if binary:
        assert fill_value==0.

        # Threshold matrix if values are >0
        sp = (sp > 0).astype(int)
        
    if rettype in ['sparse', 'dense_df', 'spmatrix', 'ig']:
        # rownames = index.cat.categories[: index.cat.codes.max() + 1]
        # colnames = columns.cat.categories[: columns.cat.codes.max() + 1]

        rownames = index.cat.categories
        colnames = columns.cat.categories

        if rettype == 'sparse':
            df_sp = pd.DataFrame.sparse.from_spmatrix(sp, index=rownames, columns=colnames)

            # Change fill value
            dtype = sp.dtype.type
            fill_value_cast = dtype(fill_value)
            assert fill_value_cast == fill_value, \
                "Need to check that the fill value was preserved. E.g. This may have" \
                "happened if the underlying data was np.int64, and the" \
                "fill_value was 0.5"
            df_sp = df_sp.astype(pd.SparseDtype(dtype, fill_value_cast))

            if fill_value != 0:
                df_sp = df_sp - offset_hack

            return df_sp
        elif rettype == 'dense_df':
            return pd.DataFrame(sp.toarray(), index=rownames, columns=colnames)
        elif rettype == 'spmatrix':
            return sp, rownames, colnames
        elif rettype == 'ig':
            return utils.create_ig(sp, weighted=True, directed=directed,
                                   rownames=rownames, colnames=None if square else colnames, square=square)
    else:
        raise Exception('Unsupported return type')


def df_str_split(df, pat, column):
    """
    Splits up a string in a dataframe row into substrings across multiple rows.

    df : dataframe

    pat : pattern to split. Same as in pd.Series.str.split

    column: column to split on
    """

    if df.shape[0] > 0:
        orig_columns = df.columns
        other_cols = [x for x in df.columns if x != column]

        df = df.set_index(other_cols)[column].str.split(pat, expand=True)
        df = df.stack().reset_index(other_cols)
        df.reset_index(drop=True, inplace=True)
        df.rename(columns={0: column}, inplace=True)

        # Necessary because reset_index() may not preserve original order of columns
        df = df[orig_columns]
    return df


def fast_concat(df_list, fill_value=np.nan):
    """
    Faster pandas concatenation? (TODO: document when is it faster?)
    """

    all_columns = pd.Series(np.concatenate([df.columns.values for df in df_list])).unique()
    all_columns.sort()
    all_columns_series = pd.Series({b: a for a, b in enumerate(all_columns)})
    all_indices = np.concatenate([df.index.values for df in df_list])

    nrows = sum([df.shape[0] for df in df_list])

    # # Test if there are more than one data type in the dataframes
    # if (df_list[0].dtypes.unique().size > 1) or np.unique([y for x in df_list for y in x.dtypes]).size > 1:
    #     concat_arr = np.full((nrows, all_columns.size), fill_value, dtype=np.dtype('O'), order='C')

    # # Else, assume it's all float data types in the dataframe
    # else:
    #     concat_arr = np.full((nrows, all_columns.size), fill_value, dtype=np.float64, order='C')

    concat_arr = np.full((nrows, all_columns.size), fill_value, dtype=np.float64, order='C')

    curr_row = 0
    for df in df_list:
        concat_arr[curr_row: curr_row + df.shape[0], all_columns_series[df.columns].values] = df.values
        curr_row += df.shape[0]
    concat_df = pd.DataFrame(concat_arr, columns=all_columns, index=all_indices)

    return concat_df


def expand_df(df, newcols, fill_value=0, dtype=None, sparse=None, copy=False, sort=False):
    """Add new columns to a dataframe. Handles sparse dataframes.

    sort :

        If True, then rearrange order of all columns lexicographically. Default: False

    copy :

        Copies the dataframe before modifying it. Only applies when
        `sparse` is set to False or inferred to be False. If `sparse`
        is True, then the dataframe is always copied regardless of
        this parameter.

    """

    if sparse is None:
        try:
            # A column is sparse if and only if the column has a '.sparse' attribute
            df.iloc[:, 0].sparse
            sparse = True
        except:
            sparse = False

    if len(newcols) > 0:
        if sparse:
            if dtype is None:
                # Set the dtype to be the first column's dtype
                # (Assumes all columns have the same type)
                dtype = df.dtypes[0].type
            sp_dtype = pd.SparseDtype(dtype, fill_value=dtype(fill_value))

            sp = scipy.sparse.coo_matrix(([], ([], [])), dtype=dtype, shape=(df.shape[0], len(newcols)))
            df_sp = pd.DataFrame.sparse.from_spmatrix(
                sp,
                index=df.index,
                columns=newcols)
            df_sp = df_sp.astype(sp_dtype)
            df = pd.concat([df, df_sp], axis=1)
        else:
            if copy:
                df = df.copy()

            # Cast the fill value to the data type of the existing columns
            if dtype is None:
                dtype = df.dtypes[0].type
            fill_value = dtype(fill_value)

            for c in newcols:
                df[c] = fill_value

    return df

def unused_cat(df, inplace=False, columns=None):
    """Alias for remove_unused_categories()"""
    return remove_unused_categories(df, inplace=inplace, columns=columns)

def remove_unused_categories(df, inplace=False, columns=None):
    """Like pd.Series.cat.remove_unused_categories(), but does this for every categorical column in a DataFrame.

    From: https://stackoverflow.com/a/57827201/13485976
    """

    if not inplace:
        df = df.copy()

    if isinstance(df, pd.Series):
        if pd.api.types.is_categorical_dtype(df):
            df = df.cat.remove_unused_categories()

    else:
        if columns is None:
            columns = df.columns

        for c in columns:
            if pd.api.types.is_categorical_dtype(df[c]):
                df[c] = df[c].cat.remove_unused_categories()

    return df

def drop_dup(df, subset=None):
    """Alias for fast_drop_duplicates()"""
    return fast_drop_duplicates(df, subset=subset)

def fast_drop_duplicates(df, subset=None):
    is_duplicated = fast_duplicated(df, subset=subset)
    df = df[~ is_duplicated]
    return df

def fast_duplicated(df, subset=None):
    """Quickly computes which indices to keep in a dataframe.
    
    Much faster than DataFrame.drop_duplicates() when columns are categorical. The trick is to convert to numerical codes
    """
    
    if subset is None:
        subset = df.columns
#    df_codes = pd.DataFrame({c : df[c] if df[c].dtype.name!='category' else df[c].cat.codes for c in subset})
    df_codes = pd.DataFrame({c : df[c].cat.codes if pd.api.types.is_categorical_dtype(df[c]) else df[c].values for c in subset})

    is_duplicated = df_codes.duplicated().values
    return is_duplicated


def fast_loc(series, query_index, rettype=None, unused=None):
    """
    Fast version of series.loc[query_index] where series is a pd.Series.
    
    If query_index has a lot of redundant values, then it's faster to just call series.loc[] on the unique values and then repeat those results
    """
    
    if isinstance(query_index, pd.Series):
        query_index = query_index.values
    if not isinstance(query_index, pd.Categorical):
        query_index = pd.Categorical(query_index)
    
    if (unused is None) or unused:
        # Remove unused categories, so you don't waste time accessing their values
        # -- This might be extra overhead to identify unused categories if there aren't many of them
        query_index = query_index.remove_unused_categories()

    loc_results = series.loc[np.array(query_index.categories)].values[query_index.codes] #query_index.rename_categories(np.array())

    if rettype=='series':
        loc_results = pd.Series(loc_results, index=query_index)

    return loc_results


def update_series(series_A, series_B):
    """
    series_A.update(series_B) will only update keys that are already in series_A.

    This function will also update keys that are NOT in series_A but are in series_B.

    Returns 
    -------

    An updated copy of series_A (NOT INPLACE)
    """

    series_A = series_A.copy()

    series_A.update(series_B)

    return pd.concat([series_A, series_B.loc[series_B.index.difference(series_A.index)]])

def subset(df, keys=None, agg='and', unused=True, copy=True, invert=False, columns=None, verbose=False, return_mask=None, **kwargs):
    """
    Convenience function for doing subsetting of dataframes, e.g.

    df[(df[c1]==v1) & utils.catin(df[c2], v2)]

    Any extra keyword arguments `kwargs` are assumed to be keys.
    """

    if len(df)==0:
        # If the input dataframe is empty, then the mask will be empty obviously. However, the returned dataframe will lose all columns, so shape is (0,0).
        # - To prevent this, just return the dataframe
        return df

    if keys is None:
        keys = {}
    keys = {**keys, **kwargs}

    mask = []
    for k, v in keys.items():
        if verbose:
            utils.tprint('Calculating mask from column %s' % k)
#        print( isinstance(df, pd.Series) )

        # if isinstance(k, str) and any(k.startswith(s) for s in ['<', '>', '!']):
        #     if k.startswith('>='):
        #         k = k[2:]
        #         op = operator.ge
        #     elif k.startswith('<='):
        #         k = k[2:]
        #         op = operator.le
        #     elif k.startswith('>'):
        #         k = k[1:]
        #         op = operator.gt
        #     elif k.startswith('<'):
        #         k = k[1:]
        #         op = operator.lt
        #     elif k.startswith('!'):
        #         k = k[1:]
        #         op = operator.ne
        #     series = df if isinstance(df, pd.Series) else df[k]
        #     mask.append(op(series,v))

        if isinstance(k, str) and any(k.startswith(s) for s in ['<', '>', '!']):
            if k.startswith('>='):
                k = k[2:]
                op = operator.ge
            elif k.startswith('<='):
                k = k[2:]
                op = operator.le
            elif k.startswith('>'):
                k = k[1:]
                op = operator.gt
            elif k.startswith('<'):
                k = k[1:]
                op = operator.lt
            elif k.startswith('!'):
                k = k[1:]
                op = 'invert' # Indicates a logical not of the mask 
#                op = operator.ne
        else:
            op = None

        series = df if isinstance(df, pd.Series) else df[k]            
        if callable(v):
            m = series.apply(v)
        elif hasattr(v, '__iter__') and not isinstance(v, str):
            if series.dtype.name=='category':
                m = catin(series, v)
            else:
                m = series.isin(v)
        elif (op is not None) and (op != 'invert'):
            m = op(series, v)
        else:
            m = series==v

        if op=='invert':
            m = ~m
        
        mask.append(m)

            # if callable(v):
            #     mask.append(series.apply(v))
            # elif hasattr(v, '__iter__') and not isinstance(v, str):
            #     if series.dtype.name=='category':
            #         mask.append(catin(series, v))
            #     else:
            #         mask.append(series.isin(v))        
            # else:
            #     mask.append(series==v)

    if verbose:
        utils.tprint('Aggregating masks')
    if agg=='and':
        mask = functools.reduce(lambda x,y: x & y, mask)
    elif agg=='or':
        mask = functools.reduce(lambda x,y: x | y, mask)
    else:
        raise Exception('Invalid agg: %s' % agg)

    if invert:
        mask = ~ mask

    if return_mask:
        return mask

    if verbose:
        utils.tprint('Applying mask')
    if columns is None:
        df = df[mask]
    else:
        df = df.loc[mask, columns]

    if unused:
        if verbose:
            utils.tprint('Removing unused categories')

        if unused is True:
            df = unused_cat(df)
        else:
            assert hasattr(unused, '__iter__')
            df = unused_cat(df, columns=unused)
    else:
        if copy:
            if verbose:
                utils.tprint('Copying dataframe')

            df = df.copy()

    if verbose:
        utils.tprint('Done')
    return df

def sp_to_df(sp,
             rownames=None,
             colnames=None,
             row_header=None,
             col_header=None,
             value_header=None):
    """Convert a scipy sparse matrix into a long-format DataFrame
    """

    sp = sp.tocoo()

    df = pd.DataFrame()

    if row_header is None:
        row_header = 'row'
    if rownames is None:
        rows = sp.row
    else:
        rows = pd.Categorical.from_codes(sp.row, rownames)
    df[row_header] = rows

    if col_header is None:
        col_header = 'col'
    if colnames is None:
        cols = sp.col
    else:
        cols = pd.Categorical.from_codes(sp.col, colnames)
    df[col_header] = cols

    if value_header is None:
        value_header = 'value'
    df[value_header] = sp.data

    return df

def read_table(A, read_table_kws=None, cols=None, verbose=None, post_apply=None, post_apply_kws=None):
    """Reads a table, by auto-inferring data type.

    Possible inputs are:

    (1) a Dataframe (which is just returned)
    (2) a pickled file containing a Dataframe
    (3) a text table
    (4) a list or tuple of any of the three above"""

    if isinstance(A, (list, tuple)):
        C = pd.concat([read_table(aa, read_table_kws=read_table_kws, cols=cols, post_apply=post_apply, post_apply_kws=post_apply_kws) for aa in A])

    elif isinstance(A, pd.DataFrame):
        C = A

    elif os.path.exists(A):
        if verbose:
            utils.tprint('Reading table from {}'.format(A))
        try:
            C = utils.unpickle(A)
        except:
            if read_table_kws is None:
                read_table_kws = {}
            if 'header' not in read_table_kws:
                read_table_kws['header'] = 0
            C = pd.read_table(A, **read_table_kws)
    elif isinstance(A, (str, Path)):
        raise Exception('File {} does not exist'.format(A))
    else:
        raise Exception('Unsupported file type')

    # print('Before apply:')
    # from IPython.display import display
    # display(C)

    if post_apply is not None:
        # Apply a function to the table
        if post_apply_kws is None:
            post_apply_kws = {}
        C = post_apply(C, **post_apply_kws)

        # print('After apply:')
        # display(C)

    if cols is not None:
        C = C[cols]

    return C

def write_table(A, path, pkl=None, txt=None, txt_kws=None, overwrite=None, overwrite_err_msg=None):
    """Writes a pandas dataframe, either by pickling+compressing or to text file.

    Auto-ignores if `path` is None

    overwrite : If False, then check if file already exists, and raise Exception if so. Default: True.
    """

    if path is None:
        # If path is None, then assume that we didn't want to write anything
        return

    if pkl is None and txt is None:
        pkl = True
    assert bool(pkl) != bool(txt)

    utils.check_overwrite(path, overwrite, overwrite_err_msg)
                
    if pkl:
        utils.pickle(A, path)
    else:
        if txt_kws is None:
            txt_kws = {}
        if 'header' not in txt_kws:
            txt_kws['header'] = True
        if 'index' not in txt_kws:
            txt_kws['index'] = False
        if 'sep' not in txt_kws:
            txt_kws['sep'] = '\t'

        A.to_csv(path, **txt_kws)
