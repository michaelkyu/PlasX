import functools
import shutil
import time
import glob
import gzip
import os
import subprocess
import gc
import tempfile
import io
from pathlib import Path
import multiprocessing
from concurrent.futures import ThreadPoolExecutor

import scipy, scipy.sparse
import pandas as pd
import numpy as np
import numba
from numba import jit, prange

from plasx import utils, nb_utils, constants
from plasx.nb_utils import extend_1d_arr, extend_2d_arr, remove_duplicates, remove_duplicates_2d, lexsort_nb, lexsort_nb2, get_boundaries, get_boundaries2

@jit(nopython=True, nogil=True)
def unique_roll_sorted(x, y, n):
    # Assume x and y are sorted
    
    # Current x and y
    xc = x[0]
    yc = y[0]
    
    g = 0
    groups = np.zeros(n, dtype=np.int64)
    counts = np.zeros(n, dtype=np.int64)
    cc = 1
    
    for i in range(x.size):
        xx, yy = x[i], y[i]
        if xx != xc:
            counts[g] = cc
            groups[g] = xc
            xc = xx
            yc = yy
            g += 1
            cc = 1
        elif yy != yc:
            yc = yy
            cc += 1
    groups[g] = xc
    counts[g] = cc
    return groups, counts

@jit(nopython=True, nogil=True)
def unique_size(x):
    size = x.size

    if size > 25:
        return np.unique(x).size
    elif size > 2:
        return len(set(x))
    elif size == 2:
        return 1 + (x[1] != x[0])
    else:
        return size

@jit(nopython=True, nogil=True)
def slice_bool(y, use_y, y_slice):
    j = 0
    for i in range(y.size):
        y_slice[j] = y[i]
        j += use_y[i]
    return y_slice[:j]
        
#@jit(nopython=True, parallel=True)
@jit(nopython=True, parallel=True, nogil=True)
def unique_roll(x, y):
    # Assumes that the same x are grouped consecutively (but not
    # necessarily sorted).  Doesn't assume that y is sorted.

    groups, boundaries = get_boundaries(x)
    counts = np.empty(groups.size, dtype=np.int32)
    
    for bi in prange(boundaries.size - 1):
        counts[bi] = unique_size(y[boundaries[bi] : boundaries[bi+1]])
    
    return groups, counts

@jit(nopython=True, nogil=True)
def unique_roll_bool(x, y, use_y, buff_size=-1):
    # Assumes that the same x are grouped consecutively (but not
    # necessarily sorted).  Doesn't assume that y is sorted.

    groups, boundaries = get_boundaries(x)
    counts = np.empty(groups.size, dtype=np.int32)
    
    if buff_size == -1:
        buff_size = y.size
    y_slice_buff = np.empty(buff_size, y.dtype)
    
    for bi in range(boundaries.size - 1):
        counts[bi] = unique_size(slice_bool(y[boundaries[bi] : boundaries[bi+1]], use_y[boundaries[bi] : boundaries[bi+1]], y_slice_buff))
#         counts[bi] = unique_size(y[boundaries[bi] : boundaries[bi+1]][use_y[boundaries[bi] : boundaries[bi+1]]])
    
    return groups, counts

def unique_roll_mt(x, y, use_y=None, buff_size=-1, threads=None):
    """
    Multi-threaded version of `unique_roll_bool`
    """

    if threads is None:
        threads = multiprocessing.cpu_count()

    with ThreadPoolExecutor(max_workers=threads) as executor:
        splits = utils.split_groups(x, threads)
        if use_y is None:
            arg_list = zip(*[(x[i:j], y[i:j]) for i, j in splits])
            results = executor.map(unique_roll, *arg_list)
        else:
            arg_list = zip(*[(x[i:j], y[i:j], use_y[i:j], buff_size) for i, j in splits])
            results = executor.map(unique_roll_bool, *arg_list)

    groups, counts = [np.concatenate(a) for a in zip(*results)]
    return groups, counts
        
def count_unique_keys(df, group, key, sort=True, method='numba'):
    df_orig = df
    df = df[[group, key]].copy()
    
    for x in [group, key]:
        if df[x].dtypes.name=='category':
            df[x] = df[x].cat.codes.values
                
    if method=='numba':        
        # NOTE, dropna????????????????????????? with y????

        if sort:
            x, y = df.values[np.lexsort(df.values.T[::-1,:]), :].T
        else:
            x, y = df.values.T
        n = np.unique(x).size
        groups, counts = unique_roll(x, y, n)
        df_counts = pd.DataFrame({key : counts}, index=groups)
    elif method=='groupby':
        unique_size = lambda x: np.unique(x.values).size
        df_counts = df.groupby(group, sort=sort, observed=True).aggregate({key : unique_size})
        
    if df_orig[group].dtypes.name=='category':
        df_counts.index = df_orig[group].cat.categories[df_counts.index]
            
    return df_counts

def make_clu_sizes(clu, 
                   headers,
                   contigs,
                   unique_table=None,
                   entities=None,
                   plasmid=True,
                   bacteria=True,
                   method='numba'):

    if unique_table is not None:
        clu = duplicate_clu(clu, unique_table)

    if entities is None:
        entities = ['genes', 'contigs', 'species', 'family', 'order']
    taxa_ranks = [x for x in entities if x in utils.ordered_ranks or (x == 'taxid')]

    # TODO: I need to actually reps be a 2-tuple of (rep, identity)
    # In the meantime, I assume that identical reps (but across different identities) are not consecutive
    reps, members = clu[['representative', 'member']].values.T

    is_plasmid = utils.categorical_isin(headers['contig'], contigs.index[contigs['is_plasmid']])[members]

    if method=='numba':
        if 'identity' in clu:
            clu_sizes = clu[['representative', 'identity']].drop_duplicates().set_index(['representative', 'identity'])
        else:
            clu_sizes = clu[['representative']].drop_duplicates().set_index('representative')
            #clu_sizes = pd.DataFrame(index=remove_duplicates(reps))

        if len(taxa_ranks) > 0:
            #tmp = contigs[taxa_ranks].loc[contigs.index.isin(headers['contig'].cat.categories)].copy()
            tmp = contigs.loc[contigs.index.isin(headers['contig'].cat.categories), taxa_ranks].copy()
            tmp.index = pd.Categorical(tmp.index.values, categories=headers['contig'].cat.categories).codes

        for ent in entities:
            # `members_ids` : Arrays that indicate the contig, species, etc. of the corresponding sequence in `headers`

            if ent=='contig':
                members_ids = headers['contig'].cat.codes.values[members]
                use_members = None
            elif ent=='genes':
                members_ids = members
                use_members = None
            elif ent in taxa_ranks or (ent == 'taxid'):
                members_ids = tmp.loc[headers['contig'].cat.codes.values[members], ent].values        
                use_members = ~ np.isnan(members_ids)
                if ent=='species':  assert use_members.all()
                members_ids = members_ids.astype(np.int32)
            elif ent in ['COG_FUNCTION', 'Pfam']:
                members_ids = headers[ent].cat.codes.values[members]
                use_members = members_ids != -1  # Value of -1 represents a NaN, i.e. no COG annotation            
            else:
                raise Exception('Unsupported entity: %s' % ent)

            if use_members is None:
                if plasmid:  clu_sizes['plasmid_%s' % ent] = unique_roll_mt(reps, members_ids, is_plasmid)[1]
                if bacteria: clu_sizes['bacteria_%s' % ent] = unique_roll_mt(reps, members_ids, ~ is_plasmid)[1]
            else:
                if plasmid: clu_sizes['plasmid_%s' % ent] = unique_roll_mt(reps, members_ids, use_members & is_plasmid)[1]
                if plasmid: clu_sizes['bacteria_%s' % ent] = unique_roll_mt(reps, members_ids, use_members & (~ is_plasmid))[1]

            if plasmid and bacteria:
                if ent in ['genes', 'contig']:
                    clu_sizes[ent] = clu_sizes['plasmid_%s' % ent] + clu_sizes['bacteria_%s' % ent]
                else:
                    clu_sizes[ent] = unique_roll_mt(reps, members_ids, use_members)[1]

    else:
        ##############################
        # Older/slower implementation
        
        clu = clu.copy()

        clu['plsdb'] = in_plsdb
        clu['contig'] = headers.loc[clu['member'].values, 'contig'].values
        clu.loc[clu['plsdb'], 'species'] = plsdb.loc[headers.loc[clu.loc[clu['plsdb'], 'member'].values, 'contig'], 'species'].values
        clu.loc[~ clu['plsdb'], 'species'] = bacteria_dict.loc[headers.loc[clu.loc[~ clu['plsdb'], 'member'].values, 'contig'], 'species'].values

        clu_sizes = pd.DataFrame(index=clu['representative'].unique()).sort_index()

        clu_sizes['plsdb_genes'] = clu.loc[clu['plsdb'], 'representative'].value_counts().sort_index()
        clu_sizes['plsdb_genes'].fillna(0, inplace=True)
        clu_sizes['refseq_genes'] = clu.loc[~ clu['plsdb'], 'representative'].value_counts().sort_index()
        clu_sizes['refseq_genes'].fillna(0, inplace=True)
        clu_sizes['genes'] = clu_sizes['plsdb_genes'] + clu_sizes['refseq_genes']

        clu_sizes['plsdb_contigs'] = count_unique_keys(clu[clu['plsdb']], 'representative', 'contig', method='groupby')
        clu_sizes['plsdb_contigs'].fillna(0, inplace=True)
        clu_sizes['refseq_contigs'] = count_unique_keys(clu[~ clu['plsdb']], 'representative', 'contig', method='groupby')
        clu_sizes['refseq_contigs'].fillna(0, inplace=True)
        clu_sizes['contigs'] = clu_sizes['plsdb_contigs'] + clu_sizes['refseq_contigs']

        clu_sizes['plsdb_species'] = count_unique_keys(clu[clu['plsdb']], 'representative', 'species', method='groupby')
        clu_sizes['plsdb_species'].fillna(0, inplace=True)
        clu_sizes['refseq_species'] = count_unique_keys(clu[~ clu['plsdb']], 'representative', 'species', method='groupby')
        clu_sizes['refseq_species'].fillna(0, inplace=True)
        clu_sizes['species'] = clu_sizes['plsdb_species'] + clu_sizes['refseq_species']

    return clu_sizes

def parse_mmseqs2_cluster(mmseqs_fasta_path, subset=None):
    """
    Returns:
    
    cluster_dict : dict

        fasta_header_of_representative_sequences --> list of Bio.SeqIO fasta records in the cluster

    """
    
    from Bio import SeqIO

    cluster_dict = {}
    curr_cluster = None
    
    # Determine whether to open by gzip or not
    if mmseqs_fasta_path[-3:]=='.gz':
        fopen = gzip.open
    else:
        fopen = open
        
    with fopen(mmseqs_fasta_path, 'rt') as f:
        for record in SeqIO.parse(f, "fasta"):
            if len(record.seq)==0:
                # Start of a new cluster
                curr_cluster = record.id
                if subset is None or (curr_cluster in subset):
                    cluster_dict[curr_cluster] = []
            else:
                if subset is None or (curr_cluster in subset):
                    cluster_dict[curr_cluster].append(record)
    return cluster_dict

def createsubdb(subset,
                db_prefix,
                output_prefix,
                mmseqs_cmd=None,
                subset_is_headers=True,
                convert2fasta=False):
    """Writes all the sequences into a fasta file and creates a mmseqs db

    db_prefix : Path of mmseqs database, from which to retrieve sequences. e.g. mmseqs_dir / 'plsdb_refseq'

    output_prefix: Path to write sequences into a fasta file and create a mmseqs db
    """

    if mmseqs_cmd is None:
        mmseqs_cmd = 'mmseqs'

    db_prefix = str(db_prefix)

    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)

    # `subset` is assumed to be a subset of the fasta headers in
    # `db_prefix`_h. If you just pass in `subset` directly into
    # `mmseqs createsubdb`, it will interpret the values as the
    # indices in `db_prefix`.indices (i.e. the first column) rather
    # than as fasta headers. This is very weird behavior. To fix this,
    # I will convert values in `subset` to indices in
    # `db_prefix`.index
    if subset_is_headers:
        seq_indices = read_seq_indices(db_prefix, usecols=['index'])
        subset = seq_indices.loc[seq_indices.index.isin(set(subset)), 'index']

    try:
        delete = False
        try:
            assert os.path.exists(subset)
            subset_file = subset
            print('WARNING: make sure that `subset` refers to indices in {0}.index'
                  'instead of a subset of fasta headers in {0}_h. Yes, this is'
                  'unintuitive. If you are unsure, look into the code for this function'.format(db_prefix))
        except:
            _, subset_file = tempfile.mkstemp()
#            delete = True
            with open(subset_file, 'wt') as f:
                f.write('\n'.join(map(str, subset)) + '\n')
        
        db_prefix = str(db_prefix)
        output = str(output_prefix)
        output_h = str(output_prefix) + '_h'
        output_fa = str(output_prefix) + '_fa'

        cmd = ' '.join([mmseqs_cmd, 'createsubdb', subset_file, db_prefix, output])
        print(cmd)
        utils.run_cmd(cmd, shell=True, tee=True)
#        subprocess.run(cmd, shell=True, check=True)


        cmd = ' '.join([mmseqs_cmd, 'createsubdb', subset_file, db_prefix + '_h', output_h])
        print(cmd)
        utils.run_cmd(cmd, shell=True, tee=True)
#        subprocess.run(cmd, shell=True, check=True)

        if convert2fasta:
            cmd = ' '.join([mmseqs_cmd, 'convert2fasta', output, output_fa])
            print(cmd)
            utils.run_cmd(cmd, shell=True, tee=True)
#            subprocess.run(cmd, shell=True, check=True)

    finally:
        if delete:
            os.remove(subset_file)

def write_cluster(cluster_names,
                  clu,
                  cluster_dict,
                  output_prefix):
    """
    
    cluster_names :

        The name of each cluster. In mmseqs2, this is the name of the 'representative' sequence of the cluster.

    clu :
    
        Clustering, one column is the 'representative' sequence, and another column is the 'member' sequence

    cluster_dict : dict

        Created by parse_mmseqs2_cluster.

        fasta_header_of_representative_sequences --> list of Bio.SeqIO fasta records in the cluster

    """

    from Bio import SeqIO

    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)

    # Write sequences in all clusters to a fasta file
    fasta_path = os.path.join(output_prefix + '.fa')
    with open(fasta_path, 'w') as f:
        for rep in cluster_names:
            SeqIO.write(cluster_dict[rep], f, "fasta")
            f.write('\n')

    # Create indices to represent the clusters, and write cluster memberships to table
    tmp = clu.loc[clu['representative'].isin(cluster_names), ['representative', 'member', 'contig']]
    cluster_indices = pd.Series({a : 'mmseqs2_%s' % i for i, a in enumerate(cluster_names)})
    tmp['gene_cluster'] = cluster_indices.loc[tmp['representative']].values
    tmp[['member', 'gene_cluster']].to_csv(output_prefix + '.clustering', sep="\t", header=True, index=False)

    # Pivot 2-column table of (contig, cluster) to a cluster-by-contig table. If multiple genes in a contig are in a cluster, then report the number of genes.
    tmp['value'] = 1.
    tmp = tmp[['gene_cluster', 'contig', 'value']].pivot_table(columns=['gene_cluster'], index=['contig'], values='value', aggfunc=np.sum, fill_value=0.)
    utils.pickle(tmp, output_prefix + ".txt.gz")

    ## Create mmseqs database of clusters
    cmd = utils.make_cmd("mmseqs createdb {0}.fa {0}.mmseqsDB 2>&1 > {0}.log".format(output_prefix))
    print(cmd)
    subprocess.call(cmd, shell=True)

def read_alignm8(path, low_memory=False, names=None, **kwargs):
    """
    Read the output of a `mmseqs convertalis` output table
    """

    if names is None:
        names = ['qId', 'tId',
                 'seqIdentity', 'alnLen', 'mismatchCnt', 'gapOpenCnt',
                 'qStart', 'qEnd', 'tStart', 'tEnd', 'eVal', 'bitScore']
    
    dtype = {'qId': np.dtype('int32'),
             'tId': np.dtype('int32'),
             'seqIdentity': np.dtype('float64'),
             'alnLen': np.dtype('int32'),
             'mismatchCnt': np.dtype('int32'),
             'gapOpenCnt': np.dtype('int32'),
             'qStart': np.dtype('int32'),
             'qEnd': np.dtype('int32'),
             'qlen': np.dtype('int32'),
             'tStart': np.dtype('int32'),
             'tEnd': np.dtype('int32'),
             'tlen': np.dtype('int32'),
             'eVal': np.dtype('float64'),
             'bitScore': np.dtype('int32')}

    if low_memory:
        for c in ['alnLen', 'mismatchCnt', 'gapOpenCnt',
                  'qStart', 'qEnd', 'tStart', 'tEnd', 'bitScore']:
            dtype[c] = np.dtype('uint16')
        for c in ['seqIdentity', 'eVal']:
            dtype[c] = np.dtype('float32')
    
    df = pd.read_csv(path,
                     sep='\t',
                     names=names,
                     dtype=dtype,
                     **kwargs)

    return df

def pickle_alignm8(path, output=None, **kwargs):
    df = read_alignm8(path, **kwargs)    
    if output is None:
        output = str(path) + '.pkl.blp'
    utils.pickle(df, output)
    return df

def pickle_clu(clu, member_category=False, dtype=None):
    if dtype is None:
        dtype = {'representative': np.dtype(np.int32), 'member' : np.dtype(np.int32)}

    df = pd.read_csv(clu, sep='\t', header=None, names=['representative', 'member'], dtype=dtype)
    if member_category:
        df['member'] = df['member'].astype('category')
    utils.pickle(df, str(clu) + '.pkl.blp')
    del df

    gc.collect()

def prep_mmseqs_fasta(input_fa, output_prefix, header_filter=None):
    """Takes a set of fasta files from anvi-get-sequences-for-gene-calls
    and preps files for mmseqs clustering. Outputs:
       
       (1) Concatenates them into a fasta file.

       (2) This fasta file has renamed headers for memory
       efficiency. File describing the renaming is created.

       (3) The fasta headers are parsed into gene_callers_id, contig,
       start, stop, etc. as expected of the header format from
       anvi-get-sequences-for-gene-calls. Pickled table of this
       parsing is saved.

    """
    output_prefix = str(output_prefix)

    if input_fa is not None:        
        df = utils.index_fasta_headers(input_fa,
                                       output_prefix + '.fa',
                                       header_filter=header_filter)
    print('sequences:', df.shape[0])
    utils.pickle(df, output_prefix + '.headers.pkl.blp')

def read_h(path, dtype=None):
    """Read a mmseqs header file (i.e., filename with suffix '_h')

    E.g. plsdb_refseq_h, or clu.rep_h
    """

    if dtype is None:
        dtype = np.int32

    with open(path, 'rb') as f:

        # Replace the null bytes '\x00' in the file with a newline char '\n'
        with io.BytesIO(bytes(f.read()).replace((bytes.fromhex('00')), b'\n')) as g:

            index = pd.read_csv(g, header=None, dtype=dtype)[0].values

    return index

def read_db_fasta(path, rettype=None):
    """Creates a fasta dictionary from a mmseqs db. Reads the
    nucleotide/amino acid sequences and matches it with the sequence names
    in the mmseqs '_h' header file"""

    headers = read_h(str(path)+'_h')
    seqs = read_h(str(path), dtype=np.str)
    
    fasta_dict = dict(zip(headers, seqs))
    
    if rettype=='series':
        return pd.Series(fasta_dict)
        
    return fasta_dict

def read_clu(prefix):
    filelist, _ = get_mmseqs_data_files(prefix)
    
    clu_list = []
    for path in filelist:
        with open(path, 'rb') as f:
            r = f.read()
            clu = [np.array(x.splitlines(), np.int32) for x in r.split(b'\x00') if len(x)>0]
            clu_list.extend(clu)
            #reps = np.array([x[0] for x in members], dtype=np.int64)
    
    return clu_list

def get_clu_prefix(identity, mmseqs_dir):
    return str(Path(mmseqs_dir) / ('clu%s' % identity) / 'clu')

def unpickle_clu(identity, mmseqs_dir):
    return utils.unpickle(get_clu_prefix(identity, mmseqs_dir) + '.tsv.pkl.blp')
    
def read_indices(path, usecols=None):

    if usecols is None:
        usecols = ['index', 'offset', 'size']

    return pd.read_csv(path,
                       sep='\t',
                       names=['index', 'offset', 'size'],
                       usecols=usecols,
                       dtype={'index' : np.int64, 'offset' : np.int64, 'size' : np.int32},
                       header=None)

def read_clu_indices(clu_prefix):
    clu_index = read_indices(str(clu_prefix) + '.index')
    clu_index.index = read_h(str(clu_prefix) + '.rep_h')
    clu_index.index.name = 'representative'

    return clu_index

def read_clu_msa_indices(clu_prefix):
    clu_msa_index = read_indices(str(clu_prefix) + '.msa.index')
    clu_msa_index.index = read_h(str(clu_prefix) + '.rep_h')
    clu_msa_index.index.name = 'representative'

    return clu_msa_index

def read_seq_indices(seq_prefix, usecols=None, try_pickle=True):
    """Read sequence database, created using `mmseqs createdb`

    """

    if try_pickle and os.path.exists(str(seq_prefix) + '.index.pkl.blp'):
        seq_indices = utils.unpickle(str(seq_prefix) + '.index.pkl.blp')
    else:
        seq_indices = read_indices(str(seq_prefix) + '.index')

        # Read list of header names (i.e. the names of each gene sequence (or the numeric ID that I created))
        h_list = read_h(str(seq_prefix) + '_h')

        # Read the index table for this header file (I know, it's confusing)
        h_indices = read_indices(str(seq_prefix) + '_h.index')

        # Make sure that the ordering of IDs in the index file for sequence data (i.e. '.index' file) is in the same order as IDs in the index file for the headers (i.e. the '_h.index' file)
        assert (seq_indices['index']==h_indices['index']).all()

        # If the above assertion is true, then we can just set the
        # index of this dataframe to be the list of header names
        seq_indices.index = h_list
        seq_indices.index.name = 'sequence'

    return seq_indices

def get_mmseqs_data_files(prefix):

    if os.path.exists(prefix):
        # The database is in one file
        file_list = [prefix]
    else:
        # The database has been split into multiple files, with suffixes <prefix>.1, <prefix>.2, etc.
        file_list = []
        for x in glob.glob(str(prefix) + '*'):
            try:
                int(x.split(os.path.basename(prefix) + '.')[1])
                file_list.append(x)
            except:
                pass
        file_list.sort(key=lambda x : int(x.split('.')[-1]))

        # Check that there is a consecutive set of file indices
        #print(file_list)
        for i in range(len(file_list)):
            assert file_list[i] == (str(prefix) + ('.%s' % i))

    filesizes = np.cumsum([os.path.getsize(x) for x in file_list])
    return file_list, filesizes

def query_mmseqs_db(seq_list,
                    db_prefix,
                    index_table=None,
                    filesizes=None,
                    remove_gaps=False,
                    rettype='list',
                    method='seek',
                    verbose=False):
    """
    Read mmseqs db
    """

    from Bio import SeqIO

    if isinstance(seq_list, str) or np.issubdtype(type(seq_list), np.integer):
        seq_list = [seq_list]

    if filesizes is None:
        utils.tprint('Reading filesizes', verbose=verbose)
        file_list, filesizes = get_mmseqs_data_files(db_prefix)

    if index_table is None:
        utils.tprint('Reading index table', verbose=verbose)
        index_table = read_seq_indices(db_prefix)
        
    #display(index_table.loc[seq_list, ['offset', 'size']].head(20))

    utils.tprint('Fetching data from mmseqs db', verbose=verbose)
    fa_list = []
    if method=='seek':
        # tmp = index_table.loc[seq_list, ['offset', 'size']].sort_values('offset')

        # # First file_i
        # file_i = np.searchsorted(filesizes, tmp['offset'][0])
        # f = open(file_list[file_i], 'rb')

        # for offset, size in zip(tmp['offset'], tmp['size']):

        #     if offset >= filesizes[file_i]:
        #         f.close()
        #         file_i = np.searchsorted(filesizes, tmp['offset'][0])
        #         f = open(file_list[file_i], 'rb')

        for seq in seq_list:
            offset, size = index_table.loc[seq, ['offset', 'size']]

            file_i = np.searchsorted(filesizes, offset)
            if filesizes[file_i] == offset:
                offset = 0
            elif file_i > 0:
                offset -= filesizes[file_i - 1]

            with open(file_list[file_i], 'rb') as f:
                f.seek(offset)
                x = f.read(size)
                x = x.replace((bytes.fromhex('00')), b'')

            x = x.decode()
            if remove_gaps:
                x = x.replace('-', '')

            x = x.strip()
            fa_list.append(x)

    elif method=='stream':
        # Read through the whole file, and output lines that match seq_list
        pass

        tmp = index_table.loc[seq_list, ['offset', 'size']].sort_values('offset')

        offset = tmp['offset'][0]
        file_i = np.searchsorted(filesizes, offset)
        

        f_tell = 0
        offset_iter = zip(tmp['offset'], tmp['size'])

        while True:

            # try:
            #     offset, size = next(offset_iter)
            # except StopIteration:
            #     break

            file_i = np.searchsorted(filesizes, offset)
            with open(file_list[file_i], 'rb') as f:
                for x in f:
                    if f_tell == offset:
                        assert len(x) == size
                        x_str = x.replace(b'\x00', b'').decode().strip()
                        fa_list.append(x_str)
                        try:
                            offset, size = next(offset_iter)
                        except StopIteration:
                            0 / asdf
                    elif f_tell > offset:
                        break

                    f_tell += len(x)
    elif method=='mmseqs':
        try:
            folder = tempfile.mkdtemp()
            subset_prefix = os.path.join(folder, 'subset')
            createsubdb(seq_list, str(db_prefix), subset_prefix, convert2fasta=True)
            fa_list = [str(seq.seq) for seq in SeqIO.parse(subset_prefix + '.fa', 'fasta')]
        finally:
            shutil.rmtree(folder)
    else:
        raise Exception('Unsupported method')

    if rettype=='list':
        return fa_list

    elif rettype=='alv':
        #assert len(fa_list)>1, 'You only have 1 sequence. You need more >=2 sequences to visualize an alignment'

        fa = '\n'.join(fa_list)
        return utils.run_alv(fa)


def muscle_align():
    """
    TODO
    """

    from Bio import SeqIO

    from Bio.Align.Applications import MuscleCommandline
    from io import StringIO

    muscle_cline = MuscleCommandline("/scratch/miniconda/envs/anvio-master-zelda/bin/muscle", clwstrict=True)

    handle = StringIO()
    SeqIO.write(cluster_dict[rep], handle, "fasta")
    data = handle.getvalue()

    stdout, stderr = muscle_cline(stdin=data)
    from Bio import AlignIO
    align = AlignIO.read(StringIO(stdout), "clustal")
    # print(align)

    import tempfile
    with tempfile.NamedTemporaryFile('w', delete=False) as f:
        f.write(align.format('fasta'))
        print(f.name)


def get_taxonomy_in_clu(rep, clu, contigs, headers):
    ranks = contigs.loc[headers.loc[clu.loc[clu['representative']==rep, 'member'], 'contig'], utils.ordered_ranks]

    import ete3
    ncbi = ete3.NCBITaxa()
    
    return {c : ncbi.translate_to_names(ranks[c].dropna().unique()) for c in ranks.columns}

def duplicate_clu(clu, unique_table, sort=True):
    """Reintroduces duplicate genes into a clustering
    
    sort:
    
        If True (default), then sort by 'representative' (they might
        be unsorted after the dask merging). Sorting is assumed for
        downstream tasks like unique_roll_mt (which is used by
        make_clu_sizes)

    """

    import dask.dataframe as dd

    if sort:
        clu = unique_table.merge(dd.from_pandas(clu, npartitions=16).set_index('member', compute=True),
                                 left_index=True, right_index=True).set_index('representative').reset_index().compute()
    else:
        clu = unique_table.merge(dd.from_pandas(clu, npartitions=16).set_index('member', compute=True),
                                 left_index=True, right_index=True).compute().reset_index(drop=True)

    clu = clu[['representative', 'member']]

    return clu

def clu_to_sp(clu):
    """TODO : remove this and replace with cluster.Cluster.from_mmseqs()"""

    clu_reps = clu['representative'].astype('category')

    clu_sp = scipy.sparse.coo_matrix((np.ones(clu.shape[0], np.bool),
                                      (clu_reps.cat.codes, clu['member'].values))).tocsr()

    return clu_sp, clu_reps.cat.categories

def merge_clu(mmseqs_dir,
              ident_list,
              headers,
              contigs,
              method='nest',
              unique_table=None,
              members_to_use=None,
              plasmid=False,
              verbose=False):
    """Merges clusters from different identities.

    members_to_use : 

        Because I am filtering members with
        `members_in_unique_contigs`, there will be strange cases where
        the representative gene will not be among its member genes.
        This is because the representative is not in
        `members_in_unique_contigs`, but it is nonetheless retained in
        order to define the cluster

    """

    if method in ['nest', 'equal']:

        # If method=='nest', then concatenate clusters but remove redundancies. If cluster X
        # contains cluster Y, and X occurs at a lower identity
        # (e.g. 50%) than Y does (e.g. 90%), then keep X and discard Y
        #
        # If method=='equal', then revemo clusters that are equal

        clu_reps_list = []
        clu_sp_list = []

        for ident in ident_list:
            if verbose: utils.time_print('Identity: %s' % ident)

            clu = unpickle_clu(ident, mmseqs_dir)

            if verbose: print('Clusters:', clu['representative'].unique().size)

            if members_to_use is not None:
                clu = clu[clu['member'].isin(members_to_use)]

            if plasmid:
                # Only keep clusters that contain only plasmid genes
                clu_sizes = make_clu_sizes(clu, headers, contigs,
                                           unique_table=unique_table,
                                           entities=['genes'],
                                           plasmid=False, bacteria=True)
                clu = clu.loc[clu['representative'].isin(clu_sizes.index[clu_sizes['bacteria_genes']==0]), :]
                if verbose: print('Plasmid-only clusters:', clu['representative'].unique().size)

            clu_sp, clu_reps = clu_to_sp(clu)
            clu_reps_list.append(clu_reps)
            clu_sp_list.append(clu_sp)

        Ak_list, Z = cluster.nest_reduce(clu_sp_list, method=method)
        Z_reps = np.concatenate([clu_reps[Ak].astype(np.int32) for Ak, clu_reps in zip(Ak_list, clu_reps_list)])
        if verbose: print('Total clusters after merging:', Z_reps.size)
        if verbose: print('Unique set of representatives (might be reused across clusters):', np.unique(Z_reps).size)

        info = pd.concat([pd.Series({'Identity' : ident,
                                     'All clusters' : '%s' % A.shape[0],
                                     'Non-redundant singletons' : (A[Ak, :].sum(1) == 1).sum(),
                                     'Non-redundant total' : Ak.size,
                                     '% non-redundant' : '%0.2f' % (Ak.size/A.shape[0])}) \
                          for ident, A, Ak in zip(ident_list, clu_sp_list, Ak_list)], axis=1).T

        ident_reps = np.repeat(ident_list, [x.size for x in Ak_list])
        i, j = Z.nonzero()
        clu_merge = pd.DataFrame({'representative' : Z_reps[i],
                                  'identity' : ident_reps[i],
                                  'member' : j}).astype(
                                      {'representative' : np.int32,
                                       'identity' : np.int32,
                                       'member' : np.int32})

        return clu_merge, info

    elif method=='concat':
        # Simply concatenate the clusters across identities

        assert (members_to_use is None) \
            and (plasmid is False) \
            and (unique_table is None), 'Not supported'

        clu_list = []
        for ident in ident_list:
            if verbose: utils.time_print('Identity: %s' % ident)

            clu_prefix = mmseqs_dir / ('clu%s' % ident) / 'clu'
            clu = utils.unpickle(str(clu_prefix) + '.tsv.pkl.blp')
            clu['identity'] = ident
            clu = clu[['representative', 'identity', 'member']]
            clu_list.append(clu)

        clu_merge = pd.concat(clu_list, ignore_index=True)
        return clu_merge

def process_clu(ident, mmseqs_dir):
    """
    Reads the clustering for a given identity
    
    Also calculate clu_sizes
    """
    
    clu_prefix = mmseqs_dir / ('clu%s' % ident) / 'clu'
    clu = utils.unpickle(str(clu_prefix) + '.tsv.pkl.blp')

    unique_contigs = set(plsdb.index) | set(bacteria_dict.index[~ bacteria_dict['in_PLSDB']])
    clu_sizes = make_clu_sizes(clu, headers, contigs, entities=['genes'])
    
    return clu, clu_sizes

def createdb(fa, output):
    """
    Run `mmseqs createdb`
    """

    utils.run_cmd('mmseqs createdb {} {}'.format(fa, output))
    # cmd = [mmseqs_cmd, 'createdb', fa, output]
    # print(' '.join(map(str, cmd)))
    # subprocess.run(cmd, check=True)

def get_unique_sequences(db,
                         mmseqs_cmd=None,
                         table_output=None,
                         createsubdb_output=None,
                         threads=None):

    if mmseqs_cmd is None:
        mmseqs_cmd = 'mmseqs'

    if threads is None:
        threads = utils.get_max_threads()

    with tempfile.TemporaryDirectory() as tempdir:
        clusthash = os.path.join(tempdir, 'clusthash')
        clu = os.path.join(tempdir, 'clu')
        if table_output is None:
            table_output = os.path.join(tempdir, 'clu.tsv')

        cmd = [mmseqs_cmd, 'clusthash', db, clusthash, '--min-seq-id', '1.0', '--threads', threads]
        utils.run_cmd(' '.join(map(str, cmd)), verbose=True)
        # print(' '.join(map(str, cmd)))
        # subprocess.run(cmd, check=True) 

        cmd = [mmseqs_cmd, 'clust', db, clusthash, clu, '--threads', threads]
        utils.run_cmd(' '.join(map(str, cmd)), verbose=True)
        # print(' '.join(map(str, cmd)))
        # subprocess.run(cmd, check=True)

        # Two column-tabler of cluster memberships
        cmd = [mmseqs_cmd, 'createtsv', db, db, clu, table_output, '--threads', threads]
        utils.run_cmd(' '.join(map(str, cmd)), verbose=True)
        # print(' '.join(map(str, cmd)))
        # subprocess.run(cmd, check=True)

        pickle_clu(table_output)
        unique_ids = utils.unpickle('{0}.pkl.blp'.format(table_output))['representative'].unique()
        #unique_ids = pd.read_csv(table_output, header=None, sep='\t')[0].unique()
        
        if createsubdb_output is not None:            
            createsubdb(unique_ids,
                        db,
                        createsubdb_output)
        
    return unique_ids

@jit(nopython=True, parallel=True)
def overlap_nb(s1, e1, s2, e2):
    
    assert (s1 < e1).all()
    assert (s2 < e2).all()
        
    intersection = np.empty(s1.size, s1.dtype)
    union = np.empty(s1.size, s1.dtype)
    length1 = np.empty(s1.size, s1.dtype)
    length2 = np.empty(s1.size, s1.dtype)
    
    for i in prange(s1.size):
    
        # First sort to see which interval has the earliest starting point, and then label that interval A
        if s1[i] < s2[i]:            
            sA, eA = s1[i], e1[i]
            sB, eB = s2[i], e2[i]
        else:
            sA, eA = s2[i], e2[i]
            sB, eB = s1[i], e1[i]
        
#         print(sA, eA, sB, eB)
        
        if eA < sB:
            intersection[i] = 0
        else:
            intersection[i] = min(eA, eB) - sB
            
        length1[i] = e1[i] - s1[i]
        length2[i] = e2[i] - s2[i]
            
    return intersection, length1, length2

def overlap(df, s1, e1, s2, e2, method='jaccard'):
    """
    df : pd.DataFrame
    s1, e1 : columns in `df` that represents the start and end of interval 1
    s2, e2 : start / end of interval 2
    """
    
    return overlap_nb(df[s1].values, df[e1].values, df[s2].values, df[e2].values)

def merge_hits_align(hits, align, threshold=None):
    hits_overlap = pd.merge(align[['qId', 'tId', 'tStart', 'tEnd', 'seqIdentity']].astype({'tStart' : np.uint16, 'tEnd' : np.uint16}),
                            hits[['qId', 'tId', 'tStart', 'tEnd', 'seqIdentity']], on='tId', how='inner')

    intersection, length1, length2 = overlap(hits_overlap, 'tStart_x', 'tEnd_x', 'tStart_y', 'tEnd_y')
    
    if threshold is not None:
        hits_overlap['jaccard'] = intersection / (length1 + length2 - intersection)
        hits_overlap = hits_overlap.loc[hits_overlap['jaccard'] >= threshold, ['qId_y', 'tId', 'seqIdentity_x', 'seqIdentity_y']].rename(columns={'qId_y':'qId'}).drop_duplicates()

    return hits_overlap, intersection, length1, length2


def add_identity_relations(clu_merge):
    """Because some plasmid contigs were represented in both plsdb and
    ncbi-refseq-bacteria, I removed genes from `clu_merge_plasmid`
    that came from duplicate contigs. In the process, there are
    cluster 'representative' Ids (which is the set used by hits['tId']
    that do not show up as cluster 'member' Ids. This is a hack to put
    back those identity relationships

    """

    tmp = clu_merge[['representative', 'identity']].drop_duplicates()
    tmp['member'] = tmp['representative']
    tmp = pd.concat([clu_merge, tmp], ignore_index=True).drop_duplicates()
    tmp = tmp.sort_values(['representative', 'member'], kind='mergesort').sort_values(['identity'], kind='mergesort', ascending=False)
    
    return tmp

def augment_hits(hits,                 
                 q_headers,
                 t_headers,
                 clu_merge_plasmid,
                 clu_merge_plasmid_sizes):

    # Get coverage and metagenome sample of query genes
    hits = hits.merge(q_headers[['length', 'contig']].rename(columns={'length':'qLen', 'contig':'qContig'}),
                      left_on='qId', right_index=True, how='left')
    hits['qCov'] = (hits['qEnd'] - hits['qStart']) / (hits['qLen'] / 3)
    hits['qSample'] = hits['qContig'].apply(lambda x: '_'.join(x.split('_')[:2])).astype('category')
    
    # Get target gene info: alignment coverage, function annotation, and partial gene
    hits = hits.merge(t_headers[['length', 'COG_FUNCTION', 'Pfam', 'partial']].rename(columns={'length' : 'tLen'}),
                      left_on='tId', right_index=True, how='left')
    hits['tCov'] = (hits['tEnd'] - hits['tStart']) / (hits['tLen'] / 3)

    # For each tId, calculate the number of query metagenomes samples it has a hit against
    tId_2_qSamples = hits[['qSample', 'tId']].sort_values(['qSample', 'tId']).drop_duplicates()['tId'].value_counts().to_frame('tSamples').rename_axis('tId')
    hits = hits.merge(tId_2_qSamples, on='tId')

    hits['has_func'] = ~ hits[['COG_FUNCTION', 'Pfam']].isna().all(1)

    ## Add info about mmseqs clusters
    ## -- member_2_cluster has index which are the cluster members, and columns are taxonomy of its containing cluster

    ## -- Because some plasmid contigs were represented in both plsdb
    ## -- and ncbi-refseq-bacteria, I removed genes from
    ## -- `clu_merge_plasmid` that came from duplicate contigs. In the
    ## -- process, there are cluster 'representative' Ids (which is
    ## -- the set used by hits['tId'] that do not show up as cluster
    ## -- 'member' Ids. This is a hack to put back those identity
    ## -- relationships
    clu_merge_plasmid = add_identity_relations(clu_merge_plasmid)
    # member_2_cluster = clu_merge_plasmid[clu_merge_plasmid['member'].isin(hits['tId'].unique()) | clu_merge_plasmid['representative'].isin(hits['tId'].unique())].set_index('member')
    member_2_cluster = clu_merge_plasmid[clu_merge_plasmid['member'].isin(hits['tId'].unique())].set_index('member')
    member_2_cluster = member_2_cluster.merge(clu_merge_plasmid_sizes[['genes', 'species', 'genus', 'family', 'phylum']],
                                              left_on=['representative', 'identity'],
                                              right_index=True)
    # There are duplicates members (drop them) associated with multiple identities
    member_2_cluster = member_2_cluster.loc[~ member_2_cluster.index.duplicated(), :]

    hits = hits.merge(member_2_cluster, left_on='tId', right_index=True)

    return hits

def print_hits_reduction(hits, hits_reduced, q_headers):
    print('Reduced %s to %s pairs\n'
          '        %s to %s query genes\n'
          '        %s to %s plasmid genes\n'
          '        %s to %s query contigs' % \
          (hits.shape[0], hits_reduced.shape[0],
           hits['qId'].unique().size, hits_reduced['qId'].unique().size,
           hits['tId'].unique().size, hits_reduced['tId'].unique().size,
           q_headers['contig'].unique().size, q_headers.loc[hits_reduced['qId'].unique(), 'contig'].unique().size))

def summarize_align_qId(align):
    # For each cluster, get the minimum sequence identity from a member of the cluster and min bitScore to its representative
    rep_radius = align.groupby('qId').agg(minIdentity=pd.NamedAgg(column='seqIdentity', aggfunc=np.min),
                                          maxIdentity=pd.NamedAgg(column='seqIdentity', aggfunc=np.max),
                                          minBitScore=pd.NamedAgg(column='bitScore', aggfunc=np.min),
                                          maxBitScore=pd.NamedAgg(column='bitScore', aggfunc=np.max))
    rep_radius = rep_radius.reset_index()

    ## For each cluster, get the first and last position in the representative that is aligned to ANY member
    # Non-singletons, take the min qStart and the max qEnd
    tmp = align[align['qId'] != align['tId']]
    rep_span = pd.merge(tmp[['qStart', 'qId']].sort_values(['qStart', 'qId']).drop_duplicates('qId'),
                        tmp[['qEnd', 'qId']].sort_values(['qEnd', 'qId'], ascending=False).drop_duplicates('qId'),
                        on='qId')
    # For singletons, just take the entire length
    tmp2 = align.loc[~ align['qId'].isin(rep_span['qId']), ['qId', 'qStart', 'qEnd']]
    tmp2 = tmp2.drop_duplicates('qId') # There may be duplicates across multiple clustering identities
    rep_span = pd.concat([rep_span, tmp2],
                         ignore_index=True, sort=True)

    rep_df = pd.merge(rep_radius, rep_span, on='qId')
    rep_df = rep_df.rename(columns={'qId' : 'rep', 'qStart' : 'start', 'qEnd' : 'end'})

    return rep_df

def hits_to_anvio_functions_table(hits, q_headers, t_headers, contigs_info, source):
    tmp = hits.copy()
    tmp['function'] = t_headers.loc[tmp['tId']].apply(lambda x: 'contig={ncbi_id};start={start};stop={stop};direction={direction};partial={partial}'.format(ncbi_id=contigs_info.loc[x['contig'], 'NCBI_ID'], **x),
                                                         axis=1).values
    tmp['function'] = tmp.apply(lambda x: 'bitScore={bitScore};seqIdentity={seqIdentity:0.3f},qStart={qStart};qEnd={qEnd};qCov={qCov:0.3f},tStart={tStart};tEnd={tEnd};tCov={tCov:0.3f}    {function}'.format(**x), axis=1)
    tmp = tmp[['qId', 'tId', 'function', 'bitScore']].rename(columns={'bitScore':'e_value'})    
    tmp['source'] = source
    tmp = tmp.merge(q_headers[['gene_callers_id', 'contig']], left_on='qId', right_index=True)
    tmp['accession'] = tmp['tId'].apply(lambda x: ('%s:%s' % (source, x)))
    tmp = tmp.drop(columns=['qId', 'tId'])

    return tmp


def augment_headers_with_functions(headers, out_path):
    plsdb_genes = utils.unpickle('/home-nfs/mikeyu/mbm-data/plsdb/plsdb-gene-calls-blastp.pkl.blp')
    refseq_genes = utils.unpickle('/share/data/mbm-data/ncbi-refseq-bacteria/refseq_bacteria-gene-calls.pkl.blp')
    gene_calls = utils.better_pd_concat([plsdb_genes[['gene_callers_id','contig','partial']],
                                         refseq_genes[['gene_callers_id','contig','partial']]], ignore_index=True)
    headers = pd.merge(headers, gene_calls, on=['gene_callers_id', 'contig'], how='left')

    del gene_calls, plsdb_genes, refseq_genes
    import gc ; gc.collect()

    for function_name in ['COG_FUNCTION', 'Pfam']:
        utils.time_print(function_name)

        if function_name == 'COG_FUNCTION':
            plsdb_functions = utils.unpickle('/home-nfs/mikeyu/mbm-data/plsdb/plsdb-functions-blastp_%s.pkl.blp' % function_name)
            refseq_functions = utils.unpickle('/share/data/mbm-data/ncbi-refseq-bacteria/refseq_bacteria-functions-diamond-sensitive_%s.pkl.blp' % function_name)

        elif function_name == 'Pfam':
            plsdb_functions = utils.unpickle('/home-nfs/mikeyu/mbm-data/plsdb/plsdb-functions_%s.pkl.blp' % function_name)
            refseq_functions = utils.unpickle('/share/data/mbm-data/ncbi-refseq-bacteria/refseq_bacteria-functions_%s.pkl.blp' % function_name)

        functions = utils.better_pd_concat([plsdb_functions[['gene_callers_id','accession','contig']],
                                            refseq_functions[['gene_callers_id','accession','contig']]], ignore_index=True)

        # TODO: I'm dropping COGs and Pfams, but I don't want to do this!
        functions.drop_duplicates(subset=['gene_callers_id', 'contig'], inplace=True)
        functions.rename(columns={'accession' : function_name}, inplace=True)

        headers = pd.merge(headers, functions[['gene_callers_id', 'contig', function_name]], on=['gene_callers_id', 'contig'], how='left')

    # Due to a bug in pandas, categoricals after merging may not remain categoricals (known in a github issue)
    headers['contig'] = headers['contig'].astype('category')

    # Convert to more efficient dtypes
    headers = headers.astype({c : np.int32 for c in ['gene_callers_id', 'start', 'stop', 'length', 'partial']})
    headers = headers.astype({'partial' : bool})

    # Only need to keep headers
    del functions, plsdb_functions, refseq_functions
    import gc ; gc.collect()

    utils.pickle(headers, out_path)

    del headers
    gc.collect()

def create_contig_2_clusters(headers, clu, unique_table=None, method='multiply'):
    """
    Combines three tables to create a table of contigs-by-clusters (long or wide format)

    headers :
    
        contig-by-genes long format

    clu :
    
        gene-cluster-by-unique-genes long format

    unique_table :

        genes-by-unique-genes long format
    """

    if method=='pd_merge':
        "Merge long-format dataframes by pd.merge. Advantage: preserve information like gene coordinates"
        
        raise Exception()

    if method=='multiply':
        "Convert by matrix multiplication"

        # contigs-by-genes
        headers_sp = Cluster.from_longdf(headers, cluster='contig', member_index=True,
                                        catnames=True, catmembers=False)
        print('contig-by-genes:', headers_sp.S.shape)

        # gene-cluster-by-unique-genes
        clu_sp = Cluster.from_mmseqs(clu, is_merge=True)
        print('gene-cluster-by-unique-genes:', clu_sp.S.shape)

        # Convert to integer, so that the matrix multiplication product
        # will be integer (it might be incorrectly boolean, if everything were left as boolean)
        assert np.issubdtype(clu_sp.S.dtype, np.bool) or np.issubdtype(clu_sp.S.dtype, np.integer)
        clu_sp.S = clu_sp.S.astype(np.int32)

        # Check dimensions
        assert headers_sp.S.shape[1] == clu_sp.S.shape[1]

        if unique_table is None:
            unique_sp = scipy.sparse.identity(headers_sp.S.shape[1], dtype=np.bool)
        else:
            # If a dask dataframe, then compute into a pd.DataFrame
            if not isinstance(unique_table, pd.DataFrame):
                import dask
                assert isinstance(unique_table, dask.dataframe.core.DataFrame)
                unique_table = unique_table.compute()

            # genes-by-unique-genes
            unique_sp = Cluster.from_longdf(unique_table, cluster='member', member_index=True,
                                            catnames=False, catmembers=False)
            print('genes-by-unique-genes:', unique_sp.S.shape)

        prod = mkl_spgemm.dot(mkl_spgemm.dot(headers_sp.S, unique_sp.S), clu_sp.S.T)
        contig_2_clusters = Cluster(S=prod,
                                    names=headers_sp.names,
                                    members=clu_sp.names)

        return contig_2_clusters

def number_clu(clu, prefix=None, use_identity=None):
    """Assigns a unique ID to each 'representative' (if clusters are at a
    single identity), or to each ('representative', 'identity') pair
    in a merging of mmseqs clusters.

    """
    
    if prefix is None:
        prefix = 'mmseqs_'

    if use_identity is None:
        use_identity = True

#    columns = ['identity', 'representative']
    columns = (['identity'] if use_identity else []) + ['representative']

    accession_df = clu.drop(columns=['member']).sort_values(columns).drop_duplicates(columns)

    accession_df['accession'] = pd.Categorical.from_codes(
        np.arange(accession_df.shape[0]),
        categories=prefix + ((accession_df['identity'].astype(str) + '_') if use_identity else '') + accession_df['representative'].astype(str))

        # categories=functools.reduce(lambda x,y: x+y,
        #                             [prefix] + \
        #                             [accession_df['identity'].map(str), '_'] + \
        #                             [accession_df['representative'].map(str)]))

    # Merge these accessions to the original table
    member_2_cluID = clu[columns + ['member']].merge(accession_df[columns + ['accession']], on=columns)

    member_2_cluID = member_2_cluID.drop(columns=columns).set_index('member')

    # Change 'acccession' from a categorical back to a string object
    accession_df['accession'] = accession_df['accession'].astype('object')
    accession_df = accession_df.set_index('accession')

    return member_2_cluID, accession_df


def mmseqs_to_anvio_functions_table(clu, source, headers, unique_table=None, dask=True, prefix=None, use_identity=None):
    """Merges the mmseqs cluster information (containing at least the
    columns ['representative', 'identity']) with sequence header info
    (e.g. ['contig', 'start', 'stop', 'etc']) to create a table that
    looks similar to the output of `anvio-export-functions`.

    source : name of gene function source, e.g. 'mmseqs_90'

    unique_table : a table mapping redundant genes
    """

    if dask:
        merge_func = utils.dask_merge
    else:
        merge_func = pd.merge

    if prefix is None:
        prefix = source + '_'

    member_2_cluID, accession_df = number_clu(clu, prefix=prefix, use_identity=use_identity)
    if unique_table is not None:
        # Make sure that the 'representative' column was set to the index
        assert unique_table.index.name == 'representative' 

        # Faster method than doing merge below: implemented 2-2-7-21
        tmp = unique_table.loc[utils.isin_int(unique_table.index, member_2_cluID.index)].copy()
        tmp['accession'] = member_2_cluID.loc[tmp.index, 'accession'].values
        tmp.set_index('member', inplace=True)
        member_2_cluID = tmp

        # member_2_cluID = merge_func(unique_table, member_2_cluID,
        #                                   left_index=True, right_index=True).set_index('member')

    # return member_2_cluID
    # display(member_2_cluID)
    member_2_cluID = merge_func(member_2_cluID, headers, left_index=True, right_index=True, how='left')
    member_2_cluID['e_value'] = 0.0
    member_2_cluID['source'] = source
    member_2_cluID['source'] = member_2_cluID['source'].astype('category')

    return member_2_cluID, accession_df

def concat_mmseqs_functions(mmseqs_df, known_df, known_descriptions, child_source=None, parent_source=None):
    """Concatenates a table of gene annotations to mmseqs clusters with a
    table of gene annotations to a known table, e.g. COG_FUNCTION.

    Removes redundant gene families/clusters.

    Example arguments for child/parent source:
    
        child_source=['mmseqs_merge'], parent_source=['COG_FUNCTION', 'Pfam']
    """
    
    assert (child_source is not None) and (parent_source is not None)

    utils.tprint('Start')

    columns = ['accession', 'contig', 'gene_callers_id', 'start', 'stop', 'direction', 'partial', 'e_value', 'source']
    func = utils.better_pd_concat([mmseqs_df[columns], known_df[columns]], ignore_index=True)
    func['accession'].cat.remove_unused_categories(inplace=True)

    utils.tprint('mmseqs/known categories:',
                 func.groupby(['source']).apply(lambda x : x['accession'].cat.codes.unique().size).to_dict())
    utils.tprint('mmseqs/known instances:',
                 func['source'].value_counts().to_dict())

    # Add dummy descriptions of mmseqs clusters and any other gene accessions with no explicit description
    descriptions = known_descriptions.reindex(func['accession'].cat.categories)
    descriptions[descriptions.isna()] = descriptions.index[descriptions.isna()].values

    # Map from accession to source, e.g. 'COG1216'-->'COG_FUNCTION' or 'mmseqs_50_123'-->'mmseqs_merge'
    accession_2_source = utils.df_2cols(func, 'accession', 'source')

    utils.tprint('Calculating overlap between {} and {}'.format(child_source, parent_source))

    # Filter out set of mmseqs that are ALWAYS occurring exactly as a COG or Pfam
    # -- same[i,j] = Number of times that function i occurs exactly where function j is
    # -- nest[i,j] = 1 if function i ALWAYS occurs exactly where function j is
    same, same_dic = enrich.overlap_accessions(func, single_counts=True)
    jacc = enrich.jaccard(same, same_dic['instances'].values)

    utils.tprint('Calculating redundant functions from {}'.format(child_source))

    is_parent = accession_2_source[same_dic.index].isin(parent_source).values
    is_child = accession_2_source[same_dic.index].isin(child_source).values
    redundant = same_dic.index[is_child][utils.as_flat(jacc.tocsr()[is_child, :][:, is_parent].sum(1) > 0)]
    #redundant = same_dic.index[is_child][utils.as_flat((jacc.tocsr()[is_child, :][:, is_parent] == 1).any(1))]

    utils.tprint('Removing {:n} redundant functions from {}'.format(redundant.size, child_source))

    # Remove redundant mmseqs functions, and then drop unused accession categories
    func = func[~ func['accession'].isin(redundant)]
    func['accession'].cat.remove_unused_categories(inplace=True)
    
    utils.tprint('Sorting functions table')
    func.sort_values(['contig', 'start', 'stop'], inplace=True)

    utils.tprint('After removing redundancies - mmseqs/known categories:',
                 func.groupby(['source']).apply(lambda x : x['accession'].cat.codes.unique().size).to_dict())
    utils.tprint('After removing redundancies - mmseqs/known instances:',
                 func['source'].value_counts().to_dict())

    func['source'] = func['source'].astype('category')

    return func, descriptions, same, same_dic, jacc

def examine_member(rep, identity, mmseqs_dir, msa_char=100000, seq=True, msa=True, db_name=None):
    """Get the sequences AND/OR the MSA for a representative in a mmseqs cluster.

    E.g. 

    seq_list = mmseqs.examine_member(42932356, 5, mmseqs_dir, seq=True, msa=False)


    """

    clu_prefix = get_clu_prefix(identity, mmseqs_dir)

    if hasattr(rep, '__iter__'):
        rep_list = rep
    else:
        assert isinstance(rep, (int, np.integer)), "Input needs to be integer(s)"
        rep_list = [rep]

    if seq:
        if db_name is None:
            db_name = 'plsdb_refseq.all'
        mmseqs_dir = Path(mmseqs_dir)

        #seq_prefix = mmseqs_dir / 'plsdb_refseq' # This is wrong, becuase it's just be the set of unique sequences
        seq_prefix = mmseqs_dir / db_name
        seq_index = read_seq_indices(seq_prefix)

    if msa:
        clu_msa_index = read_clu_msa_indices(clu_prefix)
            

    seq_list = []

    for rep in rep_list:    
        if seq:
            # Print out the sequence
            out = query_mmseqs_db(rep, seq_prefix, seq_index, rettype='list')
            seq_list.append(out[0])
            print('>%s\n%s' % (rep, out[0]))
            print()

        if msa:
            # Print out the msa
            msa_out = query_mmseqs_db(rep, clu_prefix+'.msa', clu_msa_index, rettype='list')            
            msa_alv = utils.run_alv('\n'.join(msa_out))
            print(msa_alv[:msa_char])
        else:
            msa_out = None
    
    return seq_list, msa_out


def extract_clu_merge(clu_merge, DB, subset_dir, ident_list=None, get_clu_dir=None):
    """
    Gathers subsets of different mmseqs cluster runs, according to a master table clu_merge of clusters across those runs.
    
    These runs are assumed to be over different identities, e.g. 90%, 80%, etc.
    
    Params
    -------
    
    clu_merge :
    
        pd.DataFrame with three columns : representative, identity, member
        
    DB :
    
        The original sequence database from which the mmseqs clusters were generated
        
    subset_dir :
    
        Base directory to create subset databases
        
    ident_list :

        List of cluster identities to use in `clu_merge` (Default: use all identities)
        
    get_clu_dir : 

        A function that given a sequence identity, will return the directory containing the clustering at that identity

    Output
    ------

    <subset_dir>/clu<identity>.subset : a list of cluster representatives that defines the subset
    <subset_dir>/clu<identity> : the subset of the cluster database
    <subset_dir>/clu<identity>.profile : a profile database of the subset of clusters
    <subset_dir>/clu<identity>.rep : the subset of the original sequence database consisting of the cluster representatives
    <subset_dir>/clu<identity>.members : the subset of the original sequence database consisting of the cluster members (so not just the representatives)

    """

    if ident_list is None:
        ident_list = clu_merge['identity'].unique()

    if get_clu_dir is None:
        get_clu_dir = lambda ident: plasx.constants.mmseqs_dir / ('clu%s' % ident) / 'clu'

    os.makedirs(subset_dir, exist_ok=True)
    
    for ident in ident_list:
        utils.time_print('######################### Identity: %s ##########################' % ident)

        clu = get_clu_dir(ident)

        # Read cluster index table
        clu_indices = read_clu_indices(clu)
  
        DB_indices = read_seq_indices(DB)
   
        output_prefix = subset_dir / ('clu%s' % ident)

        # print(clu)

        # Get and write the list of representatives
        # -- `reps` is a list of sequence names (or numeric IDs I created, and now I need to lookup their mmseqs IDs
        reps_file = str(output_prefix) + '.subset_reps'
        reps = clu_merge.loc[clu_merge['identity']==ident, 'representative'].drop_duplicates().values

        if len(reps)==0:
            ## This sequence identity has not representatives/members in the table `clu_merge`
            utils.time_print('######################### SKIPPING Identity %s, which has not representatives/members #########################' % ident)

            continue

        # Lookup the mmseqs IDs, and write to file
        clu_indices.loc[reps, 'index'].sort_values().to_csv(reps_file, header=False, index=False)

        # Get and write the list of members
        members_file = str(output_prefix) + '.subset_members'
        members = clu_merge.loc[clu_merge['identity']==ident, 'member'].drop_duplicates().values
        #clu_indices.loc[members, 'index'].sort_values().to_csv(members_file, header=False, index=False)
        DB_indices.loc[members, 'index'].sort_values().to_csv(members_file, header=False, index=False)

        cmd_kws = dict(mmseqs_cmd='mmseqs',
                       reps_file=reps_file,
                       members_file=members_file,
                       clu=clu,
                       output_prefix=output_prefix,
                       DB=DB)

        # EXPLANATION OF CREATESUBDB
        # The input of createsubdb commands: `mmseqs createsubdb <input_queries> <source_db> <output_file>
        # where <input_queries> is a set of queries into the source database <source_db>
        # The subset of <source_db> is then written to <output_file>

        # Takes the subset of the cluster files
        utils.run_cmd('{mmseqs_cmd} createsubdb {reps_file} {clu} {output_prefix}'.format(**cmd_kws))

        # Takes the subset of reps from the original sequence db's sequence data
        utils.run_cmd('{mmseqs_cmd} createsubdb {reps_file} {DB} {output_prefix}.rep'.format(**cmd_kws))
        # Takes the subset of reps from the original sequence db's headers
        utils.run_cmd('{mmseqs_cmd} createsubdb {reps_file}  {DB}_h {output_prefix}.rep_h'.format(**cmd_kws))

        # Turn this subset of clusters into a profile
        utils.run_cmd('{mmseqs_cmd} result2profile {output_prefix}.rep {DB} {output_prefix} {output_prefix}.profile'.format(**cmd_kws))

        # Takes the subset of members from the original sequence db's sequence data
        utils.run_cmd('{mmseqs_cmd} createsubdb {members_file} {DB} {output_prefix}.members'.format(**cmd_kws))
        # Takes the subset of members from the original sequence db's headers
        utils.run_cmd('{mmseqs_cmd} createsubdb {members_file} {DB}_h {output_prefix}.members_h'.format(**cmd_kws))


def prepare_assemblies_for_plasmid_search(samples, table_summary, table_dir, mmseqs_prefix,
                                          anvio_export=False,
                                          source_list=None, verbose=False):
    """
    Prepares a set of assemblies (either contigs.db or a pair of fasta + annotations)
    
    (1) Requires gene function annotations. Thus, must either give contigs.db with the annotations, or a functions table
    (2) Exports the gene sequences and searches them against an mmseqs database
    
    ## Export gene calls and COG function annotations, and then compile into tables
    
    Params
    -------
    table_summary : File prefix for CONCATENATED gene functions and calls
    table_dir : Directory to put gene functions, calls, and sequences of individual contigs.db
    mmseqs_prefix : File prefix to create mmseqs db and related files
    
    Output
    -------
    
    <mmseqs_prefix>.all : file prefix for concatenated fasta file of gene sequences and mmseqs db
    <mmseqs_prefix> : file prefix for mmseqs db of UNIQUE gene sequence
    """

    os.makedirs(table_dir, exist_ok=True)

    if anvio_export:
        # Export and concatenate gene calls and functions
        utils.anvi_export_and_concat(samples, table_summary, table_dir, source_list=source_list, verbose=verbose)
    
    sample_prefixes = [os.path.basename(x).split('-contigs.db')[0] for x in samples]
    fasta_paths = [table_dir / ('%s-gene-sequences.fa.blp' % x) for x in sample_prefixes]
        
    # Export gene sequences 
    gene_calls = utils.unpickle(str(table_summary) + '-gene-calls.pkl.blp')

    assert 'aa_sequence' in gene_calls.columns, 'Column aa_sequence is needed to extract gene sequences'

    for i, (sample, sample_prefix, fasta_path) in enumerate(zip(samples, sample_prefixes, fasta_paths)):
        utils.time_print(i, end=', ')
        
        with utils.silence_output():
            _ = utils.get_gene_sequences(db_path=sample,
                                         output_path=fasta_path,
                                         gene_calls=gene_calls[gene_calls['db']==sample],
                                         gene_source='prodigal',
                                         subset='all',
                                         compress='blp',
                                         verbose=True,
                                         method='from_gene_calls')    

    os.makedirs(os.path.dirname(mmseqs_prefix), exist_ok=True)
            
    ## Concatenate gene sequences and prepare mmseqs database    
    create_mmseqs_db(fasta_paths, mmseqs_prefix)


def create_mmseqs_db(fasta, output_prefix, threads=None):
    """
    Concatenate multiple fasta files together and create an mmseqs db
    -- Fasta headers are automatically renamed to integers, for more efficient memory use
    -- A smaller database of only unique sequences is created    
    
    <output_prefix>.all : the file prefix for the concatenated fasta file and mmseqs db
    <output_prefix> : the file prefix for the UNIQUE set of sequences
    
    
    """
    output_prefix = str(output_prefix)
    
    # Concatenate fasta and write to file (also renamed headers to integers)
    prep_mmseqs_fasta(fasta, output_prefix + '.all')

    # Create mmseqs db from the above fasta file
    createdb(output_prefix + '.all.fa', output_prefix + '.all')

    # Pickle the sequence mmseqs index, for faster lookup later
    seq_index = read_seq_indices(output_prefix + '.all')
    utils.pickle(seq_index, output_prefix + '.all.index.pkl.blp')

    # Create mmseqs database of only unique gene sequences
    unique_ids = get_unique_sequences(output_prefix + '.all',
                                      table_output = output_prefix + '.unique.tsv',
                                      createsubdb_output = output_prefix,
                                      threads=threads)

    # NOTE: commenting this out, because I think this pickling already happens in the call to get_unique_sequences() above
#    pickle_clu(str(output_prefix) + '.unique.tsv')

    print('Unique gene sequences:', len(unique_ids))


def mmseqs_search(source_db, target_db, output, tmp_dir=None, threads=None, splits=None, sleep_seconds=300):
    """
    Do sequence-to-sequence search with mmseqs
    """
    
    if threads is None:
        threads = utils.get_max_threads()

    if splits is not None and splits != 0:
        assert isinstance(splits, int)
        splits = f"--split {splits} --split-mode 0"
    
    # Update 12/13/21: Added qlen and tlen to the output, so that you don't need to read these from the header files
    cmd_list = [f"mmseqs search {source_db} {target_db} {output}.search {output}.search.tmp -a {splits} --threads {threads} 2>&1 | tee {output}.search.log 2>&1",
                f"mmseqs convertalis {source_db} {target_db} {output}.search {output}.m8 --format-output query,target,pident,alnlen,mismatch,gapopen,qstart,qend,qlen,tstart,tend,tlen,evalue,bits"]
    for cmd in cmd_list:
        utils.run_cmd(cmd, verbose=True)    
        utils.tprint(f'Sleeping for {sleep_seconds} seconds to let the file system flush')
        time.sleep(sleep_seconds) # Sleep, to let files from command freshen up

    if not os.path.exists(f"{output}.m8"):
        raise FileNotFoundError(f"The file {output}.m8 was supposed to be created, but it doesn't exist. This might be because the search using mmseqs2 ran out of system RAM. Consider setting the -S flag to reduce the maximum RAM usage. E.g., if you only have ~8Gb RAM, we recommend setting -S to 32 or higher.")

    names = ['qId', 'tId',
             'seqIdentity', 'alnLen', 'mismatchCnt', 'gapOpenCnt',
             'qStart', 'qEnd', 'qlen', 'tStart', 'tEnd', 'tlen', 'eVal', 'bitScore']
    _ = pickle_alignm8(output + '.m8', low_memory=True, names=names)

def mmseqs_merge_search(source_db, target_db_dir, output, ident_list, threads=None, splits=None):
    """Runs mmseqs_search on a set of nested-merged clusters, across multiple identities"""

    target_db_pattern = os.path.join(target_db_dir, 'clu{identity}.profile')

    os.makedirs(output, exist_ok=True)
    output_pattern = os.path.join(output, 'clu{identity}')

    for identity in ident_list:
        start_time = time.time()

        print('THREADS*******', threads)

        utils.tprint(f"################ Running Identity {identity} ######################")
        if os.path.exists(target_db_pattern.format(identity=identity)):
            mmseqs_search(source_db,
                          target_db_pattern.format(identity=identity),
                          output_pattern.format(identity=identity),
                          threads=threads,
                          splits=splits,
                          sleep_seconds=5)
        else:
            utils.tprint(f"################ SKIPPING Identity {identity} (file doesn't exist probably it had no clusters after merging clusters ######################")

        print(f"Total time to run identity {identity}:", "{:.2f} minutes".format((time.time() - start_time)/60))

def process_mmseqs_merge_search(source_db,
                                target_db_dir,
                                search_results_dir,
                                ident_list,
                                output_raw=None,
                                output=None,
                                output_kws=None):
    """Processes the mmseqs search against the reference plasmid/chromosome database.

    (1) Compiles all mmseqs hits
    (2) Filters for hits that meet >=80 identity and >80% coverage

    source_db : Prefix for the query database
    target_db_dir : Directory containing the nested-merged mmseqs clusters. This directory is used to retrieve target sequence lengths and within-cluster min sequence idetnties
    ident_list : Alignment identities to compile
    output : Location to save formatted table of hits
    """

    #### Read data about databases
    utils.tprint('Reading database info')
    # - Get lengths of query genes (the header table for the HMP assembly genes created when I made the mmseqs database for the genes)
    q_headers = utils.unpickle(str(source_db) + '.all.headers.pkl.blp')
    q_len = q_headers['length'].rename('q_length')
    # - Get lengths of target genes (i.e. the representatives of clusters)
    target_db_dir = Path(target_db_dir)
    t_len = utils.unpickle(target_db_dir / 'rep_lengths.pkl.blp').set_index('representative')['length']
    # - Get the minimum sequence identity of each target gene (i.e. representative) to the other sequences in the cluster
    min_identities = utils.unpickle(target_db_dir / 'rep_min_align_identity.pkl.blp')

    # Compile hits across identities (40 min for 1,782 global metagnomes)
    utils.tprint('Compiling hits across identities')
    search_results_pattern = os.path.join(search_results_dir, 'clu{ident}.m8.pkl.blp')
    hits = pd.concat([shallow_filter(utils.unpickle(search_results_pattern.format(ident=ident)).assign(cluster_identity=ident),
                                     ident, q_len, t_len,
                                     utils.subset(min_identities, identity=ident).set_index('representative')['min_seqIdentity'],
                                     copy=False) \
                      for ident in ident_list if os.path.exists(search_results_pattern.format(ident=ident))], sort=False, ignore_index=True)

    ### Reverse the uniquification of genes that was done before mmseqs search (13 min for 1,782 global metagenomes)
    utils.tprint('Hits over unique genes: {:n}'.format(len(hits)))
    # - Read table mapping unique sequences to their duplicates
    unique_q_headers = utils.unpickle(str(source_db) + '.unique.tsv.pkl.blp')
    # - Take a subset of the unique_q_headers, for equivalent but faster merging with `hits` below
    unique_q_headers_in_hits = utils.subset(unique_q_headers, representative=hits['qId'].values)
    # - Do merging
    hits = hits.set_index('qId')[['eVal','cluster_identity', 'tId']].join(
                unique_q_headers_in_hits.set_index('representative')).reset_index(drop=True).rename(columns={'member':'qId'})
    hits = utils.rearrange(hits, ['qId'], [0])
    utils.tprint('Hits over all genes: {:n}'.format(len(hits)))

    ### Format and save anvio-format table of hits (1 hr 7 min for global metagenomes)
    utils.tprint('Formatting hits')
    # - Convert hits to create a table of gene functions in every assembly contig (anvio format)
    utils.tprint('- Creating mmseqs_x_y accessions with categorical dtype')
    hits['accession'] = pd.Categorical(['mmseqs_{}_{}'.format(x,y) for x,y in zip(hits['cluster_identity'], hits['tId'])])
    # - Merge query header info (e.g. contig, start, stop info)
    utils.tprint('- Merging query header info (e.g. contig, start, stop)')
    q_headers_in_hits = utils.unused_cat(q_headers[utils.isin_int(q_headers.index, hits['qId'].drop_duplicates().values)])
    hits_formatted = q_headers_in_hits.join(hits.set_index('qId')[['eVal','accession']].rename(columns={'eVal':'e_value'})).reset_index(drop=True)

    # - Save hits table to file
    if output_raw is not None:
        utils.tprint('- Saving raw hits table to', output_raw)
        utils.write_table(hits, output_raw, **output_kws)
    if output is not None:
        utils.tprint('- Saving formatted hits table to', output)
        utils.write_table(hits_formatted, output, **output_kws)

    return hits_formatted


def shallow_filter(hits, identity,
                   q_len, t_len, min_identity,
                   copy=True,
                   nuc_lengths=True):
    """Filters mmseqs hits based on coverage and sequence identity.

    Assumes that the search was against a set of profiles of gene clusters. Thus, the target sequences are ij

    q_len : pd.Series. Length of query (nucleotide) sequences

    t_len : pd.Series. Length of target (nucleotide) sequences (i.e. the lengths of representative sequences of clusters whose profiles were searched against).
        -- Note: because profiles were created in the default mode, where columns in the profile match columns in the representative sequence, then the length of the representative sequence equals the length of the profile.

    nuc_lengths : If True, then q_len and t_len are nucleotide lengths (including the stop codon)
        
    min_identities : pd.Dataframe with columns (representative). The minimum identity of each representative to the other sequences in its cluster.
    """

    utils.tprint('Identity:', identity)
    utils.tprint('Hits before thresholds: {:n}'.format(len(hits)))
    if copy:
        utils.tprint('Copying dataframe')
        hits = hits.copy()

    hits['q_length'] = utils.int_loc(hits['qId'].values, q_len)
    hits['t_length'] = utils.int_loc(hits['tId'].values, t_len)

    if nuc_lengths:
        # Calculate the amino acid sequence length by dividing by 3. The correction of -1 accounts for the stop codon, which doesn't become an amino acid
        hits['q_length'] = hits['q_length'] / 3 - 1
        hits['t_length'] = hits['t_length'] / 3 - 1

    # Update 12/17/21: Need to add +1 because these are 1-based closed indices
    hits['tCov'] = (hits['tEnd'] - hits['tStart'] + 1) / hits['t_length'] 
    hits['qCov'] = (hits['qEnd'] - hits['qStart'] + 1) / hits['q_length']

    hits = hits[(hits['tCov'] >= 0.8) & (hits['qCov'] >= 0.8)]
    utils.tprint('Hits after 80% coverage threshold: {:n}'.format(len(hits)))

    # Read the table of all clusters in this identity, and calculate what was the minimum pairwise identity in each cluster. Use this as a threshold
    hits = hits[hits['seqIdentity'].values >= (min_identity.loc[hits['tId']].values - 0.05)]
    utils.tprint('Hits after seqIdentity threshold: {:n}'.format(len(hits)))
    hits = hits.astype({'cluster_identity' : np.uint16, 'q_length' : np.uint32, 't_length' : np.uint32, 'tCov' : np.float32, 'qCov' : np.float32})

    return hits

def get_target_db(target_db):

    if target_db is None:
        target_db = constants.data_dir / 'PlasX_mmseqs_profiles'
        assert os.path.exists(target_db), f"Cannot find precomputed mmseqs profiles at {target_db}. Run `plasx.py setup_de_novo`"
    else:
        assert os.path.exists(target_db), f'Target directory {target_db} does not exist'

    return target_db

def download_pretrained_plasx_model(ver=None, data_dir=None, mmseqs_profiles_url=None, coefficients_url=None):

    import urllib.request, tarfile

    if data_dir is None:
        data_dir = constants.data_dir
        os.makedirs(data_dir, exist_ok=True)
    data_dir = Path(data_dir)
    assert os.path.exists(data_dir)
    utils.tprint(f'Saving data to directory {data_dir}')

    if ver is None:
        ver = 1.0

    if mmseqs_profiles_url is None:
        # mmseqs_profiles_url = 'file:///mnt/data3/supp_tables/PlasX_mmseqs_profiles.tar.gz'
        mmseqs_profiles_url = f"https://zenodo.org/record/5732447/files/PlasX_mmseqs_profiles.tar.gz?download=1"

    if coefficients_url is None:
        # coefficients_url = f"file:///mnt/data3/supp_tables/PlasX_coefficients_and_gene_enrichments.txt.gz"
        coefficients_url = f"https://zenodo.org/api/files/a220a7f1-1813-455c-8905-304a1038b0b3/PlasX_coefficients_and_gene_enrichments.gz?versionId=b5d06fed-1105-4b7c-ae48-f2eb41e55a6a"

    # Download model coefficients and intercept
    # - Get just the filename. Remove stuff after '?' in a web URL
    out_path = data_dir / os.path.basename(coefficients_url).split('?')[0].split('.gz')[0]
    utils.tprint(f'Downloading PlasX model (intercept and coefficients) from {coefficients_url}')
    with urllib.request.urlopen(coefficients_url) as url_file,\
         open(out_path, 'wb') as local_file:
        with gzip.open(url_file, 'rb') as f:
            shutil.copyfileobj(f, local_file)

    # Download mmseqs profiles
    out_path = data_dir / 'PlasX_mmseqs_profiles'
    os.makedirs(out_path, exist_ok=True)
    utils.tprint(f'Downloading and extracting mmseqs profiles from {mmseqs_profiles_url}')
    with urllib.request.urlopen(mmseqs_profiles_url) as url_file:
        with tarfile.open(fileobj=url_file, mode='r:gz') as f:
            f.extractall(out_path)


def annotate_de_novo_families(gene_calls, target_db=None, output_dir=None, ident_list=None, 
                              threads=None, splits=None, output=None, output_kws=None, overwrite=None):
    """Takes in a table of amino acid sequences and searches them against
    a target mmseqs database of gene clusters

    The target database is actually a set of multiple mmseqs
    databases, each of which clusters the same set of reference genes
    but using a different identity threshold.

    output_kws : params sent to pd_utils.write_table() to decide if writing a pickle-compress or text file
    """

    if ident_list is None:
        ident_list = constants.ident_list

    if threads is None:
        threads = utils.get_max_threads()

    if output_kws is None:
        output_kws = {}
        if 'overwrite' not in output_kws:
            output_kws['overwrite'] = overwrite

    # Directory with mmseqs profiles to search against
    target_db_dir = get_target_db(target_db) 

    # Create temporary directory, if one is not specified
    with utils.TemporaryDirectory(name=output_dir, verbose=True, overwrite=overwrite) as output_dir:

        # Setup output paths
        mmseqs_dir = output_dir / 'mmseqs'
        os.makedirs(mmseqs_dir, exist_ok=True)
        mmseqs_source_db = mmseqs_dir / 'source_db'
        gene_calls_out = output_dir / f'gene_calls.blp'
        hits_filtered_out = output_dir / 'hits_filtered.pkl.blp'

        # Write amino acid sequences to a fasta file
        _ = utils.gene_calls_to_fasta(gene_calls, source='prodigal', output=gene_calls_out, compress=True)

        # Create mmseqs database from fasta file
        create_mmseqs_db(gene_calls_out, mmseqs_source_db)

        # Search sequences against target databases of gene clusters
        mmseqs_merge_search(mmseqs_source_db, target_db_dir, mmseqs_dir, ident_list, threads=threads, splits=splits)

        # Process search - retain only hits with >80% query AND target coverage, and with sufficient identity
        hits = process_mmseqs_merge_search(mmseqs_source_db, target_db_dir, mmseqs_dir, ident_list,
                                           output_raw=hits_filtered_out,
                                           output=output,
                                           output_kws=output_kws)

        return hits
