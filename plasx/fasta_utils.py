import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from plasx import utils, compress_utils

def gene_calls_to_fasta(gene_calls, source=None, output=None, compress=None):
    """Converts a table of gene calls (where one column is amino acid sequence) into a fasta string

    The headers in the fasta are formatted as (inspired by anvio's format)
    
    >{id}|contig:{contig_name}|start:{start}|stop:{stop}|direction:{direction}|rev_compd:{rev_compd}|length:{length}

    Note: This function is called as a subroutine in get_gene_sequences(), but it can also be called independently.
    """

    gene_calls = utils.read_table(gene_calls)

    gene_calls = gene_calls.dropna(subset=['aa_sequence'])
    if source is not None:
        gene_calls = gene_calls[gene_calls['source']==source]

    fasta_items = '>' + gene_calls['gene_callers_id'].astype(str) + \
                  '|contig:' + gene_calls['contig'].astype(str) + \
                  '|start:' + gene_calls['start'].astype(str) + \
                  '|stop:' + gene_calls['stop'].astype(str) + \
                  '|direction:' + gene_calls['direction'].astype(str) + \
                  '|rev_compd:' + (gene_calls['direction']=='r').astype(str) + \
                  '|length:' + (gene_calls['stop'] - gene_calls['start']).astype(str) + \
                  '\n' + gene_calls['aa_sequence']                

    fasta = '\n'.join(fasta_items)

    if output is not None:
        # Write fasta into a plain text file (optionally, compressed)
        if compress:
            compress_utils.write_compressed_txt(fasta, output)
        else:
            with open(output, 'wb') as f:
                f.write(fasta+'\n')
    
    return fasta

def subset_fasta(headers, fa_path):
    """Return a subset of the sequences in a fasta file.
    
    headers : The headers of the desired sequences
    
    fa_path : path to fasta file    
    """
    
    raise Exception('Deprecated. Use utils.get_fasta_dict()')

    headers = set(headers)
    sequences = []

    from Bio import SeqIO
    for seq in SeqIO.parse(fa_path, 'fasta'):
        if seq.name in headers:
            sequences.append(seq)
    return sequences

def run_alv(fa, width=None):

    if width is None:
        width = 150

    with tempfile.NamedTemporaryFile('wt') as f:
        f.write(fa)
        f.flush()
#        alv_proc = subprocess.run(['/scratch/miniconda/envs/anvio-master-zelda/bin/alv', '-k', '-w', str(width), f.name],

#        alv_proc = subprocess.run(['alv', '-k', '-w', str(width), f.name], capture_output=True)
        alv_proc = subprocess.run(' '.join(['alv', '-k', '-w', str(width), f.name]), capture_output=True, shell=True)

        alv_proc.check_returncode()
        alv_output = alv_proc.stdout.decode()
    return alv_output

def muscle(sequences, intype=None, rettype='fasta', width=None, headers=None, cmd=None):
    #child = subprocess.Popen('/scratch/miniconda/envs/anvio-master-zelda/bin/muscle',
    if cmd is None:
        cmd = 'muscle'
#        cmd = 'muscle -maxiters 2 -diags1'
#        cmd = 'muscle -maxiters 1 -diags1 -sv -distance1 kbit20_3'

#        cmd = shlex.split(cmd)
        shell = True

    if isinstance(sequences, dict):
#        print(sequences.items())
        headers, sequences = list(zip(*sequences.items()))
    elif (intype is None) or (intype=='seq_list'):
        if headers is None:
            headers = ('seq%s' % i for i in range(len(sequences)))
    else:
        raise Exception("Unsupported input format")

    child = subprocess.Popen(cmd,
                             stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             universal_newlines=True,
                             shell=shell)

    for h, seq in zip(headers, sequences):
        if os.path.isfile(seq):
            from Bio import SeqIO
            tmp = SeqIO.read(seq, 'fasta')
            h, seq = tmp.name, tmp.seq
        child.stdin.write('>%s\n%s\n' % (h, seq))
    child.stdin.close()

    muscle_fa = child.stdout.read()

    if rettype=='fasta':
        return muscle_fa
    elif rettype=='alv':
        return run_alv(muscle_fa, width=width)
    else:
        raise Exception("Unsupported output format")

def fasta_dict_2_str(fasta_dict):
    """Converts dictionary mapping fasta headers to sequences, to a fasta string"""

    return '\n'.join('>{}\n{}\n'.format(h,s) for h,s in fasta_dict.items()) + '\n'

def write_fasta(fasta, output, separate=False, compressed=False, input_type=None):
    """Writes fasta to file.  Can take in a fasta string or dictionary.

    output:

        Write all sequences to this file (or directory, if separate=True)

    separate:

        If True, write each sequence into a separate file into the
        output directory. Each file will be named as <header>.fa

    """
    
    if input_type is None:
        if isinstance(fasta, pd.Series):
            fasta = fasta.to_dict()
            input_type = 'dict'
        elif isinstance(fasta, dict):
            input_type = 'dict'
        elif isinstance(fasta, str):
            input_type = 'str'
        else:
            raise Exception('Cannot infer fasta input type')
    assert input_type in ['dict', 'str']

    if separate:
        if input_type == 'str':
            fasta = get_fasta_dict(fasta, input_type=='str')

        os.makedirs(output, exist_ok=True)
        for h, s in fasta.items():
            write_fasta('>{}\n{}\n'.format(h,s),
                        output=os.path.join(output, '{}.fa'.format(h)),
                        compressed=compressed)
    else:
        if input_type == 'dict':
            fasta = fasta_dict_2_str(fasta)

        if compressed:
            compress_utils.write_compressed_txt(fasta, output)
        else:
            with open(output, 'wt') as o:
                o.write(fasta)

def get_fasta_dict(fasta, input_type=None, input_fmt='fasta', subset=None, verbose=False, n_jobs=None):
    """Reads in the filename or the fasta string itself, and outputs a
    dictionary mapping fasta headers to sequences.

    fasta :

        Can be a list of inputs, each of which are opened and then all fasta headers+sequences are concatenated in a single dictionary.

    input_type :

        If 'str', then `fasta` is interpreted as the fasta string itself

        If 'filename', then `fasta` is interpreted as a file path

        If 'pickle', then 'fasta' is interpreted as a pickled file

        If None, then the type is inferred.

    """

    if input_type is None:
        ## Infer input type
        if isinstance(fasta, (str, Path)):
            if os.path.exists(fasta):
                fasta = str(fasta)

                if fasta[-8:]=='.pkl.blp':
                    input_type = 'pickle'
                elif fasta[-4:]=='.blp':
                    input_type = 'blp'
                elif fasta[-3:]=='.gz':
                    input_type = 'gz'
                else:
                    input_type = 'filename'
            else:
                input_type = 'str'
        elif hasattr(fasta, '__iter__'):
            if isinstance(fasta, dict):
                input_type = 'dict'
            else:
                input_type = 'list'

    if input_type=='list':
        if n_jobs is None:
            n_jobs = 1
        n_jobs = min(n_jobs, len(fasta))

        fasta_dict = {}
        if n_jobs==1:
            for f in fasta:
                fasta_dict.update(get_fasta_dict(f, subset=subset, verbose=verbose))

        else:
            from joblib import Parallel, delayed
            results = Parallel(n_jobs=n_jobs)(delayed(get_fasta_dict)(f, subset=subset, verbose=10) for f in fasta)
            for f in results:
                fasta_dict.update(f)
    else:
        assert not isinstance(subset, str), Exception('`subset` cannot be a string.') 
        
        parse_with_SeqIO = True

        if input_type=='pickle':
            if verbose: print('Reading {}'.format(fasta))
            fasta = compress_utils.unpickle(fasta)
            if isinstance(fasta, bytes):
                fasta = fasta.decode()

            if isinstance(fasta, dict):
                return fasta

            import io
            fasta = io.StringIO(fasta)
        elif input_type=='blp':
            fasta = compress_utils.read_compressed_txt(fasta)
            import io
            fasta = io.StringIO(fasta)
        elif input_type=='gz':
            import gzip
            fasta = gzip.open(fasta, 'rt')
        elif input_type=='str':
            import io
            fasta = io.StringIO(fasta)
        elif input_type=='filename':
            if verbose: print('Reading {}'.format(fasta))
        elif input_type=='dict':            
            parse_with_SeqIO = False
        else:
            raise Exception()

        if parse_with_SeqIO:
            from Bio import SeqIO
            fasta_list = ((record.id, str(record.seq)) for record in SeqIO.parse(fasta, input_fmt))
        else:
            fasta_list = fasta.items()

        if isinstance(subset, str):
            raise Exception('`subset` cannot be a string.') 
        elif subset is None:
#            fasta_dict = {record.id : str(record.seq) for record in itertools.islice(SeqIO.parse(fasta, input_fmt), 0, 10000000)}
#            fasta_dict = {record.id : str(record.seq) for record in SeqIO.parse(fasta, input_fmt))}
            fasta_dict = {header : seq for header, seq in fasta_list}
        elif callable(subset):
#            fasta_dict = {record.id : str(record.seq) for record in SeqIO.parse(fasta, input_fmt) if subset(record.id)}
            fasta_dict = {header : seq for header, seq in fasta_list if subset(header)}
        elif hasattr(subset, '__iter__'):
#            fasta_dict = {record.id : str(record.seq) for record in SeqIO.parse(fasta, input_fmt) if record.id in subset}
            fasta_dict = {header : seq for header, seq in fasta_list if header in subset}

    return fasta_dict        



def concat_fasta_files(input_list, output, compress='infer', verbose=True):
    """
    Reads sequences from multiple fasta files, and concatenates them into a single file

    TODO: merge this with `concat_files`
    """
    
    with open(output, 'w') as f:
        for i, fa_path in enumerate(input_list):
            if verbose and (i%100==0):  print(i, end=',')
            if compress=='blp' or (compress=='infer' and str(fa_path)[-4:]=='.blp'):
                fa = compress_utils.read_compressed_txt(fa_path)
            else:
                with open(fa_path, 'r') as g:
                    fa = g.read()
            f.write(fa)
            f.write('\n')                    
    if verbose: print()

def concat_files(input_files, output_file, filesep=None):
    """
    Concatenates files together, byte for byte.

    Optionally, append an string (e.g. newline) between files. Be careful that newlines are encoded appropriately.

    Code inspired by https://stackoverflow.com/questions/13613336/python-concatenate-text-files
    """

    if isinstance(filesep, str):
        # Attempt to encode string into bytes
        filesep = filesep.encode()
    
    with open(output_file,'wb') as wfd:
        for f in input_files:
            with open(f,'rb') as fd:
                shutil.copyfileobj(fd, wfd)
                
            if filesep is not None:
                wfd.write(filesep)

def index_fasta_headers(input_fa, output_fa, start_count=0, header_filter=None):
    """Formats a fasta file, so that the sequence headers are replaced
    with consecutive integers, e.g. 0,1,2,...

    TO_IMPLEMENT: if `nr` is True, then redundant sequences are checked and removed (they are given the same index)

    input_fa : input fasta file

    output_fa : output fasta file, with renamed headers
    
    header_file : Two column table, with first column being the
    original header, and the second column being the new header
    
    start_count : The start of the header indexing (Default: 0)

    Returns
    --------

    headers : list of the original headers, in the same order as the input

    """
    count = start_count
    headers = []
    headers_df_list = []

    max_headers_per_chunk = 5000000

    if isinstance(input_fa, (str, Path)):
        input_fa = [ input_fa ]

    with open(output_fa, 'wt') as out:
        for i_fa_idx, i_fa in enumerate(input_fa):
            if (i_fa_idx%100)==0: utils.tprint(i_fa_idx, end=', ')

            i_fa = str(i_fa)

            try:
                if i_fa[-4:]=='.blp':
                    open_file = False
                    f = compress_utils.read_compressed_txt(i_fa).splitlines()
                else:
                    open_file = True
                    f = open(i_fa, 'rt')

                keep_sequence = False
                for i, x in enumerate(f):
                    x = x.strip() # Remove newlines

                    if len(x)>0:
                        if x[0]=='>':
                            x = x[1:]
                            
                            keep_sequence = (header_filter is None) or header_filter(x)
                            if keep_sequence:
                                if count > start_count:
                                    out.write('\n')

                                # Write sequence to file
                                out.write('>%s\n' % count)
                                headers.append(x)
                                count += 1

                                if len(headers) == max_headers_per_chunk:
                                    utils.tprint('Read %s sequences' % (count - start_count))
                                    headers_df_list.append(fasta_headers_to_table(headers))
                                    if len(headers_df_list) > 0:
                                        headers_df_list = [ utils.better_pd_concat(headers_df_list, ignore_index=True) ]
                                    headers = []
                        elif keep_sequence:
                            out.write(x)
            finally:
                if open_file:
                    f.close()

    headers_df_list.append(fasta_headers_to_table(headers))
    headers_df = utils.better_pd_concat(headers_df_list, ignore_index=True)

    if start_count != 0:
        # The default index is 0,1,2,..., because "ignore_index=True" is passed to pd.concat
        # If start_count is not 0, then manually need to set the index
        headers_df = np.arange(start_count, start_count + headers_df.shape[0])

    return headers_df

def fasta_headers_to_table(headers):
    """Takes the fasta headers that are output by anvi-get-sequences-for-gene-calls and parses them out into a table. 
    
    Table has memory-efficient data types, i.e. strings are converted into categoricals
    """
    
    df = pd.Series(headers).str.split('|', expand=True)
    df.columns = ['gene_callers_id', 'contig', 'start', 'stop', 'direction', 'rev_compd', 'length']
    df['gene_callers_id'] = df['gene_callers_id'].astype(np.int64)
    for c in df.columns[1:]:
        x = df[c].str.split(':', expand=True)[1]
        if c in ['contig', 'direction']:
            df[c] = x.astype('category')
        elif c in ['start', 'stop', 'length']:
            df[c] = x.astype(np.dtype('int64'))
        elif c=='rev_compd':
            #df[c] = x.astype('bool')
            df[c] = (x == 'True').astype('bool')
    return df
