import argparse
import textwrap

from plasx import constants, compress_utils

def setup(args):
    from plasx.mmseqs import download_pretrained_plasx_model
    download_pretrained_plasx_model(data_dir=args.data_dir,
                                    mmseqs_profiles_url=args.mmseqs_profiles_url,
                                    coefficients_url=args.coefficients_url)

def predict(args):
    from plasx import utils
    from plasx.model import PlasX_model

    if args.model is None:
        args.model = constants.data_dir / 'PlasX_coefficients_and_gene_enrichments.txt'

    utils.tprint('Loading model from {}'.format(args.model))
    try:
        model = PlasX_model.from_table(args.model)
    except FileNotFoundError:
        raise FileNotFoundError("No PlasX model found. Need to specify a table with PlasX coefficients, or download this table via `plasx setup` to the PlasX install directory.")
#    model = PlasX_model.load(args.model)

    utils.tprint('Running model')
    scores = model.predict(args.annotations,
                           verbose=True,
                           gene_calls=args.gene_calls,
                           output=args.output,
                           output_kws=dict(overwrite= 2 if args.overwrite else 1))

def search(args):
    from plasx.mmseqs import annotate_de_novo_families

    compress_utils.set_threads(args.threads)

    annotate_de_novo_families(args.gene_calls,
                              output=args.output,
                              output_dir=args.tmp,
                              target_db=args.target_db,
                              overwrite= 2 if args.overwrite else 1,
                              output_kws=dict(txt=True),
                              threads=args.threads,
                              splits=args.splits,
                              clean_tmp=not args.save_tmp)

def fit(args):
    raise Exception("Not Implemented")
    print(args)

def get_parser():   
    description = textwrap.dedent(\
"""Runs a PlasX model on a set of sequences. A score is returned for
each contig. Plasmids have scores closer to 1, whereas chromosomes
have scores closer to 0.""")
 
    parser = argparse.ArgumentParser(
        description=description)
    parser.set_defaults(func=lambda args: parser.print_help())
    
    # Add subparsers
    subparsers = parser.add_subparsers()

    setup_parser = subparsers.add_parser('setup', help='Downloads a pretrained PlasX model')
    setup_parser.set_defaults(func=setup)
    required = setup_parser.add_argument_group('required arguments')
    optional = setup_parser.add_argument_group('optional arguments')
    optional.add_argument(
        '--de-novo-families', dest='mmseqs_profiles_url', default=None,
        help="""URL of precomputed mmseqs profiles""")
    optional.add_argument(
        '--coefficients', dest='coefficients_url', default=None,
        help="""URL of precomputed coefficients and intercept of logistic regression""")
    optional.add_argument(
        '-o', dest='data_dir', default=None,
        help="""Directory to save model. Default: in the install location of PlasX.""")
    
    fit_parser = subparsers.add_parser('fit', help='(Not implemented yet) Trains a new PlasX model')
    fit_parser.set_defaults(func=fit)

    ## Search parser
    search_parser = subparsers.add_parser('search_de_novo_families', help='Annotates genes to de novo families. Uses MMseqs2 for fast sequence alignment.')
    search_parser.set_defaults(func=search)
    required = search_parser.add_argument_group('required arguments')
    optional = search_parser.add_argument_group('optional arguments')
    required.add_argument(
        '-g', '--gene-calls', dest='gene_calls', required=True,
        help='Table of gene calls, with amino acid sequences')
    required.add_argument(
        '-o', '--output', dest='output', required=True,
        help="""File to save predictions""")
    optional.add_argument(
        '-db', dest='target_db', default=None,
        help="""Location of precomputed MMseqs2 profiles. You only need to specify this if you downloaded the PlasX model to a custom location using the '-o' flag in `plasx setup`. Default: PlasX will search the default download location in its installation diretory.""")
    optional.add_argument(
        '-T', '--threads', dest='threads', type=int, default=None,
        help="""Number of threads to run MMseqs2 with. Default: The number of CPUs available on this machine.""")
    optional.add_argument(
        '-S', '--splits', dest='splits', type=int, default=1,
        help="""This will split PlasX's set of de novo families into an equal number of chunks for searching, reducing the maximum RAM usage. Searching will fail if your machine has insufficient RAM, so you should consider increasing the number of splits. If you have only ~8Gb of RAM, then we recommend settings -S to 32. Setting -S to 0 will let MMseqs2 decide the number of splits automatically. Default: 1 split.""")
    optional.add_argument(
        '--tmp', dest='tmp', default=None,
        help="""Directory to save temporary intermediate files, including ones created by MMseqs2. This directory is deleted upon termination, unless '--save-tmp' is specified. Default: a directory with a random name is created (e.g. inside /tmp)""")
    optional.add_argument(
        '--save-tmp', dest='save_tmp', action='store_true',
        help="""Do NOT delete the directory of intermediate files upon termination. Note: these files use a lot of disk storage, so only keep them if needed for debugging.""")
    optional.add_argument(
        '--overwrite', dest='overwrite', action='store_true', default=False,
        help="""Overwrite files""")


    ## Predict parser
    predict_parser = subparsers.add_parser('predict', help='Runs trained PlasX model on annotated sequences')
    predict_parser.set_defaults(func=predict)
    required = predict_parser.add_argument_group('required arguments')
    optional = predict_parser.add_argument_group('optional arguments')
    required.add_argument(
        '-a', '--annotations', dest='annotations', nargs='+', required=True,
        help='Table of gene annotations to COGs, Pfams, and de novo families')
    required.add_argument(
        '-o', '--output', dest='output', required=True,
        help="""File to save predictions. If not set, then predictions are printed to stdout.""")
    optional.add_argument(
        '-m', '--model', dest='model',
        help='Pretrained PlasX model')
    optional.add_argument(
        '-g', '--gene-calls', dest='gene_calls', default=None,
        help='Table of gene calls, mapping gene_callers_id to contig')
    optional.add_argument(
        '--overwrite', dest='overwrite', action='store_true', default=False,
        help="""Overwrite files""")

    return parser

def run(args=None):
    parser = get_parser()

    args = parser.parse_args(args=args)
    args.func(args)

if __name__=='__main__':
    run()
