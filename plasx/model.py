import os

import numpy as np
import pandas as pd

from plasx import utils, constants, learn

class PlasX_model:
    """Consists of a logistic regression set of coefficients, and a list of annotation sources."""

    def __init__(self, coef, intercept, features, params):

        self.coef = coef
        self.intercept = intercept
        self.features = features
        self.params = params

    @classmethod
    def load(cls, path):
        """First try unpickling, and if that doesn't work, then try parsing it as a text table"""

        try:
            return utils.unpickle(path)
        except:
            return cls.from_table(path)

    def save(self, path):
        utils.pickle(self, path)

    def __getstate__(self):
        return utils.subset_dict(self.__dict__.copy(), ['coef', 'intercept', 'features', 'params'])

    def __setstate__(self, state):
        self.__dict__.update(state)
        
    @classmethod
    def from_sklearn(cls, model, features):
        instance = cls(utils.as_flat(model.coef_), model.intercept_[0], features, model.get_params())
        
        # Remember the sklearn class name
        instance.params['class'] = str(type(model))

        return instance

    @classmethod
    def from_table(cls, path=None):
        if path is None:
            path = constants.data_dir / 'PlasX_coefficients_and_gene_enrichments.txt'
        if not os.path.exists(path):
            raise FileNotFoundError

        df = utils.read_table(path).set_index('accession')['PlasX_coefficient']
        assert 'INTERCEPT' in df.index,\
            "One of the accessions needs to be the word 'INTERCEPT', which represents the intercept of the logistic regression"
        df_coef = df.drop(index='INTERCEPT')

        instance = cls(list(df_coef.values), df.loc['INTERCEPT'], list(df_coef.index), {})
        return instance

    def _get_feature_matrix(self, C, contig_column, annotation_column, shift_1=False, verbose=False):
        """Converts a long-format table of gene family annotations into a contig-by-gene-family feature matrix.

        Return type is a scipy sparse csr matrix."""

        import scipy.sparse

        C = C.astype({col : 'category' for col in [contig_column, annotation_column]})

        # Remove annotations that are not in the vocab
        C = C[utils.catin(C[annotation_column], self.features)].copy()
        C[annotation_column] = C[annotation_column].cat.set_categories(self.features)

        utils.tprint('Pivoting annotation table into a contig-by-gene feature matrix', verbose=verbose)
        X, rownames, colnames = utils.sparse_pivot(C, index=contig_column, columns=annotation_column, rettype='spmatrix')
        rownames, colnames = np.array(rownames), np.array(colnames)

        # Check that categorical ordering of accessions is consistent with the model coefficients
        assert np.all(self.features == colnames)
        assert(len(self.coef) == len(colnames))

        # adds 1 to the accession values so that 0 is reserved to
        # designate padding (and then recognized for masking)
        # -- See learn.get_data() where this is done too
        if shift_1:
            # A dummy column of zeros
            pad_column = scipy.sparse.csr_matrix(([],([],[])),(X.shape[0],1), dtype=X.dtype)
            X = scipy.sparse.hstack([pad_column, X])
            colnames = np.append('PAD', colnames)

        X = scipy.sparse.csr_matrix(X)

        return X, rownames, colnames

    def predict(self, annotations, verbose=None, contig_column=None, annotation_column=None, gene_calls=None,
                output=None, output_kws=None):

        # If going to save file, check that overwriting is okay now, before doing more computation
        if (output is not None) and ('overwrite' in output_kws):
            utils.check_overwrite(output, output_kws['overwrite'])

        # Set default names of contig/annotation-columns
        if contig_column is None:
            contig_column = 'contig'
        annotation_column_was_none = False
        if annotation_column is None:
            annotation_column_was_none = True
            annotation_column = 'accession'

        if gene_calls is not None:
            gene_calls = utils.read_table(gene_calls, verbose=verbose)

        def map_gene_callers_id_to_contig(functions):
            """Maps gene_callers_id column to contig.
            
            Example use case: a table from anvi-export-functions
            contains only gene_callers_id. The mapping to contigs can
            be found from anvi-export-gene-calls
            """

            if (gene_calls is not None) and (contig_column not in functions.columns):
                functions = functions.merge(gene_calls[[contig_column, 'gene_callers_id']], on=['gene_callers_id'])

            return functions
                
        try:
            C = utils.read_table(annotations,
                             cols=[contig_column, annotation_column],
                             verbose=verbose,
                             post_apply=map_gene_callers_id_to_contig)
        except KeyError:
            if annotation_column_was_none:
                annotation_column = 'annotation'
            C = utils.read_table(annotations,
                     cols=[contig_column, annotation_column],
                     verbose=verbose,
                     post_apply=map_gene_callers_id_to_contig)


        # Process annotations table by string splitting '!!!'
        C = utils.df_str_split(C, '!!!', annotation_column)

        # Create contig-by-gene-family feature matrix
        utils.tprint('Formatting table', verbose=verbose)
        X, rownames, colnames = self._get_feature_matrix(C, contig_column, annotation_column)

        # Run model by applying logistic transformation
        utils.tprint('Calculating logistic transformation', verbose=verbose)
        pred = learn.lr_transform_base(self.coef, self.intercept, X, prob=True)

        # Format prediction scores into a pd.Series with the contigs as index
        utils.tprint('Creating pd.Series', verbose=verbose)
        pred = pd.Series(pred, index=rownames)

        print('output:', output)

        utils.write_table(pred.rename_axis('contig').rename('score').reset_index(),
                          output,
                          txt=True,
                          **output_kws)

        return pred
