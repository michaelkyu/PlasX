PlasX is a machine learning classifier for identifying plasmid sequences based on genetic architecture.

# Installation

PlasX requires `python >=3.7`. We recommend installing PlasX in a new virtual environment using Anaconda, to make installing dependencies easier.

```
# Create virtual environment named "plasx" (you can change the name to whatever you like)
conda create -n plasx
# Install dependencies
conda install --name plasx -c anaconda -c conda-forge -c bioconda  numpy pandas scipy scikit-learn python-igraph numba python-blosc matplotlib seaborn pysam mmseqs2=10.6d92c
```

Alternatively, create the environment and install dependencies in a single command

```
conda create --name plasx -c anaconda -c conda-forge -c bioconda  numpy pandas scipy scikit-learn python-igraph numba python-blosc matplotlib seaborn pysam mmseqs2=10.6d92c
```

Then, activate the new environment
```
conda activate plasx
```

Finally, download PlasX (e.g. with `git clone`) and then install it with `pip`

```
git clone git@github.com:michaelkyu/PlasX.git
pip install PlasX
```

# Tutorial for predicting plasmids using PlasX

PlasX can be run using the command line (make sure you've activated the correct virtual environment).

```
$ plasx -h
```

```
usage: plasx [-h] {setup,fit,search_de_novo_families,predict} ...

Runs a PlasX model on a set of sequences. A score is returned for each contig.
Plasmids have scores closer to 1, whereas chromosomes have scores closer to 0.

positional arguments:
  {setup,fit,search_de_novo_families,predict}
    setup               Downloads a pretrained PlasX model
    fit                 (Not implemented yet) Trains a new PlasX model
    search_de_novo_families
                        Annotates genes to de novo families
    predict             Runs trained PlasX model on annotated sequences

optional arguments:
  -h, --help            show this help message and exit
(plasx) 
```

In this tutorial, we will identify plasmids using a PlasX model that was pretrained on a set of reference bacterial genomes and plasmid sequences from NCBI. We'll run this model on an example set of 40 contigs in `test/test-contigs.fa`, **but you can repeat these steps with your own contigs**. Running PlasX consists of three steps:

1. Identify genes and annotate COGs and Pfams using [anvio](https://merenlab.org/software/anvio/)
2. Annotate *de novo* gene families
3. Use PlasX to classify contigs as plasmid or non-plasmid sequences, based on the annotations in step #1 and #2

## Step 0. Preliminary setup of your command line environment


```bash
# Change into the directory where PlasX was downloaded (e.g. where you ran `git clone`).
cd /path/to/PlasX
# Change into the `test` subdirectory that contains test-contigs.fa
cd test

# Input and output filename will start with this prefix
PREFIX='test-contigs'

# The number of CPU cores that will be used to annotate genes. We recommend you setting it to a high number, to speed up the processing of many contigs
THREADS=4
```



```bash
# We are going to run PlasX on a fasta file containing 40 sequences. The file looks like this.
head test-contigs.fa
```

    >AST0002_000000019451
    ATCTGATCCATCACGTACAGTGGATCCCCACCGTCATTATAGACCGGCTCCTGCAAACTCATAGGCACCTGTATGGCGTCCAGCGCATAGACAAAGGAGGTTTTCACTGCCTTTTCCGGTGCGCCGCCATGCTCCAGCCAGGAGGAGTTATGATAGTATTCCAGCACATTGACGCTGTTGGAAAAGCACAGGGCAGCGGT...
    
    >AST0002_000000028382
    TTGAGTACCATACATTTTTACATCATTCTCAAACCTCAAATTCAAAAATGAAATGCCATCACCTGCACAAAACTAGGCACTCAAACTGTATACTTTAATAAATTTCACATCTATCTTTATCTATAATATACCGTTCTACTCCCTCCACACAAGCTTTCTCGCTATTTTCAATAAATAATCCTTTTATTTCTATATATTTT...
    
    >AST0002_000000001053
    TCTCTCACGAAACAGCTTATTTATAATATCATCCCTTCTTAAACTTGTCAATAGCTTTTTTTAAAGTTTTGGAACCTTTTTTATTGCTGAACAAGTTATTTGGGACTTTACTAGTTTATCAAGTTTAAGTAATAAAGTCAACAAATTCTTCTTTTTAAAAAACAAAAAGAAGATACTAGTAACCTTTTTCTAGAATTTTA...
    
    ...

## Step 1. Identify genes and annotate COGs and Pfams using [anvio](https://merenlab.org/software/anvio/)
* anvio is a software platform for analyzing omics data. Please install it using instructions at https://merenlab.org/2016/06/26/installation-v2/.
* Here, we use anvio to call and annotate genes. All of the relevant anvio commands are described below. But if you're curious about anvio's other functionalities, you can learn more at https://merenlab.org/tutorials/infant-gut/.

### Create an anvio contigs database


```bash
# Change into the Anaconda environment that contains anvio
# - This step assumes that you installed anvio into a separate Anaconda environment than PlasX. You may modify this step depending on how you installed anvio.
conda deactivate
conda activate anvio
```


```bash
# Create an anvio contigs database from the fasta file
# - The `-L 0` parameter ensures that contigs remain intact and aren't split
anvi-gen-contigs-database -L 0 --project-name $PREFIX -f $PREFIX.fa -o $PREFIX.db

# Export gene calls (including amino acid sequences) to text file
anvi-export-gene-calls --gene-caller prodigal -c $PREFIX.db -o $PREFIX-gene-calls.txt
```


```bash
# You should see something like this
head $PREFIX-gene-calls.txt
```

    gene_callers_id	contig	start	stop	direction	partial	call_type	source	version	aa_sequence
    0	AST0002_000000019451	2	1001	r	1	1	prodigal	v2.6.3	MKKKATYGNESISMLKGADRVRLRPAVIFGSDGLEGCEHAVFEILSNAIDEAREGHGKEILVTRYLDHSIQVEDFGRGCPVDWNPAEQRFNWELVFCELYAGGKYSNNEGENYEYSLGLNGLGACATQYASRYFDAVIRRDGKKYTLHFEKGENIGGLKTEKADRKQTGSCFHWMPDLDVFTDINIPAEYYADILKRQAVVNAGVTFRLRTEMADGSFQETDFCYENGILDYVKELTEGKAMTMPQFWEAERKGRDRADMADYKVKITAALCFSNSVNVLEYYHNSSWLEHGGAPEKAVKTSFVYALDAIQVPMSLQEPVYNDGGDPLYVMDQ
    1	AST0002_000000019451	1152	1908	r	1	1	prodigal	v2.6.3	ALAECLFDDEKNMVRIDMSEYMEKYSVSRLIGAPPGYVGYEEGGQLTEAVRRRPYSVILFDEVEKAHPDVFNVLLQVLDDGRITDSQGRTVDFKNTIIILTSNLGSQYLLDGIGPDGSITQEAKDQVNALLKKSFRPEFLNRLDEIVFYKPLTRDNITSIIDLQMKDLNRRLADKQLSCRLTPEAKHFIIDAAYDPLYGARPLRRYLQHTVETLIAKKILTGDMPMGSVLEIRVEDGELTVVVVMEATIVE
    2	AST0002_000000028382	94	766	r	0	1	prodigal	v2.6.3	MKLVHGDLSGEIINEQKECVEWIIESPDLFSKYVGELYRQFNKDEGKFVLSENNKEIDIAKYSEIIINPLSIEINNRKVLNKLYEELNKLSFNEVLYMKTLELTKLIQEYLLELEQETNYILEFNNEVEMSALFKAVDLKYEDSGEDFLERLVKYIKILVDLLSVKLFVFINARCFMNDEQIKKLCEEIKYIEIKGLFIENSEKACVEGVERYIIDKDRCEIY
    3	AST0002_000000028382	762	1068	r	0	1	prodigal	v2.6.3	MRVLVFFDLPVLTSENRRAYAKFRKFLLKNGFLMLQESVYCKLALNGSAVNAIVDSVHKNKPEEGLIQLLTVTEKQYAKMDIILGQVKSEVLDTDERLVIL
    4	AST0002_000000028382	1072	1321	r	1	1	prodigal	v2.6.3	FNLASDLMEPFRVLVDQEVYNMRLEQFEHEEKMILVNILNKEVQIDGKNQYVNNAIKIYCKSIFDALNEDDSALIRFYKIEL
    5	AST0002_000000001053	197	596	r	0	1	prodigal	v2.6.3	MIKIYTISSCTSCKKAKTWLNKHQLPYKEQNLGKEPLTKEEILTILSKTENGVESIVSKKNRYAKALNCDIDELSVSEVIDLIQENPRILKSPILIDDKRLQVGYKEDDIRAFLPRSIRNVENSQARMRAAL
    6	AST0002_000000001053	772	1927	r	0	1	prodigal	v2.6.3	MAKKTKKTEEITKKFGDERLKALDDALKNIEKDFGKGAVMRLGERAEQKVQVMSSGSLALDIALGAGGYPKGRIIEIYGPESSGKTTVALHAVAQTQKEGGIAAFIDAEHALDPAYAAALGVNIDELLLSQPDSGEQGLEIAGKLIDSGAVDLVVVDSVAALVPRAEIDGDIGDSHVGLQARMMSQAMRKLSASINKTKTIAIFINQLREKVGIMFGNPETTPGGRALKFYASVRLDVRGNTQIKGTGDKKDQNVGKETKIKVVKNKVAPPFKEAFVEIMYGEGISQTGELVKIASDIGIIQKAGAWFSYNGEKIGQGSENAKKYLADHPEIFAEIDHKVRVHYGLVELDEDDVVEDTQVEDTQVEDTSDELILDLDSTIEIEE
    7	AST0002_000000001053	1962	3234	r	0	1	prodigal	v2.6.3	MKAELIAVGTEILTGQITNTNAQFLSEKLAELGIDVYFHTAVGDNENRLLSVLDQSSKRSDLVILCGGLGPTEDDLTKQTLAKFLGKELIFDEEASKKLDSFFATRPKHTRTPNNERQAQIVEGAVPLQNLTGLAVGGIITVEGVTYVVLPGPPSELKPMVNQELIPALTENHTTLYSRVLRFFGVGESQLVTIIKDLIVNQTDPTIAPYAKVGEVILRLSTKASSQEEADRKLDVLEEQIRSTKTLDGKSLSDLIYGYGESNSLAYEVFYLLKKYGKTITAAESLTAGLFQASVADFPGASQVFKGGFVTYSMEEKAKMLDIPLSKLEEHGVVSHFTAEKMAEGARVKTDSDYGIALTGVAGPDALEGHQAGTVFIGIADRNQVRSIKVVIGGRSRSDVRYISTLYAFNLVRQALLQEDNSI
    8	AST0002_000000001053	3708	3885	r	0	1	prodigal	v2.6.3	MVNRCRWVPTDNKLYCDYHDKEWGKPIGDDEKLFELLCLESYQSGLSWLTVLKKTPGF



### Download COG 2014 and Pfam v32 databases


```bash
# Download COGs (~2 min on fast network)
# - The flags `--cog-version COG14` and `--pfam-version 32.0` directs anvio to download the 2014 version of the COG database and v32.0 of Pfam, which are used by PlasX. Without these flags, anvio will by default download the latest versions of COG (v. 2020) and Pfam (v35.0), which PlasX does not use.
anvi-setup-ncbi-cogs --cog-version COG14 --cog-data-dir COGS_2014 -T $THREADS

# Download Pfam (~20 min on fast network)
# - The flags `--cog-data-dir COG2014` and `--pfam-data-dir Pfam_v32` directs anvio to store the COG and Pfam databases in new subfolders named `COG2014` and `Pfam_v32`, respectively.
anvi-setup-pfams --pfam-version 32.0 --pfam-data-dir Pfam_v32
```

### Annotate COGs and Pfams and export to text files


```bash
# Annotate COGs
anvi-run-ncbi-cogs -T $THREADS --cog-version COG14 --cog-data-dir COG_2014 -c $PREFIX.db

# Annotate Pfams
anvi-run-pfams -T $THREADS --pfam-data-dir Pfam_v32 -c $PREFIX.db

# Export functions to text file
anvi-export-functions --annotation-sources COG14_FUNCTION,Pfam -c $PREFIX.db -o $PREFIX-cogs-and-pfams.txt
```


```bash
# You should see something like this
head $PREFIX-cogs-and-pfams.txt
```

    gene_callers_id	source	accession	function	e_value
    0	COG14_FUNCTION	COG0187	DNA gyrase/topoisomerase IV, subunit B	4.4e-136
    1	COG14_FUNCTION	COG0542	ATP-dependent Clp protease ATP-binding subunit ClpA	1.7e-95
    3	COG14_FUNCTION	COG3512	CRISPR/Cas system-associated protein Cas2, endoribonuclease	2.1e-28
    4	COG14_FUNCTION	COG1518	CRISPR/Cas system-associated endonuclease Cas1	3.1e-30
    5	COG14_FUNCTION	COG1393	Arsenate reductase and related proteins, glutaredoxin family	1.9e-61
    6	COG14_FUNCTION	COG0468	RecA/RadA recombinase	4.4e-180
    7	COG14_FUNCTION	COG1058!!!COG1546	Predicted nucleotide-utilizing enzyme related to molybdopterin-biosynthesis enzyme MoeA!!!Nicotinamide mononucleotide (NMN) deamidase PncC	7.8e-170
    8	COG14_FUNCTION	COG2818	3-methyladenine DNA glycosylase Tag	3.1e-21
    9	COG14_FUNCTION	COG0632	Holliday junction resolvasome RuvABC DNA-binding subunit	6.8e-84



## Step 2. Annotate *de novo* gene families

### Change into the Anaconda environment that contains PlasX


```bash
conda deactivate
conda activate plasx
```

### Download database of *de novo* families and the set of coefficients that define the PlasX model.


```bash
# ~5 min download on fast network
plasx setup \
    --de-novo-families 'https://zenodo.org/record/5819401/files/PlasX_mmseqs_profiles.tar.gz?download=1' \
    --coefficients 'https://zenodo.org/record/5819401/files/PlasX_coefficients_and_gene_enrichments.txt.gz?download=1'
```

### Annotate genes using pretrained database of *de novo* gene families


```bash
plasx search_de_novo_families \
    -g $PREFIX-gene-calls.txt \
    -o $PREFIX-de-novo-families.txt \
    --overwrite
```


```bash
# You should see something like this
head $PREFIX-de-novo-families.txt
```

    gene_callers_id	contig	start	stop	direction	rev_compd	length	e_value	accession
    1	AST0002_000000019451	1152	1908	r	True	756	0.0	mmseqs_40_33078316
    1	AST0002_000000019451	1152	1908	r	True	756	0.0	mmseqs_30_43406241
    1	AST0002_000000019451	1152	1908	r	True	756	0.0	mmseqs_25_49900063
    1	AST0002_000000019451	1152	1908	r	True	756	0.0	mmseqs_20_50193611
    2	AST0002_000000028382	94	766	r	True	672	6.194e-19	mmseqs_30_33026007
    2	AST0002_000000028382	94	766	r	True	672	4.061e-32	mmseqs_20_18536381
    2	AST0002_000000028382	94	766	r	True	672	1e-45	mmseqs_5_34842522
    3	AST0002_000000028382	762	1068	r	True	306	9.561e-31	mmseqs_40_40392677
    3	AST0002_000000028382	762	1068	r	True	306	1.752e-36	mmseqs_30_46987655




## Step 3. Use PlasX to classify contigs as plasmid or non-plasmid sequences, based on the annotations in step #1 and #2
PlasX assigns a score to every contig. This score ranges from 0 (likely not plasmid) to 1 (likely plasmid). We recommend applying a threshold of at least >0.5 to identify plasmids. You can raise this threshold even higher (e.g. >0.9) to filter for high-confidence plasmids.


```bash
plasx predict \
    -a $PREFIX-cogs-and-pfams.txt $PREFIX-de-novo-families.txt \
    -g $PREFIX-gene-calls.txt \
    -o $PREFIX-scores.txt \
    --overwrite
```


```bash
# The output should look like this. The first column are the contig names, and the second column is the PlasX score.
# - In this example, PlasX reports that AST0002_000000001053 is probably not a plasmid (score is close to 0), while AST0002_000000008287 is probably a plasmid (score is close to 1)
head $PREFIX-scores.txt
```

    contig	score
    AST0002_000000001053	2.3737687782754214e-08
    AST0002_000000001122	2.995346606467998e-07
    AST0002_000000001459	6.333592291893041e-06
    AST0002_000000001776	0.9592232672966521
    AST0002_000000003048	0.5622421419516015
    AST0002_000000006389	0.6089640039139832
    AST0002_000000006967	0.2730847890122725
    AST0002_000000007621	0.6630016070098446
    AST0002_000000008287	0.9848824181163179
