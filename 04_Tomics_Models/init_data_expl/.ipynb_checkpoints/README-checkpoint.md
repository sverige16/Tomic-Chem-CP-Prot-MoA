## files for accessing transcriptomics data
clue_cellinfo_beta.txt: Metadata for each cell line that was used in the experiments
clue_compoundinfo_beta.txt: Metadata for each perturbagen that was used in the experiments
clue_geneinfo_beta.txt: Metadata for each measured feature / gene (metadata for rows of the data matrices)
clue_siginfo_beta.txt: Metadata for each signature in the Level 5 matrix (metadata for the columns 
	in the Level 5 data matrix)

## notebooks for looking at data
gene_expression_data.ipynb: looking at which compounds/transcriptomic profiles that exist and if we have enough data 
	to perform deep learning.
testing_GCTX_extract.ipynb: testing the various methods to extract specific transcriptomic profiles from GCTX files

## links on GCTX, GCT and clue.io data


### Using pandasGEXpress with .gct/x files
https://github.com/cmap/cmapPy/blob/master/tutorials/cmapPy_pandasGEXpress_tutorial.ipynb
### Documentation on pandasGEXpress
https://clue.io/cmapPy/pandasGEXpress.html
### CMAPpy GitHub
https://github.com/cmap/cmapPy
### Information behind connectopedia
https://clue.io/connectopedia/
### Where data is located from
https://clue.io/data/CMap2020#LINCS2020
