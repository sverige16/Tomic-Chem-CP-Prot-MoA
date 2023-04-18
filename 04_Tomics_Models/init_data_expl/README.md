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
investig_gene_meta.ipynb: looking at metadata from clue_geneinfo_beta.txt
investig_cell_lines.ipynb: looking at the metadata from clue_cellinfo_beta.txt
investig_siginfo.ipynb:  looking at the metadatate from clue_siginfo_beta.txt
Cell_Line_Analysis.ipynb: notebook used to create Erik_Cell-Line_Correlation. DEPRECATED

Erik_Cell-Line_Correlation.ipynb: notebook used to assess correlation of transcriptomic profiles
	within a single MoA. FUNCTIONAL
## python scripts
Erik_UMAP_compound_clustering.py: Performing PCA and UMAP to see if transcriptomic profiles caused by
	 a compounds cluster near each other within an MoA class
Erik_cmps-tprofiles_given_celltype-dosage: Finding the number of compounds per MoA class given
	 restrictions put on cell type (prefer high correlation) and dosages (uniform dosages better)
Erik_PCA_UMAP.py: Assessing whether we can visualize separation of MoA classes using transcriptomic 
	profiles by using using dimensionality reduction techniques 
## subdirectories
PNG_Cell_Line_Investigations: location where the .png files are placed when created 
	from Erik_Cell-Line_Correlation.ipynb
## links on GCTX, GCT and clue.io data


### Using pandasGEXpress with .gct/x files
https://github.com/cmap/cmapPy/blob/master/tutorials/cmapPy_pandasGEXpress_tutorial.ipynb
### Documefrntation on pandasGEXpress
https://clue.io/cmapPy/pandasGEXpress.html
### CMAPpy GitHub
https://github.com/cmap/cmapPy
### Information behind connectopedia
https://clue.io/connectopedia/
### Where data is located from
https://clue.io/data/CMap2020#LINCS2020
