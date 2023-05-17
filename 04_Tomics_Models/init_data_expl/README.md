## files for accessing transcriptomics data
clue_cellinfo_beta.txt: Metadata for each cell line that was used in the experiments
clue_compoundinfo_beta.txt: Metadata for each perturbagen that was used in the experiments
clue_geneinfo_beta.txt: Metadata for each measured feature / gene (metadata for rows of the data matrices)
clue_siginfo_beta.txt: Metadata for each signature in the Level 5 matrix (metadata for the columns 
	in the Level 5 data matrix)

## notebooks for looking at data
Cell-Line_Correlation.ipynb: notebook used to assess correlation of transcriptomic profiles
	within a single MoA. 
## python scripts
PCA_UMAP_Cmpd_ Clustering.py: Performing PCA and UMAP to see if transcriptomic profiles caused by
	 a compounds cluster near each other within an MoA class
Cmpds_Tprofiles_Given_CellType_Dosage: Finding the number of compounds per MoA class given
	 restrictions put on cell type (prefer high correlation) and dosages (uniform dosages better)
PCA_UMAP_Profile_Clustering.py: Assessing whether we can visualize separation of MoA classes using transcriptomic 
	profiles by using using dimensionality reduction techniques 
## subdirectories
PNG_Cell_Line_Investigations: location where the .png files are placed when created 
	from Erik_Cell-Line_Correlation.ipynb
    
## drawio
Initial_Data_Exploration.drawio: Drawio explaining what each script does and the analysis it produces in flowchart form.

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
