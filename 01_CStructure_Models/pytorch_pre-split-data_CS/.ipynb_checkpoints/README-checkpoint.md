 ## meta data
 clue_compoundinfo_beta.txt 
 clue_siginfo_beta.txt
 clue_geneinfo_beta.txt       
 siginfo_beta.txt
 clue_sig_in_SPECS1&2.csv'
 clue_cellinfo_beta.txt
 
 # Analysis Scripts
 Cell_Line_Correlation_Analysis.ipynb: Averages the transcriptomic profiles of a single MoA in different cell types that have been perturbed and produces a                            correlation matrix.      
 PCA_UMAP_Cmpd_Clustering.py: Produces PCA And UMAP representations to investigate if transcriptomic profiles that come from the perturbation of the same              compound are clustered together in a single MoA class.             
 Cmpds_Tprofiles_Given_CellType_Dosage.py: Produces terminal output that says the number of transcriptomic profiles and unique compounds exist given a cell                          type, dosage, and high or low quality profiles   
 PCA_UMAP_Profile_Clustering.py: Produces PCA And UMAP representations of how transcriptomic profiles cluster according to which MoA class that belong to.     
 
 # other
 PNG_Cell_Line_Investigations: folder to store images produced by Cell_Line_Correlation_Analysis.ipynb