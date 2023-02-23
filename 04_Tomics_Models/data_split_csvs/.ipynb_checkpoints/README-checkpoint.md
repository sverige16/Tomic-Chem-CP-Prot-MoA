# Format for data splitting

### For 2 MoAs
Insert 3 letter abbreviations for both names
cyclo_adr
### For 10 MoAs
Either:
tian_10
/usr/bin/python3 /home/jovyan/Tomics-CP-Chem-MoA/04_Tomics_Models/Erik_Models/E_training_test_split.py
well_10
### With Cell Line Selection
_cellsel_
### With Quality of Replicates Level 4 Selection
Insert q followed by threshold value that we have demanded
_q1_

## Examples
### most used
L1000_training_set_tian10.csv
L1000_training_set_cyclo_adr.csv

### more complicated example
L1000_training_set_cyclo_adr_cellsel_q1.csv
