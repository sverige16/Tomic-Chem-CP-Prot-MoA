import pandas as pd
import numpy as np
from cmapPy.pandasGEXpress.parse import parse
import cmapPy.pandasGEXpress.subset_gctoo as sg
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate


def checking_tomics_profiles(cell_name_string, cell_name, lower_doses, higher_doses, clue_sig):
    '''Creates a table that assesses the number of transcriptomic profiles given a certain dosage range and a selection of 
    cell line from the clue.io database
    
    Input:
    cell_name_string: list of strings representing cell line name
    cell_name:        list of strings of the cell lines to include
    lower_doses:      list of integers representing lower range of dosages
    higher_doses:     list of integers representing higher range of dosages
    
    Output: Tabulate table
    '''
    table = []
    for i, s in zip(cell_name, cell_name_string):
        for a,f in zip(lower_doses, higher_doses):
            row = []
            row.append(s)
            check = clue_sig[clue_sig.cell_iname.isin(i)]
            check_with_correct_dosage = check[check.pert_dose.between(a,f)].reset_index(drop=True)
            number = check_with_correct_dosage.shape[0]
            row.append([a,f])
            row.append(number)
            table.append(row)
    print(tabulate(table, headers=["Cell line Name","Dosage", "# Transcriptomic Profiles"], tablefmt="heavy_grid"))

def number_of_transcriptomics_profiles_for_moa(clue_sig, moa_names):
    ''' Gives the number of transcriptomic profiles that exist for each moa 
    class that we send.
    
    input
    clue_sig: (pandas) meta data with transcriptomic profiles. 
            Must have moa data joined.
    moa_names: List of strings with MoAs of interest
    
    output:
    - prints table with MoA name and the number of transcriptomic profiles.
    - returns the total number of profiles for list of MoAs sent in.''' 
    
    headers = ["MoA Name", "# Tomics Profiles"]
    dictionary = {}
    total = 0
    for i in moa_names:
        transcript_num = clue_sig[clue_sig.moa == i].shape[0]
        dictionary[i] = transcript_num
        total += transcript_num
    print(tabulate(dictionary.items(), headers))
    return total

def compound_moa_classes(clue_sig, compounds1_2, moas, cell_name_string, cell_name, lower_doses, higher_doses, head_num = 5):
    '''Creates a table that assesses the number of transcriptomic profiles given a certain dosage range and a selection of 
    cell line from the clue.io database
    
    Input:
    clue_sig:         metadata from all transcriptomic level 5 signatures
    compounds1_2:     combined list of compounds found in SPECS V1 and V2.
    moas:             list of strings representing a subset of MoA classes to investigatedd
    cell_name_string: list of strings representing cell line name
    cell_name:        list of strings of the cell lines to include
    lower_doses:      list of integers representing lower range of dosages
    higher_doses:     list of integers representing higher range of dosages
    head_num:         int that determines number of rows displayed by datafram
    
    Output:
    Several data frames of a length determined by head_num

    # Note: The commented out code allows does not take into account enantiomers, but does fix it so that the
    expected number of compounds is correct compared with the five_way split document. 
    '''
    for CL, header, low_t, high_t in zip(cell_name, cell_name_string, lower_doses, higher_doses):
        dict_with_values = {}
        all_moas = []
        all_cmpds = []
        all_tprofiles = []
        all_enantiomers = []
        all_hq_profiles = []
        all_hq_cmpds = []
        #clue_sig = clue_sig[clue_sig["Compound ID"].isin(compounds1_2["Compound_ID"])]
        #check = clue_sig[clue_sig.cell_iname.isin(CL)] # only compounds found in specific cell line
        #check_with_correct_dosage = check[check.pert_dose.between(low_t, high_t)].reset_index(drop=True)
        #clue_sig = clue_sig[clue_sig.pert_id.isin(compounds1_2.CUSTOMER_ID)] #  only compounds found in SPECS v1 and v2
        for moa in moas: # for every individual MoA class 
            clue_sig = clue_sig[clue_sig.pert_id.isin(compounds1_2.CUSTOMER_ID)] #  only compounds found in SPECS v1 and v2
            #clue_sig = clue_sig[clue_sig["Compound ID"].isin(compounds1_2["Compound_ID"])]   
            check = clue_sig[clue_sig.cell_iname.isin(CL)] # only compounds found in specific cell line
            check_with_correct_dosage = check[check.pert_dose.between(low_t, high_t)].reset_index(drop=True) # only compounds found in specific dosage range
            check_with_correct_dosage = check_with_correct_dosage[check_with_correct_dosage.moa == moa] # only compounds found in specific MoA class
            tprofile_number = check_with_correct_dosage.shape[0]
            unique_all = check_with_correct_dosage.drop_duplicates(subset=["pert_id"])
            #unique_all = check_with_correct_dosage.drop_duplicates(subset=["Compound ID"])
            cmpds = unique_all[unique_all.moa == moa] # all compounds targeting specific MoA class
            hq_profiles = check_with_correct_dosage[check_with_correct_dosage["is_hiq"] == 1]
            hq_profile_cmpds = hq_profiles.drop_duplicates(subset=["pert_id"])
            all_moas.append(moa)
            all_cmpds.append(cmpds.shape[0])
            all_tprofiles.append(tprofile_number)
            all_enantiomers.append(unique_all[unique_all.duplicated("Compound ID")].shape[0])
            all_hq_profiles.append(hq_profiles.shape[0])
            all_hq_cmpds.append(hq_profile_cmpds.shape[0])
        dict_with_values["moa"] = all_moas
        dict_with_values["Compound #"] = all_cmpds 
        dict_with_values["Profile #"] = all_tprofiles # number of compounds targeting specific MoA class
        dict_with_values["Enantiomers"] = all_enantiomers
        dict_with_values["HQ Profiles"] = all_hq_profiles
        dict_with_values["HQ Compounds"] = all_hq_cmpds
             
        # printing MoA classes with x number of compounds targeting specific MoA class 
        print(f" \033[1m{'Cell Line(s):' : >20}\033[0m  {header}")
        print(f" \033[1m{'Dosage Range:' : >20}\033[0m  [{low_t},{high_t}]")
        print(f" \033[1m{'All Compounds Found in SPECS v1 and V2' : >20}\033[0m")
        print(tabulate (dict_with_values, headers = "keys", tablefmt="heavy_grid"))
        table_info = [
            ["Total # of Transcriptomic Profiles:", {sum(dict_with_values["Profile #"])}], 
            ["Total # of HQ Transcriptomic Profiles:", {sum(dict_with_values["HQ Profiles"])}],
            ["Total # of Compounds:", {sum(dict_with_values["Compound #"])}],
            ["Total # of HQ Compounds:", {sum(dict_with_values["HQ Compounds"])}],
            ["TProfiles of High Quality (%):", {round(sum(dict_with_values["HQ Profiles"])/sum(dict_with_values["Profile #"])*100,2)}],
            ["Enantiomers:", {sum(dict_with_values["Enantiomers"])}]]
        print(tabulate(table_info, tablefmt="fancy_grid"))
        #print(f' Total # of Transcriptomic Profiles: {sum(dict_with_values["Profile #"])}')
        #print(f' Total # of HQ Transcriptomic Profiles: {sum(dict_with_values["HQ Profiles"])}')
        #print(f' Total # of HQ Compounds: {sum(dict_with_values["HQ Compounds"])}')
        #print(f' TProfiles of High Quality: {round(sum(dict_with_values["HQ Profiles"])/sum(dict_with_values["Profile #"])*100,2)}%')
        #print(f' Enantiomers: {sum(dict_with_values["Enantiomers"])}')
    print(f' MoAs Investigated: {moas}')
     

# metadata for clue.io data in the SPECS1&2
clue_sig_in_SPECS = pd.read_csv('/home/jovyan/Tomics-CP-Chem-MoA/04_Tomics_Models/init_data_expl/clue_sig_in_SPECS1&2.csv', delimiter = ",")
# identifying rows with only landmark genes
clue_gene = pd.read_csv('/home/jovyan/Tomics-CP-Chem-MoA/04_Tomics_Models/init_data_expl/clue_geneinfo_beta.txt', delimiter = "\t")
# downloading information on all of the compounds
compounds_v1v2 = pd.read_csv("/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/compounds_v1v2.csv")


#single_name = ['U2OS']
#subset_names = ['MCF7', 'PC3', 'A549']
#lower_doses = [8, 8]
#higher_doses = [12, 12]
#subset2_names = ['PC3', 'A549']
all_names = list(clue_sig_in_SPECS.cell_iname)
moa_subset = input("Which MoAs to Investigate:")
if moa_subset == "erik10":
    moa_subset = ["cyclooxygenase inhibitor", "dopamine receptor antagonist","adrenergic receptor antagonist", "phosphodiesterase inhibitor",  "HDAC inhibitor", 
             "histamine receptor antagonist","EGFR inhibitor", "adrenergic receptor agonist", "PARP inhibitor",  "topoisomerase inhibitor"]
elif moa_subset == "cyc_adr":
    moa_subset = ["cyclooxygenase inhibitor", "adrenergic receptor antagonist"]
elif moa_subset == "cyc_dop":
    moa_subset = ["cyclooxygenase inhibitor", "dopamine receptor antagonist"]
else:
    moa_subset = ['Aurora kinase inhibitor', 'tubulin polymerization inhibitor', 'JAK inhibitor', 'protein synthesis inhibitor', 'HDAC inhibitor',
            'topoisomerase inhibitor', 'PARP inhibitor', 'ATPase inhibitor', 'retinoid receptor agonist', 'HSP inhibitor']
cell_name = [all_names, all_names]
cell_name_string = ["all", "all"] 
lower_doses = [0, 8]
higher_doses = [100, 12]
compound_moa_classes(clue_sig_in_SPECS, compounds_v1v2, moa_subset, cell_name_string, cell_name, lower_doses, higher_doses, head_num = 10)