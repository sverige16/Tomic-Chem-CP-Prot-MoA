import sqlalchemy
import pandas as pd
import os
from tqdm import tqdm

def choosing_correct_channel_col(channel):
    '''Takes in channel number and returns the column name of corresponding channel
    Input:
        channel: (int)
    Output:
        a string with column name (ex. "C1")
    '''
    return "C" + str(channel)

def csv_image_path_constructor(compounds_csv,project_name, moa_min = 3, neg_con = 'negcon'):
    '''
    Generates two pandas dataframes with the same format.
    The first has only treated pictures. Each row will have paths to five pictures, representing 5 channel pictures per site and compound name, moa.
    The second has only negative control. Each row will have paths to five pictures, representing 5 channel pictures per site and compound name, moa.
    
    Takes around 2 min per 10,0000 rows. Approximation can be made comparing printed output with tqdm output
    
    Input:
        compounds_csv: give path name to compounds file that you want
        project_name: give name of project 
        moa_min: hyperparameter; minimum number of moas
        neg_con: hyperparameter choosing which negative controls to include (negcon is all, [dmso], sorb, are the other options)
    Output:
        final_doc: pandas dataframe with 9 columns
        neg_doc: pandas dataframe with 9 columns
    
    '''
    '''## Psuedocode:
    1. Using sql, create pandas dataframes with dmso and treated.
    2. Join compounds list to treated based on "class id". 
        2.5 Filter compounds based on Phil's main 1) Moa is not NaN 2) At least three compounds have the same MoA
    3. Populating new pandas dataframe in the same format as Phil's main. 
    4. Filtering: Remove rows based on criteria such as 1. need moa 2. need at least three pictures
    5. Create DMSO statistics
    6. Test to see if we can get identical pandas for SPECS1 vs Phil's main using my code
    '''
    print("performing sql commands")
    # 1. DMSO

    # Connection info for the database
    db_uri = 'postgresql://pharmbio_readonly:readonly@imagedb-pg-postgresql.services.svc.cluster.local/imagedb'
    query = 'SELECT *' + "\n" + 'FROM images_all_view'+ "\n" + 'WHERE project LIKE '+ "'"+ project_name+"'" + "\n" + "AND pert_type LIKE " + "'"+ neg_con+"'"
    
    #GROUP BY plate_acquisition_id, plate_acquisition_name, plate_barcode
    #        ORDER BY plate_barcode, plate_acquisition_id
    # Query database and store result in pandas dataframe
    dmso = pd.read_sql_query(query, db_uri)

    # 1. Treated

    # Connection info for the database
    db_uri = 'postgresql://pharmbio_readonly:readonly@imagedb-pg-postgresql.services.svc.cluster.local/imagedb'
    
    query = 'SELECT *' + "\n" + 'FROM images_all_view'+ "\n" + 'WHERE project LIKE '+ "'"+ project_name+"'" + "\n" + "AND pert_type LIKE 'trt'"


    # Query database and store result in pandas dataframe
    treated = pd.read_sql_query(query, db_uri)

    # 2. Joining Compounds

    # read csv file into pandas
    compounds = pd.read_csv(compounds_csv, sep=",")

    ## 2.5 Filtering MoAs

    # create list of unique moas where moa is known there is at least three of the same moa
    moas = compounds['moa']

    # filtering step
    moas = [x for x in moas if not pd.isna(x) and len(x) > moa_min]

    # keep all of the moas that match filtering step
    compounds = compounds.loc[compounds['moa'].isin(moas)].reset_index(drop=True)


    ## Actual joining
    print("merging")
    # Join rows and add info from database where batchid martches
    df_joined = pd.merge(treated, compounds, how='left', left_on='batch_id', right_on='Batch nr')


    # create new datafrae to populate
    final_doc = pd.DataFrame(columns=['plate', 'well', 'compound', 'C1', 'C2', 'C3', 'C4', 'C5', 'moa'])


    # if you want to populate the whole pandas dataframe
    sample = df_joined

    # Group by multi-index that uniquely identifies the 5 pictures that correspond to the 5 channels of a single site
    sample_gr = sample.groupby(['plate_barcode', 'well', 'site', 'batch_id'])

    # Number of pictures with potentially 5 channels; can compare this to iterations produced by tqdm to see how long time is lef
    print(f' Number of iterations needed: {len(sample_gr.groups)}')


    print("populating data frame")
    # Populating the final_doc dataframe
    for new_index, (old_index, data) in tqdm(enumerate(sample_gr)):
        final_doc.at[new_index, "plate"]= old_index[0]
        final_doc.at[new_index, "well"]= old_index[1]

        for index, row in data.iterrows():
            channel = row["channel"]
            channel_col = choosing_correct_channel_col(channel)
            final_doc.at[new_index, channel_col] = row["path"]
            final_doc.at[new_index, "moa"] = row["moa"]
            final_doc.at[new_index, "compound"] = row["Compound ID"]


    # 4. Perform Filtering

    final_doc = final_doc[final_doc["compound"].notna()]


    # 5. Create DMSO 
    print("starting with dmso")
     # create new datafrae to populate
    dmso_doc = pd.DataFrame(columns=['plate', 'well', 'neg_control_type', 'C1', 'C2', 'C3', 'C4', 'C5'])


    # Group by multi-index that uniquely identifies the 5 pictures that correspond to the 5 channels of a single site
    dmso_gr = dmso.groupby(['plate_barcode', 'well', 'site', 'batch_id'])

    # Number of pictures with potentially 5 channels; can compare this to iterations produced by tqdm to see how long time is lef
    print(f' Number of iterations needed: {len(dmso_gr.groups)}')

    # Populating the final_doc dataframe
    for new_index, (old_index, data) in tqdm(enumerate(dmso_gr)):
        dmso_doc.at[new_index, "plate"]= old_index[0]
        dmso_doc.at[new_index, "well"]= old_index[1]

        for index, row in data.iterrows():
            channel = row["channel"]
            channel_col = choosing_correct_channel_col(channel)
            dmso_doc.at[new_index, channel_col] = row["path"]
            dmso_doc.at[new_index, "neg_control_type"] = row["cbkid"]

    return final_doc, dmso_doc

treat_pd, dmso_pd = csv_image_path_constructor('/home/jovyan/DataAnalysis/SQL-search-in-imagedb-examples/specs935-v1-compounds.csv', 'specs935-v1')

# save to csv
treat_pd.to_csv('/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/specsv2_creation/specs935-v1_paths.csv')
dmso_pd.to_csv('/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/specsv2_creation/specs935-v1_paths_dmso.csv')