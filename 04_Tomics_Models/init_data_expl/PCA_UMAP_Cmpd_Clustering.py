#!/usr/bin/env python
# coding: utf-8

# Import Statements
import pandas as pd
import matplotlib.pyplot as plt     
import umap
from sklearn.decomposition import PCA
import neptune.new as neptune
import sys
import plotly as plotly
from plotly import express as px


sys.path.append('/home/jovyan/Tomics-CP-Chem-MoA/05_Global_Tomics_CP_CStructure/')
from Erik_alll_helper_functions import checking_veracity_of_data, accessing_all_folds_csv
from Erik_alll_helper_functions import create_splits, pre_processing, set_bool_hqdose, set_bool_npy


def cmpd_clustering(df_train_features, df_train_labels, moa_subset):
    # Subset the data to only include the MoA of interest
    df_singleMoA_labels = df_train_labels[df_train_labels["moa"] == moa_subset]
    df_singleMoA_train_features = df_train_features.iloc[df_singleMoA_labels.index]
    df_singleMoA_train_features.reset_index(drop=True, inplace=True)
    df_singleMoA_labels.reset_index(drop=True, inplace=True)

    print("Investigating using PCA")
    pca_ten = PCA(n_components=10)
    pca_ten.fit_transform(df_singleMoA_train_features)
    pca_comp = plt.figure()
    plt.bar([i for i in range(0,10)], pca_ten.explained_variance_ratio_)
    plt.title("Explained Variance Ratio of PCA Components")

    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(df_singleMoA_train_features)
    principalDf = pd.DataFrame(data=principalComponents,
                               columns=["PC1", "PC2"])

    fig = px.scatter(principalDf, x='PC1', y='PC2', color=df_singleMoA_labels["Compound ID"],
                     title=f'Principal Component Analysis of {moa_subset} Dataset',
                     labels={
                         'PC1': 'Principal Component - 1',
                         'PC2': 'Principal Component - 2'
                     },
                     color_discrete_sequence=px.colors.qualitative.Plotly)
    fig.update_traces(marker=dict(size=12,
                            line=dict(width=2,
                            color='DarkSlateGrey')),
                            selector=dict(mode='markers'))

    # Customize the title's font
    fig.update_layout(title_font=dict(size=24, family='Times New Roman', color='black'))

    # Customize the axis labels' font
    fig.update_xaxes(title_font=dict(size=18, family='Times New Roman', color='black'))
    fig.update_yaxes(title_font=dict(size=18, family='Times New Roman', color='black'))

    # Customize the tick labels' font and color
    fig.update_xaxes(tickfont=dict(size=14, family='Times New Roman', color='black'))
    fig.update_yaxes(tickfont=dict(size=14, family='Times New Roman', color='black'))

    # Set axis line color
    fig.update_xaxes(showline=True, linecolor='black')
    fig.update_yaxes(showline=True, linecolor='black')

    # Add ticks that go through the axis
    fig.update_xaxes(ticks="inside")
    fig.update_yaxes(ticks="inside")

    # Set the legend title and position
    fig.update_layout(legend_title=dict(text='Compound ID', font=dict(size=15, family='Times New Roman', color='black')))
    fig.update_layout(legend=dict(x=1.05, y=0.5, xanchor='left', yanchor='middle', bgcolor='rgba(255, 255, 255, 1)'))

    # Set the background color and remove borders
    fig.update_layout(plot_bgcolor='rgba(255, 255, 255, 1)', paper_bgcolor='rgba(255, 255, 255, 1)')

    fig.show()

    print("Starting UMAP")
    umap_n_components = 20
    pca = PCA(n_components=umap_n_components)
    principalComponents = pca.fit_transform(df_singleMoA_train_features)
    principalDf = pd.DataFrame(data=principalComponents)
    map_neighbors = 15
    umap_min_dist = 0.1
    reducer = umap.UMAP(n_neighbors=map_neighbors, min_dist=umap_min_dist)
    embedding = reducer.fit_transform(principalDf)
    umap_df = pd.DataFrame(embedding, columns=["UMAP1", "UMAP2"])

    fig = px.scatter(umap_df, x='UMAP1', y='UMAP2', color=df_singleMoA_labels["Compound ID"],
                    title=f'UMAP projection of Compound Clustering with {moa_subset}',
                    labels={'UMAP1': 'UMAP - Embedding - 1', 'UMAP2': 'UMAP - Embedding - 2'},
                    color_discrete_sequence=px.colors.qualitative.Plotly)
    fig.update_traces(marker=dict(size=12,
                        line=dict(width=2,
                        color='DarkSlateGrey')),
                        selector=dict(mode='markers'))

    # Customize the title's font
    fig.update_layout(title_font=dict(size=24, family='Times New Roman', color='black'))

    # Customize the axis labels' font
    fig.update_xaxes(title_font=dict(size=18, family='Times New Roman', color='black'))
    fig.update_yaxes(title_font=dict(size=18, family='Times New Roman', color='black'))

    # Customize the tick labels' font and color
    fig.update_xaxes(tickfont=dict(size=14, family='Times New Roman', color='black'))
    fig.update_yaxes(tickfont=dict(size=14, family='Times New Roman', color='black'))

    # Set axis line color
    fig.update_xaxes(showline=True, linecolor='black')
    fig.update_yaxes(showline=True, linecolor='black')

    # Add ticks that go through the axis
    fig.update_xaxes(ticks="inside")
    fig.update_yaxes(ticks="inside")

    # Set the legend title and position
    fig.update_layout(legend_title=dict(text='Compound ID', font=dict(size=15, family='Times New Roman', color='black')))
    fig.update_layout(legend=dict(x=1.05, y=0.5, xanchor='left', yanchor='middle', bgcolor='rgba(255, 255, 255, 1)'))

    # Set the background color and remove borders
    fig.update_layout(plot_bgcolor='rgba(255, 255, 255, 1)', paper_bgcolor='rgba(255, 255, 255, 1)')

    fig.show()

# clue row metadata with rows representing transcription levels of specific genes
clue_gene = pd.read_csv('/home/jovyan/Tomics-CP-Chem-MoA/04_Tomics_Models/init_data_expl/clue_geneinfo_beta.txt', delimiter = "\t")



file_name = "erik10_hq_8_12"
fold_int = 0
print(f'Fold Iteration: {fold_int}')
training_set, validation_set, test_set = accessing_all_folds_csv(file_name, fold_int)
hq, dose = set_bool_hqdose(file_name)
L1000_training, L1000_validation, L1000_test = create_splits(training_set, validation_set, test_set, hq = hq, dose = dose)
checking_veracity_of_data(file_name, L1000_training, L1000_validation, L1000_test)
variance_thresh = 0
normalize_c = 'False'
npy_exists, save_npy = set_bool_npy(variance_thresh, normalize_c, five_fold = 'True')
df_train_features, df_val_features, df_train_labels, df_val_labels, df_test_features, df_test_labels, dict_moa = pre_processing(L1000_training, L1000_validation, L1000_test, 
        clue_gene, 
        npy_exists = npy_exists,
        use_variance_threshold = variance_thresh, 
        normalize = normalize_c, 
        save_npy = save_npy,
        data_subset = file_name)
checking_veracity_of_data(file_name, df_train_labels, df_val_labels, df_test_labels)
cmpd_clustering(df_train_features, df_train_labels, "EGFR inhibitor")