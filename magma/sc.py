import scanpy
import anndata as ad
import gseapy

import numpy as np
import pandas as pd

def get_deg_report_groups(adata, groups= None):
    raise NotImplementedError

def get_deg_report_vs_control(
    adata_control,
    adata_test,
    sample_col_name = 'drug_name',
    run_enrichment = False,
    pval_thresh = None
):
    """
    Assumes that the adatas contain 'gene_name' in .var
    """
    # Get data from drug and control
    deg_adata = ad.concat([adata_control, adata_test])

    deg_adata.var = adata_control.var

    deg_adata.var.set_index('gene_name', drop =False,inplace = False)

    control_name = adata_control[0].obs[gene_col_name].values[0]
    drug_name = adata_test[0].obs[sample_col_name].values[0]

    gps = (control_name, drug_name)

    # Run DEG
    scanpy.tl.rank_gene_groups(deg_adata, sample_col_name, method = 'wilcoxon')

    # Get report df
    df_report = mu.get_scanpy_deg_report_df(deg_adata, groups = gps)


    if run_enrichment:
        de_genes = df_report[df_report.pval_adj < pval_thresh]['gene_name'].values

        # Run enrichment test
        df_enrichment_result = gseapy.enrichr(
            de_genes, 'Reactome_2016', outdir = './tmp/', no_plot = True
        )
        return df_report, df_enrichment_result

    return df_report
