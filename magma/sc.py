import scanpy
import anndata as ad
import gseapy

import numpy as np
import pandas as pd


def get_scanpy_deg_report_df(
    adata,
    clus_annot = 'rank_genes_groups',
    groups = ('-1','1'),
    cols_annot = ["names", "logfoldchanges", "pvals_adj"]
):
    """
    Returns a report dataframe of differentially expressed genes.
    It expects an adata with a report dictionary from the output of
    scanpy.tl.rank_genes_groups().

    Params
    ------
    adata (ad.AnnData)
        AnnData with rank_genes_groups dictionary in `.uns` object.
        Ideally, this adata would only contain "prototype" cells,
        that is, the cells on the extremes of a given component.

    clus_annot(str, default = 'rank_genes_groups')
        Label in the .uns object to get the results from.

    groups (tuple, default = (-1,1))
        Tuple of groups for which to extract the DEG results.

    cols_annot(array-like, default= ["names", "logfoldchanges", "pvals_adj"])
        Columns to use from the .uns object for the report.

    Returns
    -------
    df_report (pd.DataFrame)
        Report dataframe of DEG test.
    """
    # Extract dictionary from adata
    deg_result_dict = adata.uns[clus_annot]

    print(deg_result_dict["names"][:5])

    # Initialize dataframe
    df_report = pd.DataFrame()

    # Record information for each group / cluster in the report df
    for g in groups:
        df = pd.DataFrame(
                np.vstack(([deg_result_dict[col][g] for col in cols_annot])).T,
            columns=["gene_name", "log_fc", "pval_adj"],
        )

        df["group"] = g

        df_report = pd.concat([df_report, df])

    return df_report

def run_deg_groups(
    adata,
    groups:dict,
    group_column = 'drug_name',
    top_genes = 20,
    method = 'wilcoxon',
    run_enrichment = False,
    pval_thresh = 1e-3
    ):

    """
    Runs a one-vs-all DEG test. It uses all the samples within each group
    specified in a dictionary `groups`.

    params
    ------
    adata (ad.AnnData)
        Anndata containing all samples in the values of the groups dict.

    groups (dict)
        Dictionary whose keys tell the group each sample belongs to.

    group_column (str)
        Name of the column to use as reference to label the samples from groups.
        This could be `sample_id` if you're running a deg test on

    Returns
    -------
    df_report (pd.DataFrame)
        DE genes in each group.

    df_enrichment (pd.DataFrame, optional)
        Reactome pathways enriched for the DE genes.
    """

    samples_in_scope = np.concatenate(list(groups.values()))

    adata_filt = adata[adata.obs[group_column].isin(samples_in_scope)].copy()

    assert adata_filt.n_obs > 1, "adata is empty after filtering, check samples."

    gps = list(groups.keys())

    # Invert dictionary for labeling samples in the form {sample : group}
    mapper = {}
    for name_of_group, samples_in_group in groups.items():
        for sample in samples_in_group:
            mapper[sample] = name_of_group

    adata_filt.obs['group'] = adata_filt.obs[group_column].map(mapper)

    adata_filt.obs['group'] = adata_filt.obs['group'].astype("category")

    adata_filt.var.set_index('gene_name', drop =False, inplace = True)

    #print(adata_filt.var.head())

    # Run DEG test
    scanpy.tl.rank_genes_groups(
        adata_filt,
        groupby = 'group',
        use_raw = False,
        method = method,
        n_genes = top_genes
    )

    # Extract report dataframe
    df_report = get_scanpy_deg_report_df(adata_filt, groups = gps)

    df_report.pval_adj = df_report.pval_adj.astype(float)
    df_report.log_fc = df_report.log_fc.astype(float)

    df_report = df_report[df_report.pval_adj < pval_thresh]

    df_report['mode'] = df_report.log_fc.apply(
        lambda x: 'upregulated' if x > 1 else 'downregulated'
    )

    # Run enrichment test
    if run_enrichment:
        cols_enrichment = ['Term', 'Adjusted P-value', 'Genes']

        df_enrichment_result = pd.DataFrame()

        for group in groups:
            genes_group = df_report[df_report["group"] == group]['gene_name'].to_list()

            # Run enrichment test
            enrichr_obj = gseapy.enrichr(
                list(genes_group), 'Reactome_2016', outdir = './tmp/', no_plot = True
            )

            df_enrichment_gp = enrichr_obj.results[cols_enrichment]
            df_enrichment_gp["group"] = group

            df_enrichment_result = pd.concat([df_enrichment_result, df_enrichment_gp])

        df_enrichment_result = df_enrichment_result[df_enrichment_result["Adjusted P-value"]< pval_thresh]
        return df_report, df_enrichment_result

    return df_report

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

    deg_adata.var.set_index('gene_name', drop =False, inplace = True)

    control_name = adata_control[0].obs[gene_col_name].values[0]
    drug_name = adata_test[0].obs[sample_col_name].values[0]

    gps = (control_name, drug_name)

    # Run DEG
    scanpy.tl.rank_gene_groups(deg_adata, sample_col_name, method = 'wilcoxon')

    # Get report df
    df_report = get_scanpy_deg_report_df(deg_adata, groups = gps)
    df_report.pval_adj = df_report.pval_adj.astype(float)
    df_report.log_fc = df_report.log_fc.astype(float)

    df_report = df_report[df_report.pval_adj < pval_thresh]

    if run_enrichment:
        cols_enrichment = ['Term', 'Adjusted P-value', 'Genes']

        df_enrichment_result = pd.DataFrame()

        for group in gps:
            genes_group = df_report[df_report["group"] == group]['gene_name'].to_list()

            # Run enrichment test
            enrichr_obj = gseapy.enrichr(
                list(genes_group), 'Reactome_2016', outdir = './tmp/', no_plot = True
            )

            df_enrichment_gp = enrichr_obj.results[cols_enrichment]
            df_enrichment_gp['group'] = group
            df_enrichment_result = pd.concat([df_enrichment_result, df_enrichment_gp])

        return df_report, df_enrichment_result

    return df_report






def safe_gene_selection(
    adata,
    input_list,
    gene_colname = 'gene_name',
    #keep_order=False
    )-> ad.AnnData:
    """
    Returns a new adata with a subset of query genes.

    TODO: Use indexing on the .var object, by setting gene_name as index.

    Note: It will only return the genes that are in the dataset.
    If any of the query genes are not in the dataset, the gene names
    will be dismissed. If you're not too sure of the exact gene names
    check the `df.gene_colname.str.contains()` or the
    `df.gene_colname.str.startswith()` function.

    Params
    ------
    adata(ad.AnnData)
        Dataset to select from.

    input_list(array-like)
        Query list with gene names.

    gene_colname (str, default = 'gene_name')
        Name of the column in the .var object from which to
        make the query against.

    Returns
    -------
    new_adata (ad.AnnData)
        Subset of original anndata containg only query genes.

    Example
    -------

    # Initalize dummy list and shuffle it
    gene_names = list('ABCDEFGHIJ')
    rng = np.random.default_rng(seed = 9836)
    gene_names = rng.permutation(gene_names)
    print(gene_names)

    # Create adata with random 5 cell 10 gene count matrix
    a = ad.AnnData(
        X = np.random.random((5, 10)),
        var= pd.DataFrame(gene_names, columns = ['gene_name'])
    )

    my_list = ['A', 'C', 'B','D']

    ada_new = sc.safe_gene_selection(a, my_list)

    print(ada.var.gene_name.values)
    >>> array(['A', 'B', 'C', 'D'], dtype=object)

    """

    # Gets the indices of the rows contained in my_list using bool array
    isin_indexer = adata.var[gene_colname].isin(input_list).values.nonzero()[0]

    # Returns the indices that sort the values
    # selected with the indexer array
    new_ixs = np.argsort(adata.var[gene_colname].values[isin_indexer])

    isin_list_sorted_ixs = isin_indexer[new_ixs]

    adata_new = adata[:, isin_list_sorted_ixs].copy()

    return adata_new
