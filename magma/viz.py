# viz
from .chemspace import mol_to_bokeh_encodable

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import bokeh

from bokeh.plotting import figure
from bokeh.models import HoverTool, ColumnDataSource, CategoricalColorMapper
import colorcet as cc

rcParams['axes.titlepad'] = 20

# from holoviews.operation.datashader import datashade, rasterize
# from holoviews.operation import gridmatrix
# import hvplot.pandas
# import bokeh.io
# import holoviews as hv
# import colorcet as cc

def set_plotting_style_plt():

    tw = 1.5
    rc = {'lines.linewidth': 2,
        'axes.labelsize': 18,
        'axes.titlesize': 21,
        'xtick.major' : 12,
        'ytick.major' : 12,
        'xtick.major.width': tw,
        'xtick.minor.width': tw,
        'ytick.major.width': tw,
        'ytick.minor.width': tw,
        'xtick.labelsize': 'large',
        'ytick.labelsize': 'large',
        'font.family': 'sans',
        'weight':'bold',
        'grid.linestyle': ':',
        'grid.linewidth': 1.5,
        'grid.color': '#ffffff',
        'mathtext.fontset': 'stixsans',
        'mathtext.sf': 'fantasy',
        'legend.frameon': True,
        'legend.fontsize': 12,
       "xtick.direction": "in","ytick.direction": "in"}



    plt.rc('text.latex', preamble=r'\usepackage{sfmath}')
    plt.rc('mathtext', fontset='stixsans', sf='sans')
    sns.set_style('ticks', rc=rc)

    #sns.set_palette("colorblind", color_codes=True)
    sns.set_context('notebook', rc=rc)


def get_binary_palettes():
    """
    Binary palettes in HEX format.
    Positions: colors = {
     0: blue,
     1: purple,
     2: orange,
     3: blue-green,
     4: grey,
     5: green
    }
    """
    pals = (
        ['#a8ddb5', '#2c7fb8'], # Blues
        ['#bdbdbd', '#fcc5c0'], # Pinks
        ['#969696','#807dba'], # Purple
        ['#fec44f','#d95f0e'], # Oranges
        ['#bdc9e1', '#1c9099'], # Blue-green
        ['#cccccc', '#636363'], # Greys
        ['#78c679', '#238443'] # Greens
    )

    return pals

def radar_chart(categories, values, color = 'lightgreen'):
    """
    Wrapper function to make radar chart from a counts dictionary.

    Params
    ------
    categories (array-like)
        Names of the different categories (labels).

    values (array-like)
        Counts of the given categories.

    Example
    -------
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    iris = sns.load_dataset('iris')

    val_counts=iris.species.value_counts().to_dict()

    # Makes radar chart
    radar_chart(val_counts.keys(), val_counts.values())

    """
    if not isinstance(categories, list):
        categories = list(categories)

    N = len(categories)

    if N > 30:
        print("The categories are too big to visualize.")

    else:
        values = list(values)
        # Repeat first value in last pos to close figure
        values.append(values[0])
        values_sum = np.sum(values[:-1])

        percentages = [(val / values_sum) * 100 for val in values]

        angles = [2 * np.pi * (n / float(N)) for n in range(N)]
        # Repeat first angle too
        angles.append(angles[0])

        sns.set_style("whitegrid")

        # Initialize figure, with polar plot
        plt.figure(1, figsize=(7, 7))
        ax = plt.subplot(111, polar=True)

        # Draw one ax per variable + add labels labels
        plt.xticks(angles[:-1], categories, color="grey", size=12)

        # Set first variable in the vertical axis
        ax.set_theta_offset(np.pi / 2)

        # Set clockwise rotation
        ax.set_theta_direction(-1)

        # Set yticks to gray color
        ytick_1, ytick_2, ytick_3 = (
            np.round(max(percentages) / 3),
            np.round((max(percentages) / 3) * 2),
            np.round(max(percentages) / 3) * 3,
        )

        plt.yticks(
            [ytick_1, ytick_2, ytick_3],
            [ytick_1, ytick_2, ytick_3],
            color="grey",
            size=10,
        )

        y_tickmax = np.round(max(percentages) / 3) * 4

        plt.ylim(0, y_tickmax)

        # Plot data
        ax.plot(angles, percentages, linewidth=1, color=color)

        # Fill area
        ax.fill(angles, percentages, "lightgreen", alpha=0.3)


def lollipop_plot(
    cats, values,
    title=None,
    xlabel=None,
    ylabel=None,
    sort = False,
    color="lightblue",
    figsize = None,
    xlim = None,
    log = False
):
    """
    Makes a lollipop plot from categorical values. Expects cats and values to be lists
    or numpy arrays.

    Params
    ------
    cats (array-like)

    values (array-like)

    title (str, default =None)

    xlabel(str, default =None)

    ylabel(str, default =None)

    sort(str, default = False)

    color(str, default ="lightblue")

    Example
    -------
    import matplotlib.pyplot as plt
    import seaborn as sns
    import magma.viz as viz

    dict_1 = {'LA': 3.6, 'SF':0.874, 'Sydney':5.32, 'CDMX': 8.8}

    viz.lollipop_plot(
        list(dict_1.keys()),
        list(dict_1.values()),
        xlabel = 'citizens (millions)',
        title = 'Population',
        ylabel = 'city name',
        sort = True
    )
    """

    n_datapoints = len(cats)
    range_ = np.arange(n_datapoints)

    if not isinstance(values, np.ndarray):
        values = np.array(list(values))
    if not isinstance(cats, np.ndarray):
        cats = np.array(list(cats))

    max_val = values.max()
    corrector = max_val*0.05

    if sort:
        sorted_ix = values.argsort()
        values = values[sorted_ix]
        cats = cats[sorted_ix]

    if figsize is not None:
        fig = plt.figure(figsize = figsize)
    else:
        fig = plt.figure(figsize=(2, n_datapoints * 0.44))

    plt.hlines(y=range_, xmin=0, xmax= values - corrector , color="lightgrey")

    plt.scatter(values, range_, color=color)

    plt.yticks(range_, cats)

    if log:
        plt.xscale('log')

    if xlabel is not None:
        plt.xlabel(xlabel)

    if ylabel is not None:
        plt.ylabel(ylabel)

    if title is not None:
        plt.title(title)

    if xlim is not None:
        plt.xlim(xlim)


    plt.tight_layout()

    return fig


def make_bokeh_plot_mols(
    df,
    cols_viz,
    color_by,
    x = 'dim_1',
    y = 'dim_2',
    alpha = 0.6,
    fig_kwargs ={
        "plot_width" : 600,
        "plot_height" : 300,
        "tools" : ('pan', 'wheel_zoom', 'reset')
    }
    ):
    """
    Assumes has a mol column.
    """
    assert 'mol' in df.columns, 'Needs an rdkit molecule for visualization.'

    df_viz = df[cols_viz]
    cats = df_viz[color_by].unique()
    n_cats = cats.size

    if 'image' not in df_viz.columns:
        df_viz['image'] = df_viz.mol.apply(mol_to_bokeh_encodable)

    palette = cc.glasbey_dark[:n_cats]

    color_mapping = CategoricalColorMapper(
        factors = cats, palette = palette
    )

    datasource = ColumnDataSource(df_viz)

    fig = figure(fig_kwargs)

    fig.add_tools(HoverTool(tooltips="""
    <div>
        <div>
            <img src='@image' style='float: left; margin: 2px 2px 2px 2px'/>
        </div>
        <div>
            <span style='font-size: 12px; color: #224499'>Molecule:</span>
            <span style='font-size: 14px'>@drug_name</span>
        </div>
    </div>
    """))

    fig.circle(
        x, y,
        color=dict(field=color_by, transform = color_mapping),
        size=10,
        line_alpha = 0.8,
        fill_alpha = alpha,
        source=datasource
    )

    return fig

# def plot_sample_datashade(df, sample_name, vars_, sample_col = 'sample_id', **kwargs):
#     """
#     Returns an hvplot object with the sample colored by an indicator
#     variable colored using datashading. This function is devised for
#     visualizing datasets with millions of datapoints.

#     Params
#     ------
#     df (pd.DataFrame)
#         Annotated pandas dataframe.

#     sample_name (str)
#         Name of the sample to be colored.

#     vars (list)
#         Name of the xy variables for the scatter plot.

#     sample_col(str, default = 'sample_id')
#         Name of the column for which the sample_name will be selected from.

#     kwargs
#         All kwargs go directly to format the hvplot object.


#     Returns
#     -------

#     shader_plot ()
#         Scatter plot colored by sample name using datashader.

#     """

#     df_ = df.copy()
#     # Assert sample in sample_col
#     assert sample_name in df[sample_col].values

#     # Assert there are more than two vars to plot with

#     assert len(vars_) >= 2

#     # Make binary indicator var
#     indicator_variable = [1 if smpl== sample_name else 0 for smpl in df[sample_col]]

#     # Add variable to dataframe
#     df_[sample_name] = indicator_variable

#     # Initialize plot for two variables
#     if len(vars_) == 2:
#         var_1, var_2  = vars_

#         shader_plot = df_.hvplot.scatter(
#             x = var_1,
#             y = var_2,
#             c = sample_name,
#             #width = 600,
#             datashade = True,
#             **kwargs
#         )

#     # Initialize plot for multiple variables
#     else :
#         shader_plot = df_.hvplot.scatter(
#             x = vars_[0],
#             y = vars_[1:],
#             c = sample_name,
#             datashade = True,
#             **kwargs
#         )

#     return shader_plot


# def make_gridplot_hv(
#     df,
#     col_list,
#     color_by=None,
#     n_samples=None,
#     rasterize_ = True)->hv.core.spaces.GridMatrix:

#     """
#     Returns a scatterplot gridmatrix for all pairwise combinations in col_list.
#     Example: http://holoviews.org/gallery/demos/bokeh/iris_density_grid.html

#     Note: with len(col_list)>5 the plot starts to take a lot of RAM, we recommend
#     to use rasterize_= True.

#     Params
#     ------

#     df (pd.DataFrame)
#         DataFrame to extract data from. It should contain both the `col_list`
#         and `color_by` as columns.

#     col_list (list)
#         List of columns to plot, each column should have numerical values.

#     color_by (str, default = None)
#         Name of column to color the plot, it should be an object or categorical dtype.
#         Analog to the `hue` parameter in Seaborn.

#     n_samples (int, default = None)
#         Max number of samples to use for random sampling. This parameter helps
#         when the dataset is very large (i.e. > 100K datapoints).

#     rasterize_(bool)
#         Whether to rasterize the plot to be able to plot millions of points.

#         See the datashader documentation for more info :
#         https://datashader.org/getting_started/Introduction.html

#     Returns
#     -------
#     (holoviews.core.spaces.GridMatrix)

#     """

#     if n_samples is not None:
#         df_ = df.sample(n_samples, replace = False)[col_list + [color_by]]
#     else:
#         df_ = df[col_list + [color_by]]

#     if color_by is not None:
#         ds = hv.Dataset(df_).groupby(color_by).overlay().opts(width = 150, height = 150)
#     else:
#         ds = hv.Dataset(df_)


#     if rasterize_:
#         return rasterize(gridmatrix(ds))
#     else :
#         from holoviews.operation.stats import univariate_kde

#         return gridmatrix(
#             ds,
#             #chart_type = hv.Bivariate,
#             diagonal_type = hv.Distribution
#             )
