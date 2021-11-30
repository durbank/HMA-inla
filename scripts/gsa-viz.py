# Script to generate figures for HMA investigations to be presented at GSA

# %%
# Load requisite modules
from pathlib import Path
import numpy as np
# import pandas as pd
import geopandas as gpd
# import pyproj
# from pyproj import crs
# from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import cartopy.feature as cf
import cartopy.crs as ccrs
# import holoviews as hv
import geoviews as gv
# import geoviews.feature as gf
from geoviews import opts, tile_sources as gvts
gv.extension('matplotlib')

# Define ROOT and DATA directories
ROOT_DIR = Path(__file__).parents[1]
DATA_DIR = ROOT_DIR.joinpath('data')

#%% Load data

# # Import raw data
# raw_df = pd.read_excel(
#     DATA_DIR.joinpath(
#         'HMA_glacier_mass-balance.xlsx'), 
#     sheet_name="hma_mb_20190214_1015_nmad")

# # Define custom crs (from Shean et al, 2020)
# mb_crs = pyproj.CRS.from_proj4(
#     '+proj=aea +lat_1=25 +lat_2=47 +lat_0=36 +lon_0=85 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs')

# mb_gdf = gpd.GeoDataFrame(
#     raw_df.drop(
#         ['RGIId','x','y','t1','t2'], axis=1), 
#     geometry=gpd.points_from_xy(
#         raw_df['x'], raw_df['y']), 
#     crs=mb_crs)
# mb_gdf.to_crs(epsg=4326, inplace=True)

# Load glacier/climate data
glacier_gdf = gpd.read_file(DATA_DIR.joinpath(
    'clim_data/glacier_clim.shp'))

# Add categories for HI
HI = np.zeros(glacier_gdf['hi'].shape, dtype='object')
cut_vals = [-1000, -1.5, -1.2, 1.2, 1.5, 1000]
cats = [
    'Very top-heavy', 'Top-heavy', 'Equidimensional', 
    'Bottom-heavy', 'Very bottom-heavy']
for i,var in enumerate(cats):
    idx = (glacier_gdf['hi']>cut_vals[i]) & (glacier_gdf['hi']<=cut_vals[i+1])
    HI[idx] = cats[i]

glacier_gdf['HI'] = HI

# %%

from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
def add_scale(plot, element):
    """Holoviews hook to add scalebar to generated maps.

    Args:
        plot ([type]): The Holoviews/Geoviews plot to add scalebar to.
        element ([type]): Not actually sure about this...

    Returns:
        [type]: The Holoviews/Geoviews plot with scalebar addition.
    """
    b = plot.state
    scalebar = AnchoredSizeBar(
        b.axes[0].transData, 5.67, '500 km', 'lower left', 
        pad=2, color='black', frameon=False, 
        size_vertical=1)
    b.axes[0].add_artist(scalebar)

from matplotlib.lines import Line2D
def add_legend(plot, element):
    """[summary]

    Args:
        plot ([type]): [description]
        element ([type]): [description]
    """
    b = plot.state
    colors = [
        '#7b3294', '#c2a5cf', 'black', 
        '#fdb863', '#e66101']
    cats = [
        'Very top-heavy', 'Top-heavy', 
        'Equidimensional', 
        'Bottom-heavy', 'Very bottom-heavy']
    lgnd = []
    for i in range(len(colors)):
        lgnd.append(
            Line2D([0], [0], marker='o', 
            color='white', label=cats[i],markerfacecolor=colors[i], markersize=10))
    b.axes[0].legend(handles=lgnd, loc='upper right')

def scale_legend(plot, element):
    """[summary]

    Args:
        plot ([type]): [description]
        element ([type]): [description]
    """
    b = plot.state
    b.axes[0].legend(markerscale=4)

def plot_wrapper(
    glacier_data, plt_var, clabel=None, cmap='viridis', 
    clip=True, plt_type='cont', logZ=False):
    """Wrapper function for generating formatted glacier attribute plots.

    Args:
        glacier_data ([type]): [description]
        plt_var (str): Name of variable to include in plot
        cmap (str): Name of color map to use in plot
        clabel (str): Colorbar label name to use.
    """
    # Generate graticule to use in plot
    graticules = cf.NaturalEarthFeature(
        category='physical',
        name='graticules_5',
        scale='110m')
    grids = gv.Feature(graticules, group='Lines').opts(
        opts.Feature('Lines', facecolor='none', edgecolor='gray'))

    # Generate basemap plot
    basemap = gvts.EsriUSATopo().opts(
        alpha=0.4, projection=ccrs.PlateCarree())

    # Glacier data plot
    glacier_plt = gv.Points(
        glacier_data, vdims=[plt_var])
    if clip:
        # Define colormap limits
        c_min = glacier_data[plt_var].quantile(0.025)
        c_max = glacier_data[plt_var].quantile(0.975)
        glacier_plt.vdims[0].range = (c_min, c_max)

    if plt_type == 'categorical':
        
        colors = [
            '#7b3294', '#c2a5cf', 'black', 
            '#fdb863', '#e66101']
        glacier_plt = gv.Points(
            data=glacier_data).opts(
                color=plt_var, cmap=colors, s=7)

        # tmp_plts = []
        # cats = [
        #     'Very top-heavy', 'Top-heavy', 
        #     'Equidimensional', 
        #     'Bottom-heavy', 'Very bottom-heavy']
        # colors = [
        #     '#7b3294', '#c2a5cf', 'black', 
        #     '#fdb863', '#e66101']
        # for i,cat in enumerate(cats):
        #     tmp_plts.append(gv.Points(
        #         glacier_data[glacier_data['HI']==cat], 
        #         label=cat).opts(color=colors[i], s=7, 
        #         show_legend=True, alpha=0.5))
        # glacier_plt = tmp_plts[0]
        # for i in np.arange(start=1, stop=len(tmp_plts)):
        #     glacier_plt = glacier_plt * tmp_plts[i]
        
        # glacier_plt = glacier_plt.opts(
        #     hooks=[scale_legend])

    elif plt_type == 'diverging':
        
        glacier_plt.opts(
            color=plt_var, cmap=cmap, colorbar=True, 
            logz=logZ, symmetric=True, clabel=clabel, s=7)

    else:

        glacier_plt.opts(
            color=plt_var, cmap=cmap, colorbar=True, logz=logZ, clabel=clabel, s=7)

    # Define spatial boundaries
    lon_lim = (67,104)
    lat_lim = (27,46)

    if plt_type == 'categorical':
        plt_final = (basemap * glacier_plt * grids).options(
            fig_inches=24, xlim=lon_lim, ylim=lat_lim, 
            hooks=[add_legend, add_scale])
    else:
        plt_final = (basemap * glacier_plt * grids).   options(
            fig_inches=24, xlim=lon_lim, ylim=lat_lim,
            hooks=[add_scale])

    return plt_final

# %%

plt.rcParams.update({'font.size': 14})

glacier_plts = []

vars = ['z_med', 'z_slope', 'tau']
cmaps = ['bgy', 'bmy', 'inferno']
clabels = [
    'Median elevation (m)', 'Slope', 
    'Response time (years)']

for i,var in enumerate(vars):
    glacier_plts.append(plot_wrapper(
        glacier_gdf, plt_var=var, 
        cmap=cmaps[i], clabel=clabels[i]))

glacier_plts.append(plot_wrapper(
    glacier_gdf, plt_var='z_aspct', 
    cmap='colorwheel', clabel='Aspect', clip=False))
glacier_plts.append(plot_wrapper(
    glacier_gdf, plt_var='area_m2', 
    cmap='inferno', clabel='Area (sq. m)', 
    logZ=True, clip=False))
glacier_plts.append(plot_wrapper(
    glacier_gdf, plt_var='mb_mwea', 
    cmap='bjy_r', clabel='Mass balance (mwe/a)', 
    plt_type='diverging'))
glacier_plts.append(plot_wrapper(
    glacier_gdf, plt_var='HI', 
    plt_type='categorical', clip=False))

# %% Climate plots

clim_plts = []

vars = [
    't2_w_08', 't2_w_18', 'ppt_a_08', 
    'ppt_a_18', 'ppt_s_08', 'ppt_s_18']
cmaps = [
    'magma', 'magma', 'viridis', 
    'viridis', 'plasma', 'plasma']
clabels = [
    'Warm season 2m air temperature: 2008', 
    'Warm season 2m air temperature: 2018', 
    'Annual precipitation (daily mean): 2008', 
    'Annual precipitation (daily mean): 2018',
    'Seaonal precipitation (jja / ann): 2008', 
    'Seaonal precipitation (jja / ann): 2018']

for i,var in enumerate(vars):
    clim_plts.append(plot_wrapper(
        glacier_gdf, plt_var=var, 
        cmap=cmaps[i], clabel=clabels[i]))

vars = ['t2_w_d', 't2_w_b', 'ppt_a_d', 'ppt_s_d']
cmaps = ['bjy', 'bjy', 'bjy', 'bjy']
clabels = [
    'Warm season 2m air temp change (2008-2018)', 
    'Warm season 2m air temp trend 1990-2020', 
    'Annual precip (daily mean) change (2008-2018)', 
    'Seaonal precip (jja / ann) change (2008-2018)']

for i,var in enumerate(vars):
    clim_plts.append(plot_wrapper(
        glacier_gdf, plt_var=var, 
        cmap=cmaps[i], clabel=clabels[i], 
        plt_type='diverging'))

# %% Save plots as pdfs

all_vars = [
    'z_med', 'z_slope', 'tau', 'z_aspct', 'area_m2', 
    'mb_mwea', 'HI', 't2_w_08', 't2_w_18', 'ppt_a_08',
    'ppt_a_18', 'ppt_s_08', 'ppt_s_18', 't2_w_d', 
    't2_w_b', 'ppt_a_d', 'ppt_s_d']
all_plts = glacier_plts + clim_plts

for i,plt in enumerate(all_plts):
    gv.save(
        plt, 
        filename=ROOT_DIR.joinpath(
            'outputs', all_vars[i]), 
        fmt='pdf')

# %%
