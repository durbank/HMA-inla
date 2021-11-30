# Script to reproject racmo data to normal grid.
# This is necessary because RACMO uses a rotated pole projection system that is not adequately handled by r functions.

# %%

import xarray as xr
from pathlib import Path

DATA_DIR = Path(
    '/media/durbank/WARP/', 
    'Research/Antarctica/Data/RACMO/', 
    '2.3p2_yearly_ANT27_1979_2016')

files = list(
    DATA_DIR.joinpath('raw').glob('*.nc'))
vars_raw = ['v10m', 'u10m', 'smb']
das_raw = []
for i, file in enumerate(files):
    var = vars_raw[i]
    ds = xr.open_dataset(file)
    proj4_params = ds.rotated_pole.attrs[
        'proj4_params']
    da = ds[var].squeeze()
    rlon = da.rlon.values
    rlat = da.rlat.values

    das_raw.append(da)

    # Info for performing these gdal operations came from https://gitlab.tudelft.nl/slhermitte/manuals/blob/master/RACMO_reproject.md
    print("********************************")
    print(f"gdal commands for file: {file}")

    print("Translation command...")
    print(f'gdal_translate NETCDF:"{file}":{var} -a_ullr {rlon.min()} {rlat.max()} {rlon.max()} {rlat.min()} tmp.tif')

    print("Warping command...")
    print(f'gdalwarp -s_srs "{proj4_params}" -t_srs "EPSG:3031" -te -3051000 3051000 3051000 -3051000  -tr 27000 -27000 -r bilinear tmp.tif {str(DATA_DIR.joinpath("interim", var+".tif"))}')
    print("********************************")


# %%[markdown]
# STOP TO RUN GDAL COMMANDS PRIOR TO CONTINUING!!!
# 
# %%

das_dict = dict(zip(vars_raw, das_raw))



files = list(
    DATA_DIR.joinpath('interim').glob('*.tif'))
vars = ['smb', 'u10m', 'v10m']
das = []
for i, file in enumerate(files):
    var = vars[i]
    da = xr.open_rasterio(file)
    da = da.rename(
        {'band':'time', 'y':'Northing', 'x':'Easting'})
    da.attrs = das_dict[var].attrs
    # da = da.assign_attrs(das_dict[var].attrs)
    da = da.assign_coords(
        {"time":("time",das_dict[var].time.values)})

    das.append(da)

# %%
das[0].to_netcdf(DATA_DIR.joinpath('SMB_3031-projected.nc'))
ds_final = xr.Dataset(dict(zip(vars, das)))
ds_final.to_netcdf(
    DATA_DIR.joinpath('EPSG3031-projected.nc'))
# %%
