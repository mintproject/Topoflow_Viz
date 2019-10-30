#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 11:37:07 2019

@author: deborahkhider
"""

import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import imageio
import os
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from matplotlib.colors import Normalize
import sys
import ast


class PiecewiseNorm(Normalize):
    def __init__(self, levels, clip=False):
        # the input levels
        self._levels = np.sort(levels)
        # corresponding normalized values between 0 and 1
        self._normed = np.linspace(0, 1, len(levels))
        Normalize.__init__(self, None, None, clip)

    def __call__(self, value, clip=None):
        # linearly interpolate to get the normalized value
        return np.ma.masked_array(np.interp(value, self._levels, self._normed))

def histedges_equalN(x, nbin):
    npt = len(x)
    return np.interp(np.linspace(0, npt, nbin + 1),
                     np.arange(npt),
                     np.sort(x))

dataset_name = sys.argv[2]
figsize = ast.literal_eval(sys.argv[3])


#roads = True

#open the file
dataset = xr.open_dataset(dataset_name)

#Get the only variable. According to Scott, one file/variable 
varname = list(dataset.data_vars.keys())[0]

## Get the flow values
val = dataset[varname].values
#val2 = exposure.equalize_hist(val)
nx = dataset.nx.values
ny = dataset.ny.values

## get the edges
val_attrs = dataset[varname].attrs
ymin = val_attrs['y_south_edge']
ymax= val_attrs['y_north_edge']
xmin = val_attrs['x_west_edge']
xmax = val_attrs['x_east_edge']

## get the steps
dx = val_attrs['dx']/3600
dy = val_attrs['dy']/3600

## easting/northing vectors
lon = xmin+dx/2+nx*dx
lat = ymin+dy/2+ny*dy

## convert to lat/lon for sanity
xx,yy=np.meshgrid(lon,lat)
xx2 = np.reshape(xx,xx.size)
yy2 = np.reshape(yy,yy.size)
dv= pd.DataFrame({'lon':xx2,'lat':yy2})

#make the map in cartopy
proj = ccrs.PlateCarree(central_longitude = np.mean(dataset['nx']))
idx = dataset['time'].values.size
count = list(np.arange(0,idx,1))
# get the levels to plot
levels = np.sort(np.unique(histedges_equalN(np.reshape(val,np.size(val)),60)))
# get the box
X_min= np.round(np.min(lon),2)
X_max= np.round(np.max(lon),2)
Y_min= np.round(np.min(lat),2)
Y_max= np.round(np.max(lat),2)

filenames =[]

#Make a directory if it doesn't exit
if os.path.isdir('./figures') is False:
    os.makedirs('./figures')

# loop to create all figures for each time slice
for i in count:
    v = val[i,:,:]
    fig,ax = plt.subplots(figsize=figsize)
    ax = plt.axes(projection=proj)
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.COASTLINE)
    img = plt.contourf(lon, lat, v, levels,
                transform=proj, cmap=cm.gist_gray,norm=PiecewiseNorm(levels)) # need to return to img to make colorbar work
    #m = plt.cm.ScalarMappable(cmap=cm.viridis,norm=PiecewiseNorm(levels)) # Following three lines necessary to lock ylim on colorbar
    #m.set_array(v)
    #m.set_clim(vmin, vmax)
    ticks = levels[0::15]
    ticks = np.sort(np.insert(ticks,-1,levels[-1]))
    cbar = plt.colorbar(img, orientation = 'horizontal',
                        format = '%.2e',ticks=ticks)
    #cbar.clim(vmin,vmax)
    long_name = dataset[varname].attrs['long_name']
    if '_' in long_name:
        t = long_name.split('_')
        strT = t[0].capitalize()
        for item in t[1:]:
            strT = strT + ' ' + item.capitalize()
    else:
        strT = long_name.capitalize()
    cbar.ax.set_xlabel((strT+'('+dataset[varname].attrs['units']+')'))
    
    ax.set_extent([X_min,X_max,Y_min,Y_max])
#    if (np.max(dataset['nx']) - np.min(dataset['nx']))<0.5:
#        ax.set_xticks([np.mean(dataset['nx'])], crs=ccrs.PlateCarree())
#        ax.set_yticks([np.mean(dataset['ny'])], crs=ccrs.PlateCarree())
#    else:
#        ax.set_xticks(np.linspace(np.min(dataset['nx']),np.max(dataset['nx']),4), crs=ccrs.PlateCarree())
#        ax.set_yticks(np.linspace(np.min(dataset['ny']),np.max(dataset['ny']),4), crs=ccrs.PlateCarree())
#    lon_formatter = LongitudeFormatter(zero_direction_label=True)
#    lat_formatter = LatitudeFormatter()
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=2, color='gray', alpha=0.5, linestyle='--')
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xlines = False
    gl.ylines = False
    step_lon = np.round(np.ptp(lon)/5,2)
    step_lat = np.round(np.ptp(lat)/5,2)
    gl.xlocator = mticker.FixedLocator(np.arange(X_min,X_max,step_lon))
    gl.ylocator = mticker.FixedLocator(np.arange(Y_min,Y_max,step_lat))
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 12, 'color': 'gray'}
    gl.ylabel_style = {'size': 12, 'color': 'gray'}
    ax.add_feature(cfeature.RIVERS)
    #save as jpeg
    filename = './figures/'+varname+'_t'+str(i)+'.jpeg'
    filenames.append(filename)
    plt.savefig(filename)
    plt.close(fig)

#Make the GIF
images = []
for filename in filenames:
    images.append(imageio.imread(filename))

imageio.mimsave(long_name+'_movie.gif', images)
