#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 10:25:13 2019

@author: deborahkhider


Movie for Topoflow (simple)

"""
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyproj import Proj
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.cm as cm
import imageio
import os
import sys
import ast


zone = float(sys.argv[1])
dataset_name = sys.argv[2]
figsize = ast.literal_eval(sys.argv[3])
#roads = True

#open the file
dataset = xr.open_dataset(dataset_name)

#Get the only variable. According to Scott, one file/variable 
varname = list(dataset.data_vars.keys())[0]

## Get the flow values
val = dataset[varname].values
nx = dataset.nx.values
ny = dataset.ny.values

## get the edges
val_attrs = dataset[varname].attrs
ymin = val_attrs['y_south_edge']
ymax= val_attrs['y_north_edge']
xmin = val_attrs['x_west_edge']
xmax = val_attrs['x_east_edge']

## get the steps
dx = val_attrs['dx']
dy = val_attrs['dy']

## easting/northing vectors
easting = xmin+dx/2+nx*dx
northing = ymin+dy/2+ny*dy

dataset['nx']=easting
dataset['ny']=northing
## convert to lat/lon for sanity
xx,yy=np.meshgrid(easting,northing)
xx2 = np.reshape(xx,xx.size)
yy2 = np.reshape(yy,yy.size)
dv= pd.DataFrame({'east':xx2,'north':yy2})
# use pyproj to do this
myProj = Proj("+proj=utm +zone="+str(zone)+", +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
lon_all, lat_all = myProj(dv['east'].values, dv['north'].values, inverse=True)
#now get the "unique" lon
lon = lon_all[0:nx.size]
assert lon.size == nx.size, "Longitude vector size is incorrect"
dataset['nx']=lon
#get the "unique"lat
idx_vec = np.arange(0,lon_all.size,nx.size)
lat = lat_all[idx_vec]
assert lat.size == ny.size, "Latitude vector size is incorrect"
dataset['ny']=lat


#make the map in cartopy
proj = ccrs.PlateCarree(central_longitude = np.mean(dataset['nx']))
idx = dataset['time'].values.size
count = list(np.arange(0,idx,1))
vmin = np.min(dataset[varname].values)
vmax = np.max(dataset[varname].values)
filenames =[]

#Make a directory if it doesn't exit
if os.path.isdir('./figures') is False:
    os.makedirs('./figures')

# loop to create all figures for each time slice
for i in count:
    v = dataset[varname].values[i,:,:]
    fig,ax = plt.subplots(figsize=figsize)
    ax = plt.axes(projection=proj)
    img = plt.contourf(nx, ny, v, 60,
                transform=proj, cmap=cm.viridis,
                vmin = vmin,
                vmax = vmax) # need to return to img to make colorbar work
    m = plt.cm.ScalarMappable(cmap=cm.viridis) # Following three lines necessary to lock ylim on colorbar
    m.set_array(v)
    m.set_clim(vmin, vmax) 
    cbar = plt.colorbar(m, boundaries=np.linspace(vmin, vmax,6))
    #ax.add_feature(cfeature.RIVERS) 
    ax.add_feature(cfeature.BORDERS)
    #ax.add_feature(cfeature.NaturalEarthFeature('cultural', 'roads', '10m'))
    long_name = dataset[varname].attrs['long_name']
    if '_' in long_name:
        t = long_name.split('_')
        strT = t[0].capitalize()
        for item in t[1:]:
            strT = strT + ' ' + item.capitalize()
    else:
        strT = long_name.capitalize()
    cbar.ax.set_ylabel((strT+'('+dataset[varname].attrs['units']+')'))
    if (np.max(dataset['nx']) - np.min(dataset['nx']))<0.5:
        ax.set_xticks([np.mean(dataset['nx'])], crs=ccrs.PlateCarree())
        ax.set_yticks([np.mean(dataset['ny'])], crs=ccrs.PlateCarree())
    else:
        ax.set_xticks(np.linspace(np.min(dataset['nx']),np.max(dataset['nx']),4), crs=ccrs.PlateCarree())
        ax.set_yticks(np.linspace(np.min(dataset['ny']),np.max(dataset['ny']),4), crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    
    #save as jpeg
    filename = './figures/'+varname+'_t'+str(i)+'.jpeg'
    filenames.append(filename)
    plt.savefig(filename)
    plt.close(fig)

#Make the GIF
images = []
for filename in filenames:
    images.append(imageio.imread(filename))

imageio.mimsave(varname+'_movie.gif', images)
