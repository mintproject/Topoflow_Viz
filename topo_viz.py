#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 15:40:41 2019
​
@author: deborahkhider
​
Topolfow visualization from Scott's Notebook 
"""
​
import numpy as np
import matplotlib.pyplot as plt
import imageio
import  os
import glob
import sys
from topoflow.utils import ncgs_files
from topoflow.utils import ncts_files
​
def histogram_equalize( grid, PLOT_NCS=False):
    (hist, bin_edges) = np.histogram( grid, bins=256)
    # hmin = hist.min()
    # hmax = hist.max()
​
    cs  = hist.cumsum()
    ncs = (cs - cs.min()) / (cs.max() - cs.min())
    ncs.astype('uint8');
    if (PLOT_NCS):
        plt.plot( ncs )
​
    flat = grid.flatten()
    flat2 = np.uint8( 255 * (flat - flat.min()) / (flat.max() - flat.min()) )
    grid2 = ncs[ flat2 ].reshape( grid.shape )
    return grid2
​
def power_stretch1( grid, p ):
    return grid**p
​
def power_stretch2( grid, a=1000, b=0.5):
    # Note: Try a=1000 and b=0.5
    gmin = grid.min()
    gmax = grid.max()
    norm = (grid - gmin) / (gmax - gmin)
    return (1 - (1 + a * norm)**(-b))
​
def power_stretch3( grid, a=1, b=2):
    # Note:  Try a=1, b=2 (shape of a quarter circle)
    gmin = grid.min()
    gmax = grid.max()
    norm = (grid - gmin) / (gmax - gmin)
    return (1 - (1 - norm**a)**b)**(1/b)
​
def log_stretch( grid, a=1 ):
    return np.log( (a * grid) + 1 )
​
​
def makeDirectory(case_prefix):
    home_dir   = os.path.expanduser("~")
    test_dir   = home_dir + '/TF_Output'
    output_dir = test_dir + '/' + case_prefix
    png_dir    = output_dir + '/' + 'png_files'
​
    if not(os.path.exists( test_dir )):   os.mkdir( test_dir )
    if not(os.path.exists( output_dir )): os.mkdir( output_dir)
    if not(os.path.exists( png_dir )):    os.mkdir( png_dir)
    
    os.chdir( output_dir )
    
    return png_dir, output_dir, test_dir
​
def makeGridMovie(nc_file, png_dir, case_prefix):
    
    ncgs = ncgs_files.ncgs_file()
    ncgs.open_file( nc_file )
    var_name_list = ncgs.get_var_names()
    var_name  = var_name_list[3]
    long_name = ncgs.get_var_long_name( var_name )
    var_units = ncgs.get_var_units( var_name )
    
    #create grid stack movie
    im_title = long_name.replace('_', ' ').title()
    im_file_prefix = 'TF_Movie_Frame_'
    time_pad_map = {1:'0000', 2:'000', 3:'00', 4:'0', 5:''}
    cmap = 'rainbow'
    
    time_index = 0
    
    while (True):
        # print('time index =', time_index )
        try:
            grid = ncgs.get_grid( var_name, time_index )
        except:
            break
        time_index += 1
        ## grid2 = log_stretch( grid )
        ## grid2 = power_stretch1( grid )
        grid2 = power_stretch3( grid )
        ## grid2 = histogram_equalize( grid )
        gmin = grid2.min()
        gmax = grid2.max()
    
        fig, ax = plt.subplots( figsize=(6,6), dpi=192) 
        ax.set_title( im_title )
        ax.set_xlabel('Longitude [deg]')
        ax.set_ylabel('Latitude [deg]')
        im = ax.imshow(grid2, interpolation='nearest', cmap=cmap,
                       vmin=gmin, vmax=gmax)
        cbar = plt.colorbar(im, orientation='horizontal',pad=0.2)
        cbar.ax.set_title(long_name + '('+var_units+')')
        # Build a filename for this image/frame
        tstr = str(time_index)
        pad = time_pad_map[ len(tstr) ]
        time_str = (pad + tstr)
        im_file = im_file_prefix + time_str + '.png' 
        im_file = (png_dir + '/' + im_file)
    
        plt.savefig( im_file )
        plt.close()
    
    ncgs.close_file()
    
    # Make movie
    fps = 10  # frames per second
    mp4_file = case_prefix+'_Movie.mp4'
    im_file_list = sorted( glob.glob( png_dir + '/*.png' ) )
    
    writer = imageio.get_writer( mp4_file, fps=fps )
    
    for im_file in im_file_list:
        writer.append_data(imageio.imread( im_file ))
    writer.close()
​
def tsPlot(nc_file, output_dir, case_prefix):
    ncts = ncts_files.ncts_file()
    ncts.open_file( nc_file )
    var_name_list = ncts.get_var_names()
    var_name = var_name_list[3]
    (series, times) = ncts.get_series( var_name )
    long_name = series.long_name
    v_units   = series.units
    t_units   = times.units
    values    = np.array( series )
    times     = np.array( times )
    
    if (t_units == 'minutes'):
        times = times / (60.0 * 24)
        t_units = 'days'
        
    ymin = 0.0
    ymax = values.max()
​
    plt.figure(1, figsize=(11,6))
    marker = ','  # pixel
    y_name = long_name.replace('_', ' ').title()
​
    plt.plot( times, values, marker=marker)
    plt.xlabel( 'Time' + ' [' + t_units + ']' )
    plt.ylabel( y_name + ' [' + v_units + ']' )
    plt.ylim( np.array(ymin, ymax) )
    
    im_file = output_dir+'/'+case_prefix+ '.png'
    plt.savefig( im_file )
​
​
    ncts.close_file()
    
if __name__ == "__main__":
    nc_file = sys.argv[1]
    file_name = nc_file.split('/')[-1]
    case_prefix = file_name.split('_')[0]
    png_dir, output_dir, test_dir = makeDirectory(case_prefix)
    if nc_file.split('_')[1][0:2] == '2D':
        makeGridMovie(nc_file, png_dir,case_prefix)
    else:
        tsPlot(nc_file, output_dir,case_prefix)
    
