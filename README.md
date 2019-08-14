[![PyPI](https://img.shields.io/badge/python-3.6-yellow.svg)]()
[![license](https://img.shields.io/github/license/khider/Topoflow_Viz.svg)]()

# Topoflow Visualization

**Python code to generate visualizations for Topoflow outputs**

This code is written for the MINT project. In short, the aim is to create a gif based on netcdf files generated by Topoflow.

**Table of contents**

* [What is it?](#what)
* [Version Information](#version)
* [Quickstart Guide](#quickstart)
* [Requirements](#req)
* [Files in this repository](#files)
* [Contact](#contact)
* [License](#license)

## <a name = "what">What is it?</a>

This Python routine creates a GIF for the the variable contained in the [Topoflow](https://github.com/peckhams/topoflow) netcdf output

## <a name = "version">Version Information</a>
- v0.0.1: Support built for netcdf files only. 

## <a name = "quickstart">Quickstart Guide</a>

To implement from command line

`python topo_viz.py utm_zone filename figsize`

where:
* utm_zone (int): The UTM zone
* filename (str): The name of the netcdf file  
* figsise (list): The size for the figure

Returns a .gif file for the visualization.

*Example:*

`python topo_viz.py 15 June_20_67_2D-Q_5.nc [6,8]`

## <a name="req">Requirements</a>

### Software requirements

#### Python

Current version tested under Python 3.6 with the following dependencies.

- xarray v0.11.0
- cartopy v0.16.0
- numpy v1.16.2
- pandas v0.23.4
- matplotlib v3.0.2
- pyproj v1.9.5.1
- imageio v2.4.1

### Data Requirements

This routine assumes that the data is stored as per Topoflow netCDF format and file organization. 

##<a name="files"> Files in the repository </a>

* June_20_67_2D-Q_5.nc: A sample netcdf file output from Topoflow. See the example [here](https://github.com/peckhams/topoflow/tree/master/topoflow/examples/Treynor_Iowa_30m)
* Q_movie.gif: A sample gif visualization for Treynor, Iowa
* topo_viz.py: The Python routine for the visualization

## <a name = "contact"> Contact </a>

Please report issues to <khider@usc.edu>

## <a name ="license"> License </a>

The project is licensed under the Apache v2.0 License. Please refer to the file call license.
