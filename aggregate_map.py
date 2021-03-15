"""
Created on Mon Oct 23 22:53:34 2017

@author: jadelson
"""

import netCDF4 as nc
import matplotlib.pyplot as plt
import matplotlib.cm as cm
#from mpl_toolkits.basemap import Basemap
import numpy as np
import os
from datetime import datetime
from pytz import timezone
import netCDF4 as nc
import pandas as pd

import utm
import gdal, osr

import geopandas
from geopandas.tools import sjoin

import geopandas as gpd

import rasterio
import rasterio.plot
from rasterio.mask import mask

from shapely.geometry import Point, mapping

from scipy import interpolate

from plot_formatter import PlotFormatter

import csv 

import pickle


plt_format = PlotFormatter(plt)
plt.ioff()


wind = [[320,9],[310,10]]


def nechad(c,rho):
    spm = c[0]*rho/(1 - rho/0.170991)+c[1]
    return spm

c = [4.02979661e+02, 0]


#def nechad(c,x):
#    c = [29.08789804,  5.93511266]
#    return c[1]*np.exp(x*c[0])

#c = [400,  5.18400007e-02]
poly_all = geopandas.GeoDataFrame.from_file('bay_shapefile/bayarea_onlysfb.shp')
poly = geopandas.GeoDataFrame.from_file('bay_shapefile/bayarea_sectioned.shp')

usgs_poly = geopandas.GeoDataFrame.from_file('bay_shapefile/polaris_locations.shp')

outer_directory = '/Volumes/Joe_backups/landsat_order/landsat7/converted/'#'/Users/jadelson/Dropbox/phdResearch/AllOptical/sf_bay_landsat_images/processed_data/'
savedir = '/Volumes/Joe_backups/landsat_order/landsat7_figures/'

HIST_BINS = np.linspace(0,800,2000)



polaris_file = './polaris_data.csv'

MATCH_USGS = True
usgsmatch = ''
DEBUG_BREAK = True
SINGLE_TEST = False
PLOT_IMAGE = False

if SINGLE_TEST:
    plt.ion()
    DEBUG_BREAK = True
    MATCH_USGS = False
    outer_directory = '/Users/jadelson/Dropbox/phdResearch/AllOptical/sf_bay_landsat_images/processed_data/'
    savedir = '/Volumes/Joe_backups/landsat_order/landsat7_figures/'

if MATCH_USGS:
    usgsmatch = 'match_complete'
    with open('polaris_length.dk','rb') as length_file:
        usgs_lengths = pickle.load(length_file)

    with open(polaris_file) as pfile:
        csv_reader = csv.reader(pfile, delimiter=',')
        line_count = 0
        pol_data = {}
        
        for i, row in enumerate(csv_reader):

            
            if line_count < 2 or len(row) == 0:
                line_count+=1
                continue
    
            mdy = row[0].split('/')
            row[0] = datetime(month=int(mdy[0]), day=int(mdy[1]), year=int(mdy[2]))
            
            rtemp = row[1]
            row[1] = row[2]
            row[2] = datetime(month=int(mdy[0]), day=int(mdy[1]), 
               year=int(mdy[2]), hour=int(rtemp[:2]), minute=int(rtemp[2:]))
            
            row[3:] = [np.nan if x == '' else float(x) for x in row[3:]]
            if not row[0] in pol_data.keys():
                pol_data[row[0]] = {}
            if float(row[1]) in usgs_lengths.keys():     
                pol_data[row[0]][float(row[1])] = row[2:]
            line_count+=1
        


#geoms = poly['geometry']
GEOMS = {}
all_histograms = {}
agg_hist = {}    
    
for i, sec in poly.iterrows():
#    if sec.geometry.type == 'MultiPolygon':
#        g = sec.geometry.buffer(0)
#        g = g.unary_union
#    else:
    g = sec.geometry
    GEOMS[sec['Region']]=[mapping(g)]
    
    blank_hist, bins = np.histogram([],HIST_BINS)
    all_histograms[sec['Region']] = {}
    agg_hist[sec['Region']] = blank_hist
    
#geoms = [mapping(geoms)]


#filenames = ['LE70440342015280EDC00_L2.nc']

subdirnames = os.listdir(outer_directory)
for sub_directory in reversed(subdirnames):
    

    
    if not SINGLE_TEST:
        directory = outer_directory + sub_directory + '/'
        if not os.path.isdir(directory):
            continue
        
        sub_dir_date = datetime.strptime(sub_directory[12:],'%Y%m%d')
        
        if MATCH_USGS:
            if not sub_dir_date in pol_data.keys():
                continue
    
#        if not MATCH_USGS:
#            if os.path.isfile(savedir+'map'+usgsmatch +'_'+sub_directory+'.png'):
#                continue
        
    
    
    #    if not sub_directory == 'L1TP_044034_20130323':
    #        continue
    
        filenames = os.listdir(directory)
        
        print(sub_dir_date.strftime('%m/%d/%Y'))
        plt.close('all')
    else:
        directory = outer_directory
        filenames = ['L7_ETM_2015_10_07_18_46_07_044034_L2W.nc']
    
    print
    for filename in filenames:

        

        if not filename.endswith('L2W.nc'):
            continue
        
        if True:
            print(filename)


            ncfile = filename
            
            a = nc.Dataset(directory + ncfile)
            nc_array = a['rhow_661'][:]
    #        nc_array[nc_array>100]=np.nan
            #create a geotiff from netcdf
            
            geotiff_path = './gdaltest/test.tif'
                
                
            y_pixels, x_pixels = nc_array.shape
            y_max = a.yrange[1]
            x_min = a.xrange[0]
            x_pixel_size = a.pixel_size[1]
            y_pixel_size = a.pixel_size[0]
            srs = osr.SpatialReference()                 # Establish its coordinate encoding
            srs.ImportFromEPSG(32610) 
            
            driver = gdal.GetDriverByName('GTiff')
            
                
            raster_ds = driver.Create(
                geotiff_path,
                x_pixels,
                y_pixels,
                1,
                gdal.GDT_Float32 , )
             
            raster_ds.SetGeoTransform((
                x_min,    # 0
                x_pixel_size,  # 1
                0,                      # 2
                y_max,    # 3
                0,                      # 4
                -y_pixel_size))  
                
                
            raster_ds.SetProjection( srs.ExportToWkt() ) 
            raster_ds.GetRasterBand(1).WriteArray(nc_array)
            
            raster_ds.FlushCache()
            
            raster_ds.ReadAsArray(0)   
            
            src = rasterio.open(geotiff_path)
            
            final_image = None
            
            for i, geom_key in zip(range(len(GEOMS)), GEOMS.keys()):
                
                geoms = GEOMS[geom_key]
                out_image, out_transform = mask(src, geoms, crop=False,nodata=np.nan)
                out_meta = src.meta.copy()
                
            
                    
                out_meta.update({"driver": "GTiff",
                             "height": out_image.shape[1],
                             "width": out_image.shape[2],
                             "transform": out_transform})
#            
#                with rasterio.open("RGB.byte.masked.tif", "w", **out_meta) as dest:
#                    dest.write(out_image)
                img = out_image[0,:,:]
                
                m,n = img.shape
                aff = out_meta['affine']
                left, top = aff*(0,0)
                right, bottom = aff*(n,m)
            
            
                date = datetime.strptime(a.isodate[:-3],'%Y-%m-%dT%H:%M:%S.%f')
                datetime_obj_utc = timezone('UTC').localize(date)
                datepac = datetime_obj_utc.astimezone(timezone('US/Pacific'))
                
                
                
                
                array = img.data
                if i == 0:
                    x = np.arange(0, array.shape[1])
                    y = np.arange(0, array.shape[0])
                    xx, yy = np.meshgrid(x, y)
                
                #mask invalid values
                array = np.ma.masked_invalid(array)
                
                #get only the valid values
                x1 = xx[~array.mask]
                y1 = yy[~array.mask]
                newarr = array[~array.mask]
            
            #    GD1 = interpolate.griddata((x1, y1), newarr.ravel(), (xx, yy),  method='cubic')
            #    img2 = np.ma.masked_where((GD1 < 0) | (GD1 > 0.13), GD1, copy=True)
            
                img2 = img
                
                img2.set_fill_value(np.nan) 
                img2 = nechad(c,img2)
                
                good_data = img2.data.ravel()
                good_data = good_data[~np.isnan(good_data)]
                good_data = good_data[good_data >= 0]
                good_data = good_data[good_data < 1000]
                if len(good_data) > 50:
                    hist, bins = np.histogram(good_data,HIST_BINS)
                    agg_hist[geom_key] += hist
                    all_histograms[geom_key][datepac] = hist
                if final_image is None:
                    final_image = np.zeros(img2.shape)*np.nan
                final_image[~np.isnan(img2.data)] = img2.data[~np.isnan(img2.data)]
            #Default is to apply mask
            
            if PLOT_IMAGE:
                sfb_boundary = poly_all.geometry.unary_union
                df = pd.DataFrame({'Name': ['San Francisco Bay'], 'Coordinates':[sfb_boundary]})
            
                gdf = geopandas.GeoDataFrame(df, geometry='Coordinates')
            
        
                cmap = cm.binary
                cmap.set_bad('white',1.)
                ax = gdf.plot(color='None', edgecolor='gray', linewidth=1,figsize=(6,6), alpha=.5)
                
                poly.plot(color='None', edgecolor='gray', linewidth=1, ax=ax)
                if MATCH_USGS:
                    usgs_poly.plot(ax=ax, edgecolor='black')
                _vmax = np.nanpercentile(img2.compressed(),97)
                _vmax = 80
                im = ax.imshow(final_image, extent=[left, right, bottom, top],
                               vmin=0, vmax=_vmax, cmap=cmap, interpolation='nearest')
                
            #    ax.barbs(556064.7480499124, 4163811.0485457624, wind[k][1]*np.cos(wind[k][0]/180*np.pi), wind[k][1]*np.sin(wind[k][0]/180*np.pi))
            
                plt.yticks([4150447,4205921])
                plt.xticks([544226,588454])
                labels = [item for item in plt.xticks()[0]]
                
            #    for i in range(len(labels)):
            #        latlon = utm.to_latlon(labels[i],b.bottom,10,'S')
            #        labels[i] = np.round(latlon[1],1)
                xlabels = [r'$122^\circ 30^\prime$ W',r'$122^\circ$ W']
                ax.set_xticklabels(xlabels)
                ax.set_xlim(sfb_boundary.bounds[::2])
                ax.set_ylim(sfb_boundary.bounds[1::2])
                
                
                fig = plt.gcf()
                cbar = fig.colorbar(im)
                cbar.set_label(r'SSC (mg L$^{-1}$)', fontsize=plt_format.fontsize)
                fig.tight_layout()
                fig.subplots_adjust(left=0.11)
                ax.text(544226,4150447,datepac.strftime("%B %d, %Y\n%H:%M:%S %Z"),va='top')
                
           
           
                plt.savefig(savedir+'sectioned_map_'+usgsmatch +'_'+sub_directory+'.png',dpi=plt_format.paper_dpi)    
#        
#
#    
    if DEBUG_BREAK:
        break
    
if not DEBUG_BREAK and not SINGLE_TEST:
    with open('all_histogram.dat','wb') as f:
        pickle.dump(all_histograms, f)
        
    with open('agg_histogram.dat','wb') as f:
        pickle.dump(agg_hist, f)        