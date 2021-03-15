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
import pandas as pd

import gdal, osr

import geopandas

import rasterio
import rasterio.plot
from rasterio.mask import mask


from scipy import interpolate

from plot_formatter import PlotFormatter

import csv 

import pickle

plt_format = PlotFormatter(plt)
#plt.ioff()
plt.close('all')


wind = [[320,9],[310,10]]


def nechad(c,rho):
    spm = c[0]*rho/(1 - rho/0.170991)+c[1]
    return spm

c = [4.02979661e+02, -1.06997743e-01]


#def nechad(c,x):
#    c = [29.08789804,  5.93511266]
#    return c[1]*np.exp(x*c[0])

#c = [400,  5.18400007e-02]
poly = geopandas.GeoDataFrame.from_file('bay_shapefile/bayarea_onlysfb.shp')

usgs_poly = geopandas.GeoDataFrame.from_file('bay_shapefile/polaris_locations.shp')

outer_directory = '/Volumes/Joe_backups/landsat_order/landsat7/converted/'#'/Users/jadelson/Dropbox/phdResearch/AllOptical/sf_bay_landsat_images/processed_data/'
savedir = '/Volumes/Joe_backups/landsat_order/landsat7_figures/'



polaris_file = './polaris_data.csv'

MATCH_USGS = False
usgsmatch = ''
DEBUG_BREAK = False
SINGLE_TEST = True

if SINGLE_TEST:
    plt.ion()
    DEBUG_BREAK = True
    MATCH_USGS = False
    outer_directory = '/Users/jadelson/Dropbox/phdResearch/AllOptical/sf_bay_landsat_images/processed_data/'
    savedir = '/Volumes/Joe_backups/landsat_order/landsat7_figures/'
    savedir = '/Users/jadelson/Desktop/'

if MATCH_USGS:
    usgsmatch = 'match'
    with open('polaris_length.dk','rb') as length_file:
        usgs_lengths = pickle.load(length_file)

    with open(polaris_file) as pfile:
        csv_reader = csv.reader(pfile, delimiter=',')
        line_count = 0
        pol_data = {}
        
        for row in csv_reader:
            if line_count < 2:
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
        

x = [] + np.cos(np.linspace(0, np.pi, 1000)).tolist()
y = [] + np.sin(np.linspace(0, np.pi, 1000)).tolist()

xy1 = list(zip(x, y))
s1 = max(max(x), max(y))

# ...
x = [] + np.cos(np.linspace(np.pi, 2*np.pi, 1000)).tolist()
y = [] + np.sin(np.linspace(np.pi, 2*np.pi, 1000)).tolist()
xy2 = list(zip(x, y))
s2 = max(max(x), max(y))


geoms = poly['geometry'].unary_union
from shapely.geometry import mapping
geoms = [mapping(geoms[0])]

polaris_geoms = usgs_poly['geometry']
polaris_locs = usgs_poly
polaris_geoms = [mapping(polaris_geoms[0])]

#filenames = ['LE70440342015280EDC00_L2.nc']

subdirnames = os.listdir(outer_directory)
breaktime = False
for sub_directory in reversed(subdirnames):
    breaktime = False
    if breaktime:
        break

    if not SINGLE_TEST:
        directory = outer_directory + sub_directory + '/'
        if not os.path.isdir(directory):
            continue
        
        if MATCH_USGS:
            sub_dir_date = datetime.strptime(sub_directory[12:],'%Y%m%d')
        
            if not sub_dir_date in pol_data.keys():
                continue
    
        if not MATCH_USGS:
            if os.path.isfile(savedir+'map'+usgsmatch +'_'+sub_directory+'.png'):
                continue
        
    
    
    #    if not sub_directory == 'L1TP_044034_20130323':
    #        continue
    
        filenames = os.listdir(directory)
        
        print(sub_dir_date.strftime('%m/%d/%Y'))
        plt.close('all')
    else:
        directory = outer_directory
        filenames = ['L7_ETM_2016_03_31_18_48_14_044034_L2W.nc']#,'L7_ETM_2015_10_07_18_46_07_044034_L2W.nc']#['.nc']#['L7_ETM_2016_03_31_18_48_14_044034_L2W.nc']
                     
    
    for filename in filenames:


        if not filename.endswith('L2W.nc'):
            continue
        
        if True:
#            print(filename)


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
            out_image, out_transform = mask(src, geoms, crop=False,nodata=np.nan)
            out_meta = src.meta.copy()
            
        
                
            out_meta.update({"driver": "GTiff",
                         "height": out_image.shape[1],
                         "width": out_image.shape[2],
                         "transform": out_transform})
        
            with rasterio.open("RGB.byte.masked.tif", "w", **out_meta) as dest:
                dest.write(out_image)
            rs_img = out_image[0,:,:]
        
        
            date = datetime.strptime(a.isodate[:-3],'%Y-%m-%dT%H:%M:%S.%f')
            datetime_obj_utc = timezone('UTC').localize(date)
            datepac = datetime_obj_utc#.astimezone(timezone('US/Pacific'))
            datestr = datepac.strftime("%Y-%m-%d %H:%M:%S %Z")
            print(datestr)

            array = rs_img.data
            x = np.arange(0, array.shape[1])
            y = np.arange(0, array.shape[0])
#            #mask invalid values
#            array = np.ma.masked_invalid(array)
#            xx, yy = np.meshgrid(x, y)
#
#            #get only the valid values
#            x1 = xx[~array.mask]
#            y1 = yy[~array.mask]
#            newarr = array[~array.mask]


            #Convert reflectance to ssc
            ssc_img = rs_img
            # ..set_fill_value(np.nan) 
            ssc_img = nechad(c,ssc_img)
            
            # Build an interpolator
            b = dest.bounds  
            ix = b.left + x*(b.right - b.left)/np.max(x)
            iy = b.bottom + y*(b.top - b.bottom)/np.max(y)                
            f = interpolate.RegularGridInterpolator((ix, iy), np.flip(ssc_img.transpose(),1),'linear')

            #Get USGS Transect
            total_length = usgs_poly.length[0]
            remaining_length = total_length
            dx = 300
            usgs_d = np.arange(0,total_length,dx)
            p0 = polaris_geoms[0]
            xp = []
            yp = []
            for d in usgs_d:
                
                p = usgs_poly.interpolate(d)
                xp.append(p.x[0])
                yp.append(p.y[0])
            
            usgs_pair = np.array([xp,yp]).transpose()
            usgs_ssc = f(usgs_pair)
            usgs_ssc[usgs_ssc < 0] = np.nan
            usgs_ssc[usgs_ssc > 600] = np.nan

            tran_y = np.array([])
            tran_x = np.array([])
            tran_t = np.array([])
            if MATCH_USGS:
                transect_data = pol_data[sub_dir_date]
                tran_x = np.array([usgs_lengths[r]/1000 for r in transect_data.keys()])
                tran_y = np.array([r[-1] for r in transect_data.values()])
                tran_t = np.array([timezone('US/Pacific').localize(r[0])-datepac for r in transect_data.values()])
                tran_t = np.array([(r.days*3600*24+r.seconds)/3600 for r in tran_t])
                        

            min_rs_points = 150
            min_usgs_points = 12
            # Plot the map
            if (not MATCH_USGS) | ((np.sum(~np.isnan(usgs_ssc)) > min_rs_points) & (len(tran_y) > min_usgs_points)):
                sfb_boundary = poly.geometry.unary_union
                df = pd.DataFrame({'Name': ['San Francisco Bay'], 'Coordinates':[sfb_boundary]})
            
                gdf = geopandas.GeoDataFrame(df, geometry='Coordinates')
            
        
                cmap = cm.binary
                cmap.set_bad('white',1.)
                
                if MATCH_USGS:
                    fig = plt.figure(figsize=(5, 7))
                    axs = fig.subplots(2,1, gridspec_kw = {'height_ratios':[5, 2]})
                    ax = axs[0]
                else:
                    fig = plt.figure(figsize=(6,6))
                    ax = fig.subplots(1,1)
                gdf.plot(color='None', edgecolor='gray', ax=ax)
                if MATCH_USGS:
                    polpoints = polaris_geoms[0]['coordinates']
                    pstart = polpoints[0]
                    usgs_poly.plot(ax=ax, edgecolor='black',linewidth=1)
#                    for pp in polpoints:
#                        ax.plot(pp[0], pp[1], 'ko', alpha=.5, 
#                                markersize = 2, fillstyle = 'none', linewidth=1)
                        
                    ax.plot(pstart[0], pstart[1], 'ko', markersize = 5)
                    ax.text(pstart[0]+1500, pstart[1]+600,'Transect Start',va='bottom',ha='left',
                            bbox=dict(facecolor='white', edgecolor='none', 
                                      boxstyle='round,pad=.01', alpha=.8))
                    ax.text(sfb_boundary.bounds[2] - 1000, 
                            sfb_boundary.bounds[3] - 1200,
                            '(a)',va='top', ha='right')
                    
                _vmax = np.nanpercentile(ssc_img,97)
                _vmax = 80
                im = ax.imshow(ssc_img, extent=[b.left,b.right,b.bottom,b.top], vmin=0, vmax=_vmax, cmap=cmap, interpolation='nearest')
            
                ax.set_yticks([4150447,4205921])
                ax.set_xticks([544226,588454])
                labels = [item for item in plt.xticks()[0]]
                
                
                xlabels = [r'$122^\circ 30^\prime$ W',r'$122^\circ$ W']
                ylabels = [r'$37^\circ 30^\prime$ N',r'$38^\circ$ N']  

                ax.set_xticklabels(xlabels)
                ax.set_yticklabels(ylabels) 

                ax.set_xlim(sfb_boundary.bounds[::2])
                ax.set_ylim(sfb_boundary.bounds[1::2])
#                
#                
##                fig = plt.gcf()
                cbar = fig.colorbar(im, ax=ax)
                cbar.set_label(r'SSC (mg L$^{-1}$)', fontsize=plt_format.fontsize)
                cbar.ax.tick_params(labelsize=plt_format.fontsize) 

                ax.text(544226,4151447,datepac.strftime("%B %d, %Y\n%H:%M:%S %Z"),va='top')
               

                if not MATCH_USGS:
                    fig.tight_layout()
                    fig.subplots_adjust(left=0.11)
                    plt.savefig(savedir+'map'+usgsmatch +'_'+sub_directory+'.png',dpi=plt_format.paper_dpi)    


            # Plot the usgs transect
            if (np.sum(~np.isnan(usgs_ssc)) > min_rs_points) & (~MATCH_USGS | (len(tran_y) > min_usgs_points)):                              
                if not MATCH_USGS:
                    fig = plt.figure(figsize=(6,3))
                    ax = fig.subplots(1,1)
                else:
                    ax = axs[1]
                ax.plot(usgs_d/1000, usgs_ssc,'k', linewidth=1)
                if MATCH_USGS:
                    ax.scatter(tran_x, tran_y, s=40, c='none', edgecolor='k',linewidth=2)
#                    ax.scatter(tran_x, tran_y, s=50, c='w', edgecolor='none')

                    ax.scatter(tran_x[tran_t < 0], tran_y[tran_t < 0], s=40, 
                               marker=(xy1, 0), c=np.abs(tran_t[tran_t < 0]), 
                               cmap=cmap, edgecolor='none', vmin=0, vmax=6)
                    ax.scatter(tran_x[tran_t >= 0], tran_y[tran_t >= 0], s=40, marker=(xy1, 0), c='w', alpha=1, edgecolor='none')

                    scat = ax.scatter(tran_x[tran_t > 0], tran_y[tran_t > 0], s=40,
                               marker=(xy2, 0), c=np.abs(tran_t[tran_t > 0]),
                               cmap=cmap, edgecolor='none', vmin=0, vmax=6)
                    ax.scatter(tran_x[tran_t <= 0], tran_y[tran_t <= 0], s=40, marker=(xy2, 0), c='w', alpha=1, edgecolor='none')

                    cbar = fig.colorbar(scat, ax=ax)
                    cbar.set_label(r'Time lag (hours)', fontsize=plt_format.fontsize)
                    cbar.ax.tick_params(labelsize=plt_format.fontsize) 
                    
                    
                ax.set_xlabel('Distance (km)')
                ax.set_ylabel('SSC (mg L$^{-1}$)')
                ax.set_xlim([0,125])
                ax.set_ylim([0,150])
                
                ax.set_yticks([0,50,100,150])
                fig.tight_layout()
                if MATCH_USGS:
                    ax.text(120,140, '(b)', ha='right', va='top')
                    # plt.savefig(savedir+'usgsandmap_'+sub_directory+
                    #             '.png',dpi=plt_format.paper_dpi) 
                    breaktime= True
                else:
                    pass
                    # plt.savefig(savedir+'usgs'+ usgsmatch + '_'+sub_directory+
                    #             '.png',dpi=plt_format.paper_dpi)


#        except Exception as excp:
#            print('usgs_transect went wrong for %s' % filename)
#            print(excp)
        
        
            ######
            #Get cross section interpolation
            ######
            try:
                if not MATCH_USGS:
                    t_xmin = 570960
                    t_xmax = 576660
                    t_ymin = 4154535
                    t_ymax = 4157585
                    t_x = np.linspace(t_xmin,t_xmax,100)
                    t_y = np.linspace(t_ymin,t_ymax,100)
                    dumb_d = np.sqrt((t_y-np.min(t_y))**2+(t_x-np.min(t_x))**2)
                    
                    
                    dumb_pair = np.array([t_x, t_y]).transpose()
                    dumb_ssc = f(dumb_pair)
                    dumb_ssc[dumb_ssc < 0] = np.nan
                    dumb_ssc[dumb_ssc > 600] = np.nan
                    if np.sum(~np.isnan(dumb_ssc)) > 10:               
                        plt.figure()
                        plt.plot(dumb_d ,dumb_ssc,'k')
                        plt.xlabel('Distance (m)')
                        plt.ylabel('SSC (mg L$^{-1}$)')  
                        plt.xlim([0, np.max(dumb_d)])
                        # plt.savefig(savedir+'dumb_'+sub_directory+'.png',dpi=plt_format.paper_dpi)    
            
            except Exception as excp:
                print('dumbarton_transect went wrong for %s' % filename)
                print(excp)
        
            
#        except Exception as excp:
#             print('mapping went wrong for %s' % filename)
#             print(excp)
#    
    
    if DEBUG_BREAK:
        break
