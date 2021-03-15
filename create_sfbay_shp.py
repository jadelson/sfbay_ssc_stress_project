#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 12:52:42 2018

@author: jadelson
"""
import geopandas
import matplotlib.pyplot as plt

from plot_formatter import PlotFormatter

import shapely
from shapely.geometry import mapping

import numpy as np
import utm

import pandas as pd
import pickle
import csv

import gdal, osr

from geopandas.tools import sjoin

import rasterio
import rasterio.plot
from rasterio.mask import mask

import netCDF4 as nc

import cv2

plt_format = PlotFormatter(plt)

plt.close('all')


if True:
    #poly = geopandas.GeoDataFrame.from_file('bay_shapefile/bayarea_allwater.shp')
    poly = geopandas.GeoDataFrame.from_file('bay_shapefile/bayarea_onlysfb.shp')
    
    
    #a = poly[~poly['LANDNAME'].isna()]
    poly['LANDNAME'].fillna('No name', inplace=True)
    
    all_sfbay = poly.geometry.unary_union
    
    df = pd.DataFrame({'Region': ['San Francisco Bay'], 'Coordinates':[all_sfbay]})
    gdf = geopandas.GeoDataFrame(df, geometry='Coordinates')
    ax = gdf.plot(color='white', edgecolor='black',figsize=(6,6),linewidth=.5,alpha=.3)
    
    lines = []
    for pol in all_sfbay:
    #    print(pol)
        boundary = pol.boundary
        if boundary.type == 'MultiLineString':
            for line in boundary:
                lines.append(line)
        else:
            lines.append(boundary)
    
    boundary_line = shapely.geometry.MultiLineString(lines)  
    
    removal_region = boundary_line.buffer(300)
          
    df = pd.DataFrame({'Region': ['San Francisco Bay'], 'Coordinates':[removal_region]})
    gdf4 = geopandas.GeoDataFrame(df, geometry='Coordinates')
    #gdf4.plot(color='green', edgecolor='green',figsize=(6,6),linewidth=.5,alpha=.7,ax=ax)
    
    a = poly.copy()
    names = a['LANDNAME']
    includelist = ['Bay','Delta','Strait','Cove','Basin','Harbor','Corte Madera']
    excludelist = ['Bolinas','Civic Center', 'Tomales','Sand','Corte Madera Creek',
                   'Drakes','Barries','Creamery','Bodega','Marina Lagoon',
                   'Mare', 'Mendota']
    hardcode_ids = [179,69,45,78,48] 
    first = True
    indx = None
    for i in includelist:
        if first:
            indx = a['LANDNAME'].str.contains(i)
            first = False
        else:
            indx = indx | a['LANDNAME'].str.contains(i)
    
           
    for i in hardcode_ids:
        indx = indx | (a['LANDPOLY'] == i)
        
    for i in excludelist:
        indx = indx & ~a['LANDNAME'].str.contains(i)
     
    a = a[indx]
    
    
    sfb_boundary = a.geometry.unary_union
    sfb_boundary = sfb_boundary.difference(removal_region)
    
    df = pd.DataFrame({'Region': ['San Francisco Bay'], 'Coordinates':[sfb_boundary]})
    gdf2 = geopandas.GeoDataFrame(df, geometry='Coordinates')
    
    sfb_boundary = sfb_boundary.difference(removal_region)
    
    
    left_lon = 5.40e5
    bottom_lat = 4.14e6
    
    right_lon = 6.03e5
    top_lat = 4.23e6
    
    south_bay_top_lat = 4.176e6
    
    central_bay_top_lat = 4.203e6
    
    csp_1_lat = 4.204e6
    csp_1_lon = 5.482e5
    csp_2_lat = 4.202e6
    csp_2_lon = 5.497e5
    
    san_pablo_bay_right_lon = 5.64e5
    
    carquinez_right = 5.7638e5
    
    suisunedge1_lon = 5.953e5
    suisunedge1_lat = 4.2141e6
    suisunedge2_lon = 5.946e5
    suisunedge2_lat = 4.2127e6
    suisunedge3_lon = 5.927e5
    suisunedge3_lat = 4.2115e6
    
    point_bl = shapely.geometry.Point((left_lon, bottom_lat))
    point_br = shapely.geometry.Point((right_lon, bottom_lat))
    point_tl = shapely.geometry.Point((left_lon, top_lat))
    point_tr = shapely.geometry.Point((right_lon, top_lat))
    
    point_sbtl = shapely.geometry.Point((left_lon, south_bay_top_lat))
    point_sbtr = shapely.geometry.Point((right_lon, south_bay_top_lat))
    
    point_cbtl = shapely.geometry.Point((left_lon, central_bay_top_lat))
    point_cbtr = shapely.geometry.Point((san_pablo_bay_right_lon, central_bay_top_lat))
    
    point_csp1 = shapely.geometry.Point((csp_1_lon, csp_1_lat))
    point_csp2 = shapely.geometry.Point((csp_2_lon, csp_2_lat))
    
    point_sptr = shapely.geometry.Point((san_pablo_bay_right_lon, top_lat))
    
    point_suitl = shapely.geometry.Point((carquinez_right, top_lat))
    point_suibl = shapely.geometry.Point((carquinez_right, south_bay_top_lat))
    point_suibr = shapely.geometry.Point((right_lon, south_bay_top_lat))
    
    point_suie1 = shapely.geometry.Point((suisunedge1_lon, suisunedge1_lat))
    point_suie2 = shapely.geometry.Point((suisunedge2_lon, suisunedge2_lat))
    point_suie3 = shapely.geometry.Point((suisunedge3_lon, suisunedge3_lat))
    
    
    southbay_pointlist =  [point_bl, point_br, point_sbtr, point_sbtl, point_bl]
    centralbay_pointlist = [point_sbtl, point_sbtr, point_cbtr, point_csp2, point_csp1, point_cbtl, point_sbtl]
    sanpablo_pointlist = [point_cbtl, point_csp1, point_csp2, point_cbtr, point_sptr, point_tl, point_cbtl]
    suisun_pointlist = [point_suibl, point_suibr, point_suie3, point_suie2, point_suie1, point_tr, point_suitl, point_suibl]
    
    southbay_box = shapely.geometry.Polygon([[p.x, p.y] for p in southbay_pointlist])
    southbay_geom = sfb_boundary.intersection(southbay_box)
    
    centralbay_box = shapely.geometry.Polygon([[p.x, p.y] for p in centralbay_pointlist])
    centralbay_geom = sfb_boundary.intersection(centralbay_box)
    
    sanpablobay_box = shapely.geometry.Polygon([[p.x, p.y] for p in sanpablo_pointlist])
    sanpablobay_geom = sfb_boundary.intersection(sanpablobay_box)
    
    suisunbay_box = shapely.geometry.Polygon([[p.x, p.y] for p in suisun_pointlist])
    suisunbay_geom = sfb_boundary.intersection(suisunbay_box)
    
    sectioned_df = pd.DataFrame({'Region': ['South Bay', 'Central Bay', 'San Pablo Bay', 'Suisun Bay'],
                                 'Coordinates':[southbay_geom, centralbay_geom, sanpablobay_geom, suisunbay_geom]})
    sb_df = pd.DataFrame({'Region': ['South Bay'],
                                 'Coordinates':[southbay_geom]})
    
    sectioned_gdf = geopandas.GeoDataFrame(sectioned_df, geometry='Coordinates')
    
    for i in range(4):
        
        geom_temp = sectioned_gdf.iloc[i,1]
        print(geom_temp.type)
    
        if geom_temp.type == 'MultiPolygon':
            for g in geom_temp:
                print(g.area/10**6)
                if (g.area/10**6) < 1:
                    geom_temp = geom_temp.difference(g)
        sectioned_gdf.iloc[i,0] = geom_temp
                        
    
    sectioned_gdf.plot(color='none', edgecolor='black', figsize=(6,6), ax=ax)
    sectioned_gdf[sectioned_gdf['Region'] == 'South Bay'].plot(color='sienna', edgecolor='white', ax=ax, alpha=0.5,hatch='///')
    sectioned_gdf[sectioned_gdf['Region'] == 'Central Bay'].plot(color='cornflowerblue', edgecolor='white', ax=ax, alpha=0.5,hatch='\\\\\\')
    sectioned_gdf[sectioned_gdf['Region'] == 'San Pablo Bay'].plot(color='darkseagreen', edgecolor='white', ax=ax, alpha=0.5,hatch='///')
    sectioned_gdf[sectioned_gdf['Region'] == 'Suisun Bay'].plot(color='orange', edgecolor='white', ax=ax, alpha=0.5,hatch='\\\\\\')
    sectioned_gdf.plot(color='none', edgecolor='black', figsize=(6,6), ax=ax,linewidth=.5,alpha=.9)
    
    #sectioned_gdf.plot(color='gray', edgecolor='gray', figsize=(6,6), ax=ax, hatch='\\')
    
    
    
    
    plt.tight_layout()
    #
    #a.plot(color='blue', edgecolor='black',ax=ax)
    fig = plt.gcf()
    fig.text(.43,.32,'South Bay', bbox=dict(edgecolor='none', facecolor='white', boxstyle="round"), fontsize=plt_format.fontsize)
    fig.text(.3,.59,'Central Bay', bbox=dict(edgecolor='none', facecolor='white', boxstyle="round"), fontsize=plt_format.fontsize)
    fig.text(.19,.82,'San Pablo Bay', bbox=dict(edgecolor='none', facecolor='white', boxstyle="round"), fontsize=plt_format.fontsize)
    fig.text(.6,.78,'Suisun and Grizzly Bays', bbox=dict(edgecolor='none', facecolor='white', boxstyle="round"), fontsize=plt_format.fontsize)
    
    plt.yticks([4150447,4205921])
    plt.xticks([544226,588454])
    labels = [item for item in plt.xticks()[0]]
    
    
    xlabels = [r'$122^\circ 30^\prime$ W',r'$122^\circ$ W']
    ax.set_xticklabels(xlabels)
    ylabels = [r'$37^\circ 30^\prime$ N',r'$38^\circ$ N']  
    ax.set_yticklabels(ylabels)    
    
    plt.savefig('figures/mar2019/sfb_split.png',dpi=plt_format.paper_dpi)

    #%%

# USGS STUFF    
    usgs_points = []
    station_numbers = []
    with open('polaris_locations.csv', 'r') as pol_file:
        reader = csv.reader(pol_file)
     
        rownum = 0
        for row in reader:
        # Save header row.
            if rownum ==0:
                header = row
            else:
                utm_coord = utm.from_latlon(float(row[0]),float(row[1]))
                usgs_points.append([utm_coord[0],utm_coord[1]])
                station_numbers.append(row[2])
                
            rownum += 1
    usgs_points=list(reversed(usgs_points[3:-1]))
    usgs_points.insert(3, [5.7875e5, 4.1503e6])
    station_numbers.insert(3,'-1',)
    station_numbers = list(reversed(station_numbers))
    station_numbers = station_numbers[1:-3]
    
    
    usgs_shp = shapely.geometry.LineString(usgs_points)
    
    
    
    df2 = pd.DataFrame({'Region': ['USGS_POLARIS'], 'Coordinates':[usgs_shp]})
    gdf2 = geopandas.GeoDataFrame(df2, geometry='Coordinates')       
    
    gdf2.to_file("bay_shapefile/polaris_locations.shp")
    df3 = pd.DataFrame({'Region': ['USGS_POLARIS','San Francisco Bay'], 'Coordinates':[usgs_shp,sfb_boundary]})
    gdf3 = geopandas.GeoDataFrame(df3, geometry='Coordinates')      
    #ax = gdf3.plot(color='white', edgecolor='gray',figsize=(6,6)) 
    #
    #gdf2.plot(color='white', edgecolor='black',ax=ax) 
    #plt.plot(552300,4190500,'o')
    #print(utm.to_latlon(552300,4190500,10,'N'))
    
    total_length = 0
    lengths = {}
    
    prev_point = usgs_points[0]
    i_prev = station_numbers[0]
    lengths[float(i_prev)] = total_length
    
    usgs_poly_list = []
    for point in usgs_points:
            p = shapely.geometry.Point(point)
            poly = p.buffer(300)
            usgs_poly_list.append(poly)

    usgs_multigon = shapely.geometry.MultiPolygon(usgs_poly_list)
    usgs_multigon = usgs_multigon.intersection(sfb_boundary)
    df4 = pd.DataFrame({'Region': ['USGS_POLARIS','USGS_POLY', 'San Francisco Bay'], 'Coordinates':[usgs_shp,usgs_multigon,sfb_boundary]})
    gdf4 = geopandas.GeoDataFrame(df4, geometry='Coordinates')
    gdf4.plot(color='none',edgecolor='black')
    for point, i in zip(usgs_points[1:], station_numbers[1:]):
        total_length = total_length + np.sqrt(np.square(point[0]- prev_point[0]) + 
               np.square(point[1]- prev_point[1]))
    
        lengths[float(i)] = total_length
        i_prev = i
        prev_point = point
    with open('polaris_length.dk','wb') as f:
        pickle.dump(lengths, f)             

    sectioned_gdf.to_file("bay_shapefile/bayarea_sectioned.shp")




###DEM STUFF

a = nc.Dataset('/Users/jadelson/Downloads/san_francisco_13_mhw_2010.nc')

#%%
lats = a['lat'][:]
lons = a['lon'][:]
dem = np.flipud(a['Band1'][:])

m,n = dem.shape
lat_min = 1e10
lat_max = -1e10
lon_min = 1e10
lon_max = -1e10

b = all_sfbay.bounds
for i in [0,2]:
    for j in [1,3]:
        lat, lon = utm.to_latlon(b[i],b[j],10,'N')
        if lat > lat_max:
            lat_max = lat
        if lat < lat_min:
            lat_min = lat
            
        if lon > lon_max:
            lon_max = lon
        if lon < lon_min:
            lon_min = lon
            
print(lat_min, lat_max)
print(lon_min, lon_max)

lon_indx_min = np.argmin(np.abs(lon_min-lons))
lon_indx_max = np.argmin(np.abs(lon_max-lons))

lat_indx_min = m-np.argmin(np.abs(lat_min-lats))
lat_indx_max = m-np.argmin(np.abs(lat_max-lats))

dem = dem[lat_indx_max:lat_indx_min, lon_indx_min:lon_indx_max]

lats = lats[np.argmin(np.abs(lat_min-lats)):np.argmin(np.abs(lat_max-lats))]
lons = lons[lon_indx_min:lon_indx_max]
#plt.figure()
import matplotlib.cm as cm

cutoff = -10

dem2 = dem.copy()
indx = dem2 > cutoff
dem2[dem2 <= cutoff] = 10
dem2[indx] = 0


#cmap = cm.binary
#cmap.set_bad('white',1.)
ax = gdf.plot(color='white', edgecolor='green',figsize=(6,6),linewidth=.5,alpha=.3)
#im = ax.imshow(dem2,extent=[l1,l2,t1,t2])

#imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
dem3 = dem2.astype(np.uint8)
ret, thresh = cv2.threshold(np.fliplr(dem3.transpose()), 1, 255, 0)
im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#im = ax.imshow(dem,vmin=-30, vmax=4,cmap=cmap)

inSpatialRef = osr.SpatialReference()                 # Establish its coordinate encoding
inSpatialRef.ImportFromEPSG(4326) 

outSpatialRef = osr.SpatialReference()
outSpatialRef.ImportFromEPSG(32610)

coordTrans = osr.CoordinateTransformation(inSpatialRef, outSpatialRef)



poly_list = []
for c in contours:
    if len(c) > 100:
        point_list = []
        cc = np.array(c)
        ys = lats[cc[:,0,0]]
        xs = lons[cc[:,0,1]]
        ps = np.array([xs,ys]).transpose()
        p_list = coordTrans.TransformPoints(ps)
        polygon = shapely.geometry.Polygon([[p[0], p[1]] for p in p_list])
        if (polygon.area/1000000) >  .5:           
            poly_list.append(polygon)
        
       
multigon = shapely.geometry.MultiPolygon(poly_list)
multigon = multigon.buffer(0)

channels = all_sfbay.intersection(multigon)
channels2 = sfb_boundary.intersection(multigon)

dfchannels = pd.DataFrame({'Region': ['Channels'], 'Coordinates':[channels2]})

gdfchannels = geopandas.GeoDataFrame(dfchannels, geometry='Coordinates')
gdfchannels.plot(color='gray', ax=ax, alpha=0.5, hatch='\\\\\\\\', edgecolor='white')
gdfchannels.plot(color='none', edgecolor='black', figsize=(6,6), ax=ax, linewidth=1)


##############SHOALS
cutoff_bottom = -3.5
cutoff_top = 1

dem2 = dem.copy()
indx = (dem2 < cutoff_bottom) | (dem2 > cutoff_top)
dem2[(dem2 >= cutoff_bottom) | (dem2 <= cutoff_top)] = 10
dem2[indx] = 0


#imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
dem3 = dem2.astype(np.uint8)
ret, thresh = cv2.threshold(np.fliplr(dem3.transpose()), 1, 255, 0)
im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#im = ax.imshow(dem,vmin=-30, vmax=4,cmap=cmap)

inSpatialRef = osr.SpatialReference()                 # Establish its coordinate encoding
inSpatialRef.ImportFromEPSG(4326) 

outSpatialRef = osr.SpatialReference()
outSpatialRef.ImportFromEPSG(32610)

coordTrans = osr.CoordinateTransformation(inSpatialRef, outSpatialRef)



poly_list = []
for c in contours:
    if len(c) > 100:
        point_list = []
        cc = np.array(c)
        ys = lats[cc[:,0,0]]
        xs = lons[cc[:,0,1]]
        ps = np.array([xs,ys]).transpose()
        p_list = coordTrans.TransformPoints(ps)
        polygon = shapely.geometry.Polygon([[p[0], p[1]] for p in p_list])
        if (polygon.area/1000000) >  .5:           
            poly_list.append(polygon)
        
#        
multigon = shapely.geometry.MultiPolygon(poly_list)
multigon = multigon.buffer(0)
shoals = all_sfbay.intersection(multigon)
shoals2 = sfb_boundary.intersection(multigon)

#for p in poly_list:
#    multigon = multigon.union(p)
    
dfshoals = pd.DataFrame({'Region': ['Shoals'], 'Coordinates':[shoals2]})

gdfshoals = geopandas.GeoDataFrame(dfshoals, geometry='Coordinates')
gdfshoals.plot(color='gray',ax=ax, alpha=0.5, hatch='/////', edgecolor='white')
gdfshoals.plot(color='none', edgecolor='black', figsize=(6,6), ax=ax, linewidth=1)

#fig = plt.gcf()
#cbar = fig.colorbar(im)
                
gdf.plot(color='none', edgecolor='black',figsize=(6,6),linewidth=1,alpha=.3,ax=ax)

if False:
    names = []
    coordinates = []
    chans = []
    for i, f in sectioned_gdf.iterrows():
        chan = f['Coordinates'].intersection(channels)
        shoa = f['Coordinates'].intersection(shoals)
        usgs_reg = f['Coordinates'].intersection(usgs_multigon)
        regi = f['Coordinates']
        names.append(f.Region)
        names.append(f.Region + ' Channels')
        names.append(f.Region + ' Shoals')
        names.append(f.Region + ' USGS')
        coordinates.append(regi)
        coordinates.append(chan)
        coordinates.append(shoa)
        coordinates.append(usgs_reg)
        chans.append(0)
        chans.append(1)
        chans.append(2)
        chans.append(3)
        
    df_sec_channels = pd.DataFrame({'Region': names, 'Coordinates': coordinates,
                                    'Channel': chans})
    section_channels = geopandas.GeoDataFrame(df_sec_channels, geometry='Coordinates')
    ax = section_channels.plot(edgecolor='black',linewidth=.5, column ='Channel')
    gdf3[gdf3['Region']=='USGS_POLARIS'].plot(ax=ax,edgecolor='black')
    
    section_channels.to_file("bay_shapefile/bayarea_sectioned.shp")
