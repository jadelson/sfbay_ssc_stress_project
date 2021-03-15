#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 11:36:45 2017

@author: jadelson
"""

#import matplotlib.pyplot as plt

class PlotFormatter():
    def __init__(self, plt):
        
        
        self.establish_standard_form()
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif') 
        plt.rc('legend',fancybox=False)
    
#        matplotlib.rcParams['text.usetex'] = True
#        matplotlib.rcParams['text.latex.unicode'] = True
#        matplotlib.rcParams['font.size'] = 11
#        matplotlib.rcParams['figure.dpi'] = 130
#        matplotlib.rcParams['savefig.dpi'] = 900
#        matplotlib.rcParams['figure.figsize'] = [8.0, 6.0]

    def start_plot(self, plt):
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')        
        
    def establish_standard_form(self):
        #Font details
        self.fontsize = 10
        
        #Marker details
        self.markers = ['o','x','+','s','>']
        self.markersize = 50
        self.defaultcolor = 'k'
        self.colors = ['b','r','g','k','m']
        self.error_tick_size = 40
        
        #Line details
        self.reg_linewidth = 2
        self.thin_linewidth = 1
        self.thick_linewidth = 4
        self.superthin_linewidth = .1
        
        self.linewidth = self.reg_linewidth
        
        #Labels
        self.ssc_label = r'SSC (mg L$^{-1}$)'
        self.tau_bed_label = r'$\tau_{cw}$ (kg m$^{-1}$ s$^{-2})$)'
        #Save details
        self.dpi = 300
        self.paper_dpi = 300
        self.savedir = '/Users/jadelson/Dropbox/phdResearch/AllOptical/timelapse_work/oct_30_2017_figures/'
        
        self.letters = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n',
                        'o','p','q','r','s','t','u','v','w','x','y','z']
        