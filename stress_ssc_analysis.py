#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 16:44:32 2018

@author: jadelson
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 19 08:44:45 2017

@author: jadelson
"""

#%% DO IMPORTS
import pickle
import numpy as np
import crspy.crspy as cr
import scipy.io as sio
import scipy.optimize as optimize
from sklearn.metrics import r2_score
from multiprocessing import Pool

import stress_ssc_analysis
import matplotlib.pyplot as plt

from plot_formatter_joe import PlotFormatter
plt_format = PlotFormatter(plt)


from scipy.stats import lognorm
import os
import pandas as pd
from datetime import datetime

#from rouse import get_rouse_top


#PLOT CONSTANTS
axsize = 12

resultstring = ''

#%% LOAD CONSTANTS
gm_data_dir = '/Users/jadelson/Dropbox/phdResearch/AllOptical/sfbrspy/grant_madsen_convert'
figure_dir = '/Users/jadelson/Dropbox/phdResearch/AllOptical/sfbrspy/figures/sep2019'

#SAVE_FIGURES = True
LOAD_FROM_FILE = False
LOAD_SINGLE_STRESS_SET = False
READ_FROM_MEMORY= False
MAKE_SAVE_FILES = False
PLOT_ALL_DATA = False
DO_HYDRO_COMPARISON = False
DO_CRIT_STRESS_INSTRUCTIVE = False
DO_CRIT_STRESS_LOCATION = False
DO_CRIT_STRESS_LOCATION_WITH_FLOW = False
DO_CRIT_STRESS_SEASON = False
DO_CRIT_STRESS_WAVES = False
DO_ROUSE = False
DO_TAU_CRIT_ANALYSIS = False
QQ_PLOT = False
DO_FULL_PDF = False
DO_PARTIAL_PDF = False

density = 1025
MIN_SAMPLE = 20

SAVE_FIGURES = False


if __name__ == '__main__':
    DO_CRIT_STRESS_INSTRUCTIVE = True    
    DO_CRIT_STRESS_LOCATION = False
    DO_CRIT_STRESS_LOCATION_WITH_FLOW = False
    DO_CRIT_STRESS_WAVES = False
    DO_CRIT_STRESS_SEASON = False
    QQ_PLOT = True

    SAVE_FIGURES = True


#%% DEFINE STATIC FUNCTIONS
#Nechad function for RS SSC
def nechad(c,rho):
    ssc = c[0]*rho/(1 - rho/0.170991)+c[1]
    return ssc

#def nechad(c,x):
##    c = [29.08789804,  5.93511266]
#    return c[1]*np.exp(x*c[0])

#Sigmoid fit for crit shear stress
def sigmoid(x, x0, C, A, B, k):
    y = A /(B+np.exp(-k*(x-x0))) + C
    return y

def d_sigmoid(x,d,c,a,b,k):
    return -(a*k*np.exp(-k*(x-d)))/(b + np.exp(-k*(x-d)))**2

def dd_sigmoid(x,d,c,a,b,k):
    numerator = a*np.exp(k*(x+d))*(np.exp(d*k) - b*np.exp(k*x))*k**2
    denominator = (np.exp(k*d) + b*np.exp(k*x))**3
#    numerator = a*(k**3)*np.exp(k*(d+x))*(b**2*np.exp(2*k*x)-4*b*np.exp(k*(d+x))+np.exp(2*d*k))
#    denominator = (np.exp(d*k)+b*np.exp(k*x))**4
    return numerator/denominator

def exponential_modified(x,c0,c1,c2, c3, c4):
    return c0*np.power(x,c1) + c3+(x > c2)*(c0*np.power(c2,c1) - c0*np.power(x,c1))
    
def critpoint(x, x0, C, A, B, k):
#    a = np.log(-(-2+np.sqrt(3))*np.exp(x0*k)/B)/k    
    return np.log((0.5*(4*B*np.exp(x0 *k) -  4*np.sqrt(1*np.square(B)*np.exp(2*x0 *k) -  0.25*np.square(B)*np.exp(2*x0 *k))))/np.square(B))/k#Sigmoid jacobian


def mode(x):
    mu = np.mean(np.log(x))
    sigmasquare = np.var(np.log(x))
    return np.exp(mu-sigmasquare)

def get_rouse(ustar_c):
    a = sio.loadmat('joe_contour.mat')             
    b = a['joe_contour']
    
    ws_train = b['ws'][0][0].flatten('F')
    ustar_train = np.abs(b['u_star'][0][0].flatten('F'))
    
    indx = (~np.isnan(ustar_train)) & (~np.isnan(ws_train))             
    x = ustar_train[indx]
    y = ws_train[indx]    

    xpred = ustar_c
    
    fitfunc = lambda a, _x: a[0]*np.power(a[1],a[2]*_x)
    errfunc = lambda a, _x, _y: fitfunc(a,_x) - _y
    
    coefs, success = optimize.leastsq(errfunc, [1,1,-1], args=(x, y))
    ws = fitfunc(coefs,xpred)/1000
    rouse = ws/ustar_c/.41
    
    return rouse
        
        
#stat_functions = {'mode':get_mode, 'median':np.median, 'mean':np.mean, 'std':np.std,'count':len}
plot_type =  {'mean':'+', 'median':'^', 'mode':'s','std':''}


def getgeneric(x, sample, pdftype, returnll=False):
    param = pdftype.fit(sample,loc=0)
    yhat = pdftype.pdf(x,*param)
    if returnll:
        return yhat, pdftype.logpdf(sample,*param).sum(), param
    return yhat

def getfittedpdf(x,sample,returnll=False):    
    from scipy.stats import lognorm as pdftype
#    from scipy.stats import mielke as pdftype
     
    return getgeneric(x, sample, pdftype, returnll)
#    a, l, s = fatiguelife.fit(sample,loc=0)
#    yhat = fatiguelife.pdf(x,a,l,s)
#    if returnll:
#        return yhat, fatiguelife.logpdf(sample,a,l,s).sum()
#    return yhat



def huber_norm(x,delta):
    y = np.power(x[np.abs(x) <= delta],2)/2/delta
    z = np.abs(x[np.abs(x) > delta]) - delta/2
    return sum(y) + sum(z)
    

def mode_pdf(sample):
    min_sample_length = MIN_SAMPLE
    if len(sample) < min_sample_length:
        return(np.nan)
    xs = np.linspace(0,100,10000)
    yhat = getfittedpdf(xs,sample)
    i = np.argmax(yhat)
    if xs[i] > 100:
        return 100
    return(xs[i])

#def piecewise_linear(x, x0, y0, k1, k2):  
def piecewise_linear(x, x0, y0, k2):  
    k1 = 0

    return np.piecewise(x, [x < x0], [lambda x:k1*x + y0-k1*x0, lambda x:k2*x + y0-k2*x0])
        

func = mode_pdf
#func = np.median
#def func(samp):
#    return np.percentile(samp, 75) - np.percentile(samp, 25)
#def func(samp):
#    n, edge = np.histogram(samp, 50)
#    indx = np.argmax(n)
#    return (edge[indx] + edge[indx+1])/2

shoal_max_depth = 5
shoal_min_depth = .5
bounder = ([np.log10(.001),17,0,0], [np.log10(.2),22,np.inf,np.inf])

#%% LOAD DATA

dflow = pd.read_csv('dayflow-results-1997-2018.csv')
df_date = np.array([datetime.strptime(d,'%d-%b-%y') for d in dflow['Date']])
df_out = np.array([float(d) for d in dflow['OUT']])

flow_dict = {}

for a,b in zip(df_date, df_out):
    flow_dict[a.strftime('%Y-%m-%d')] = b
    
    
dataset_name = 'l7_full_dataset.dk'
if (not 'd_load' in locals()) or (not d_load == dataset_name):
    with open(dataset_name,'rb') as f:
        d = pickle.load(f)
        d_load = dataset_name
    
    #phase
#    with open('tide_phase.dk','rb') as f:
#        tl = pickle.load(f)
    
    #Tau
    tau_orig = d['tau_b']
    
    #SSC
    c = [414.202,  0]
#    c[1] = -1*nechad(c,min(d['RHOW_661']))
    #c[1] = 0
    ssc_orig = nechad(c, d['RHOW_661'])

    #Depth
    D_orig = d['depth'] + d['eta']
    eta_orig = d['eta']
    depth_orig = d['depth']
    
    #Velocity
    umag_orig = np.sqrt(np.square(d['u']) + np.square(d['v']))
    u_orig = d['u']
    v_orig = d['v']
    
    #Wave properties
    Hs_orig = d['a0']*2
    k_orig = d['k']
    ub_orig = d['ub']
    
    #Ustars
    ustr_orig = d['ustar_cw']    

    #Locations
    x = d['easting']
    y = d['northin']
    

        
    
    #Dates
    dates = np.array([[d.year,d.month,d.day] for d in d['date']])
    month = np.asarray([d.month for d in d['date']])
    year = np.asarray([d.year for d in d['date']])
    day = np.asarray([d.day for d in d['date']])

    #Dayflow access string
    date_string = []
    dayflow = []
    for _date in d['date']:
        string_key = _date.strftime('%Y-%m-%d')
        date_string.append(string_key)
        dayflow.append(flow_dict[string_key])
    dayflow = np.array(dayflow)
        

#%% Establish subsets (if you want to rerun this: subsets_established = False)
if (not 'subsets_established' in locals()) or (not subsets_established):
    subsets_established = True 

    
    #SSC Limit
    indx_fundamental = d['RHOW_661'] < .15 
    
    #Locations subssets
    lat1 = 4.18e6
    lat2 = 4.203e6
    lon4 = 5.75e5
    lon5 = 5.42e5 #ocean removal
    lat5 = 4.1879e6 #ocean removal
    lon6 = 5.49513e5 #ocean removal
    lat6 = 4.14842e6 #ocean removal
    lon7 = 6.05e5 #delta removal
    lon8 = 5.464e5 #golden gate removal
    lat8a = 4.182e6
    lat8b = 4.188e6

    location_indx = [None]*4
    location_indx[0] = (y < lat1) & (x >= lon5)
    location_indx[1] = (y >= lat1) & (y < lat2) & (x < lon4) & (x >= lon5)
    location_indx[2] = (y >= lat2) & (x < lon4) & (x >= lon5)
    location_indx[3] = (y >= lat1) & (x >= lon4) & (x >= lon5)
    locationlabels = ['South Bay', 'Central Bay', 'San Pablo Bay', 'Suisun and Grizzly Bays']
    locationlabels_multiline = ['South Bay', 'Central Bay', 'San Pablo Bay', 'Suisun and\nGrizzly Bays']
    
    #Depth subsets
    depthlabels = [r'$D < 1$',r'$1 \leq D < 5$',r'$5 \leq D < 10$',r'$10 \leq D $',]
    depthindx = [(D_orig >= 0)& (D_orig < 1),(D_orig >= 1)& (D_orig < 3),(D_orig >= 3)& (D_orig < 10),(D_orig >= 10)]
    
    #Date subsets
    winter = (month >= 11 )| (month <= 3)
    summer = (month > 4) & (month < 11) 
    
    seasonlabels = ['Winter','Summer']
    stat_label_names = {'mean':'Mean', 'median':'Median', 'mode':'Mode', 'std':'Std. Dev.', 'count':'Count'}



#%% get stress
def get_taus(z0):
    filepath = gm_data_dir + "/gm_con_%s_.dat" % str(z0)
    if os.path.isfile(filepath):
        with open(filepath,'rb') as f:
            data_array = pickle.load(f)
        tau_cw_orig = data_array[1][0]
        tau_c_orig = data_array[1][1]
        tau_w_orig = data_array[1][2]
        return {'z0':z0, 'cw':tau_cw_orig, 'c':tau_c_orig, 'w':tau_w_orig, 'indx':indx_fundamental}                     
    else:
        return compute_taus(z0, True)
 

#%% compute stress
def compute_taus(z0, save_file):
    re_ustr_cw =[]# np.zeros(len(indx_fundamental))
    re_ustr_c =[]# np.zeros(len(indx_fundamental))
    re_ustr_w =[]# np.zeros(len(indx_fundamental))
    
    model_D = [0,0.1, 0.6, 2.0, 6.5, 8.5,100000]
    model_z0 = [1.23e-3,1.23e-3,2.29e-3,1.16e-3,2.5e-4,1e-5,1e-5]
    from scipy.interpolate import interp1d
    f = interp1d(model_D, model_z0)

    kb = 30*z0
    z0_orig = f(D_orig)
    ustrc_est = np.sqrt(u_orig*u_orig+v_orig*v_orig)*.41/(D_orig*(np.log(D_orig/z0_orig) + z0_orig/D_orig- 1))
    zr = D_orig/2
    u_mag_zr = ustrc_est/.41*np.log(zr/z0_orig)
    u_mag_zr[u_mag_zr<0] = 0
    ub_ = d['ub'].tolist()
    omega_ = d['omega'].tolist()
    phi_c_ = d['phi_c'].tolist()
    
    
    #    u_mag_zr_ = u_mag_zr.tolist()
    #    zr_ = zr.tolist()
    #    ustrc_est_ = ustrc_est.tolist()
    for k in range(0,len(tau_orig)):
        if indx_fundamental[k]:
    
            ustrc, ustrr, ustrwm, dwc, fwc, zoa = cr.m94( ub_[k], omega_[k], u_mag_zr[k], zr[k], phi_c_[k], kb)
    
            re_ustr_c.append(ustrc)
            re_ustr_cw.append(ustrr)
            re_ustr_w.append(ustrwm)
        else:
            re_ustr_c.append(np.nan)
            re_ustr_cw.append(np.nan)
            re_ustr_w.append(np.nan)
            
    re_ustr_cw = np.array(re_ustr_cw)
    re_ustr_c = np.array(re_ustr_c)
    re_ustr_w = np.array(re_ustr_w)
    
    tau_cw_orig = re_ustr_cw*re_ustr_cw*density
    tau_c_orig = re_ustr_c*re_ustr_c*density
    tau_w_orig = re_ustr_w*re_ustr_w*density
    

    if save_file:
        filepath = gm_data_dir + "/gm_con_%s_.dat" % str(z0)
        print(filepath)
        data_struct = [z0, [tau_cw_orig, tau_c_orig, tau_w_orig]]
        with open(filepath,'wb') as f:
            pickle.dump(data_struct, f)
    
    return {'z0':z0, 'cw':tau_cw_orig, 'c':tau_c_orig, 'w':tau_w_orig, 'indx':indx_fundamental}                
        

  
 
#%% Load previously computed stress for multiple z0
    
# Reads in data that has already been computed by z0
if LOAD_FROM_FILE:

    all_data = {}
    Z0=[]
    for filename in os.listdir(gm_data_dir):
        if not filename.startswith('gm_con'):
            continue
        if 'indx' in filename:
            with open(gm_data_dir + '/' + filename,'rb') as f:
                indx_fundamental = pickle.load(f)
        else:#if '0.5' in filename:
            
            print(filename)
            splits = filename.split('_')
            z0name = splits[2].split('.dat')
#                z0name = re.findall('\d+\.\d+',filename)
            z0 = float(z0name[0])
            if z0 > .001:
                continue

            
            with open(gm_data_dir + '/' + filename,'rb') as f:
                data_array = pickle.load(f)
            z0 = data_array[0]
            Z0.append(z0)
            all_data[z0] = data_array[1]
    Z0 = np.sort(np.array(Z0))

#%% load a previously computed data for single z0
if LOAD_SINGLE_STRESS_SET:
    z0 = 6.661e-4
    
    try:
        recent_data
    except NameError:
        print('exception')
        recent_data = {'z0':None,'indx':None}

    if recent_data == None or not ((recent_data['z0'] == z0) and all(recent_data['indx'] == indx_fundamental)):
        recent_data = get_taus(z0)
            
#%%                
if READ_FROM_MEMORY:
    Z0, POPT_ORIG = stress_ssc_analysis.get_curves()
#    BINS, POPT_ORIG_BIN = get_curves_by_bin()

      
#%%        
if MAKE_SAVE_FILES:
    import multiprocessing
    print('this may take a while...')
    done = [float(x.split('_')[2]) for x in os.listdir(gm_data_dir) if 'gm_con' in x]

    Z0 = np.logspace(-6,-3,20)
    Z0 = np.flip(Z0, 0)
    all_data = {}
    i = 1
    
    
    def compute_gm(z0):
        if z0 in done:
            return
        print(z0, flush=True)
        
        tau_data = compute_taus(z0, True)

        tau_cw_orig = tau_data['cw']
        tau_c_orig = tau_data['c']
        tau_w_orig = tau_data['w']
        all_data[z0] = [tau_cw_orig, tau_c_orig, tau_w_orig]
        
        return [tau_cw_orig, tau_c_orig, tau_w_orig]
    
    with open('./gm_con_indx.dat','wb') as f:
        pickle.dump(indx_fundamental, f)
        
    with multiprocessing.Pool() as p:
        p.map(compute_gm, Z0)
        p.close()
        p.join()

 #%%  
if PLOT_ALL_DATA:
    tau = tau_orig[indx_fundamental]
    ssc = ssc_orig[indx_fundamental] 
    ssc = ssc[::10]
    tau = tau[::10]

    bins = np.logspace(-4,1,40)
    plt.scatter(tau,ssc,facecolor='k',edgecolor='none',alpha=.01)
    for b in bins:
        a = np.linspace(0,100,1000)
        plt.plot(a*0 + b, a, 'r--', linewidth=plt_format.thin_linewidth)
    ax = plt.gca()
    ax.set_xscale('log') 
    plt.ylim([0,100])
    plt.xlim([10**-4,1])
    plt.xlabel(plt_format.tau_bed_label, fontsize=plt_format.fontsize)
    plt.ylabel(plt_format.ssc_label, fontsize=plt_format.fontsize)
    
    if SAVE_FIGURES:
        plt.savefig(plt_format.savedir + 'alldatabinned.png',dpi=plt_format.dpi)




#%%











   
z0 = 6.661e-4
try:
    recent_data
except NameError:
    recent_data = {'z0':None,'indx':None}
if recent_data == None or not ((recent_data['z0'] == z0) and all(recent_data['indx'] == indx_fundamental)):
    print('uhoh gotta do some computations')
    recent_data = get_taus(z0)
   

#Get raw data
tau_cw_orig = recent_data['cw']
tau_c_orig = recent_data['c']
tau_w_orig = recent_data['w']


tau = tau_cw_orig








for loc_indx_run in [0]:#range(4):
    if DO_CRIT_STRESS_INSTRUCTIVE:
        q_true = []
        q_fit1 = []
        q_fit2 = []
        fig = plt.figure(figsize=(6,6))
        
        try:
            use_saved_outer
        except NameError:
            use_saved_outer = False
            use_saved_inner = False
            
        if not use_saved_outer:    
            z0 = 6.661e-4
            try:
                recent_data
            except NameError:
                recent_data = {'z0':None,'indx':None}
            if recent_data == None or not ((recent_data['z0'] == z0) and all(recent_data['indx'] == indx_fundamental)):
                print('uhoh gotta do some computations')
                recent_data = get_taus(z0)
       
            
            #Get raw data
            tau_cw_orig = recent_data['cw']
            tau_c_orig = recent_data['c']
            tau_w_orig = recent_data['w']
            
         
            #Filter (keep relevant data)
            #    loc=2
            indx = indx_fundamental & ~np.isnan(tau_cw_orig) & ~np.isnan(tau_c_orig) & ~np.isnan(tau_w_orig)
            
            indx = indx & (ssc_orig > 0) & winter

    #        indx = indx & (D_orig < 8)& (D_orig > .5)
        
            indx = indx & (D_orig < 5) & (D_orig > 0.5)
#            indx = indx  & (location_indx[0] |  location_indx[2])#& summer ##REMOVE IF CONFUSED
            indx = indx & location_indx[loc_indx_run]
            xdata = tau[indx] + 1e-8
            ydata = ssc_orig[indx]
            
            #Bin and filter
            nbins = 24
            min_stress = 10**-3
            max_stress = 1
                
            xs = np.logspace(np.log10(min_stress) ,np.log10(max_stress), nbins, endpoint=True)
        
            digitized = np.digitize(xdata, xs, right=True)
            
            digilist = [ydata[(digitized == i)] for i in range(0,nbins)] 
            with Pool(8) as pool:
                ys = np.array(pool.map(func, digilist))
            
        plt.plot(xs,ys,'o')
        if True:
    
            if not use_saved_outer:
                #Curve fit
                
                gheight = 60
                gtop = 36
                gbot = 21
                gpad = gheight - gtop - gbot
    
            xss = np.linspace(0.00001,100,10000)
            fitlinex = []
            fitliney = []
                
            if not use_saved_inner:
                n = int(nbins/2)
                YSS = [None] * nbins
                PARAM = [None] * nbins
    
            for i in range(nbins):
                sample = ydata[(digitized == i)]
                if len(sample) < 30:
                    fitliney.append(np.nan)
                    continue
                if use_saved_inner:
                    yss = YSS[i]
                    param = PARAM[i]
                else:
                    if len(sample) > 30:
                        yss, ll, param = getfittedpdf(xss, sample, True)
                    else:
                        yss = np.nan
                        param = [np.nan]*3
                    YSS[i] = yss
                    PARAM[i] = param
                    
                funcpoint = xss[np.argmax(yss)]
        
                fitliney.append(funcpoint)
                if QQ_PLOT:
                    q_true.append(np.percentile(sample, np.linspace(1,99,100)))
                    q_fit1.append(lognorm.ppf(np.linspace(.1,.99,100),*param))
                    param2 = lognorm.fit(sample,loc=0)
                    q_fit2.append(lognorm.ppf(np.linspace(.1,.99,100),*param2))
                
    #        fitliney = ys
                
                
            for i in range(n):
                sample = ydata[(digitized == 2*i)]
                ax = plt.subplot2grid((gheight,n), (0,i), rowspan=gtop, colspan=1)
                
                sample = ydata[(digitized == i*2)]
                N = len(ydata)
                try:
    #        pdf = np.histogram(sample[sample<100],bins=100)
                    pdf = np.histogram(sample[sample<102],bins=51,density=True)
            
                    plt.plot(pdf[0],pdf[1][:-1],'k-',linewidth=.5)
        
        
        
                    yss = YSS[i*2]
        #            
                    plt.plot(yss, xss,'k:',linewidth=2,alpha=.75)
    

    #                funcpoint = int(np.round(func(sample)/2))-1
    #                ax.scatter(pdf[0][funcpoint]/N,pdf[1][funcpoint],color='k',marker='o',facecolor='w')
    
                    funcpoint = xss[np.argmax(yss)]
                    
                    if np.mod(i,1) == 0:
                        ax.scatter(np.max(yss),funcpoint,color='k',marker='o',facecolor='w')
                except:
                    ax.scatter(func(sample),.1,color='k',marker='o',facecolor='w')
                ax.set_ylim([0,75])
                ax.set_xticks([])
                ax.set_yticks([])
                if i == 0:
        #            ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
                    ax.set_yticks([0,15,30,45,60,75])
                    ax.set_ylabel(r'SSC (mg L$^{-1}$)',fontsize=plt_format.fontsize)
        #            ax.yaxis.tick_right()
                if i == 6:
                    ax.set_xlabel(r'$\tau_{cw}$--binned probability density',fontsize=plt_format.fontsize,labelpad=5)
                ax.invert_xaxis()
        #        for spine in plt.gca().spines.values():
        #            spine.set_visible(False)
                    
            ax = plt.subplot2grid((gheight,n), (gtop+gpad,0), colspan=n, rowspan=gbot)
        
            ax.scatter(xdata,ydata,s=1,color='xkcd:grey', marker='.',alpha=.1)
    #        plt.plot(tau_crit,sigmoid(tau_crit,*popt), 'kx',markersize=10,markeredgewidth=3, label=r'$\tau_{\textrm{crit}}$')
    #        plt.plot(xcurve, sigmoid(xcurve,*popt),'k--',linewidth=2,alpha=.8,label=r'Sigmoid Fit')
    
            fitliney = np.array(fitliney)
            def piece_error(_p):
                err = piecewise_linear(np.log10(xs[xs<.6]),*_p) - fitliney[xs<.6]
    #            err = piecewise_linear(np.log10(xs),*_p) - fitliney
                return np.linalg.norm(err,1)
            
            bnd_1 = np.mean(fitliney[0:3])
#            bools = (fitliney-bnd_1)/bnd_1>.2
            bools = ((ys-bnd_1)/bnd_1) > (4/bnd_1)
            trues = [i for i, x in enumerate(bools) if x]
    
            bnd_2_a = -2.8#xs[trues[0]-2] 
            bnd_2_b = np.log10(1.5)#xs[trues[0]] 
            bnds = ((bnd_2_a, bnd_2_b), (bnd_1, bnd_1), (0, 100))
            p_full = optimize.minimize(piece_error, x0=[-1, 15, 10], bounds=bnds)
            p = p_full.x
    #        p , e = optimize.curve_fit(piecewise_linear, np.log10(xs[xs<.6]), fitliney[xs<.6], method='lm')#'dogbox')#,bounds=bounder)
            # r2 = r2_score(ys,piecewise_linear(np.log10(xs),*p))
            # r2string = '%.2f' % r2 
            r2string = 'r2string not loaded'
            plt.plot(xs, piecewise_linear(np.log10(xs),*p), 'k-',linewidth=1)
            plt.plot(10**p[0],piecewise_linear(p[0],*p),'kx',markersize=12)
            print(10**p[0])
            plt.plot(xs, fitliney,'k--',linewidth=2,alpha=.8)#,label=r'Sigmoid Fit')
            
            ax.scatter(xs[::2], ys[::2],color='k',marker='o',facecolor='w')
            #Add axis labels
            ax.set_xlim([10**-3, 1])
            ax.set_ylim([0,75])
            ax.set_yticks([0,15,30,45,60,75])
        #    ax.set_yscale('log')
            ax.set_xscale('log')
        
            ymin, ymax = ax.get_ylim()
            yalign = (ymax - ymin)*.95 + ymin
            
            resultstring+='All Shoals: ' + str(np.round(10**p[0],3))+ ', ' + r2string + '\n'
        #        
            #Make plot pretty
            ax.set_xlabel(r'$\tau_{cw}$ (Pa)',fontsize=plt_format.fontsize)
            ax.set_ylabel(r'SSC (mg L$^{-1}$)', fontsize=plt_format.fontsize)
           
        #    fig.tight_layout()
         
        
            #Save figure    
            if SAVE_FIGURES:
                plt.savefig(figure_dir + '/instructive_crit_stress.%d.png' % loc_indx_run, dpi=plt_format.dpi)
    
    
    
    
    
    
    
    

    #%%
# if True:
    def latex_float(x):
    #    float_str = "{0:.2g}".format(f)
    
        exp = int(np.floor(np.log10(abs(x))))
        xnew = x / 10**exp
        if exp == 0:
            return '$%.2f$' % x
        return '$%.1f\\times10^{%d}$' % (xnew, exp)
        
    if QQ_PLOT:
        llll = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
        # fig = plt.figure(figsize=(6,6))
        fig, axs = plt.subplots(3, 2, sharex='col', sharey='row',figsize=(5,6))
        ax = fig.add_subplot(111, frameon=False) 
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
        # ax.grid(False)# The big subplot
        iii = 0
        I = [0,4,9,14,19,23]
        for ii in range(3):
            for jj in range(2):
                i = I[iii]
                
        #        plt.subplots(3,2,ii+1, sharex='col')
                axs[ii,jj].plot(np.linspace(0,120),np.linspace(0,120),'k-',linewidth=.3)
                axs[ii,jj].plot(q_fit1[i][::1], q_true[i][::1], 'k+', fillstyle='none', linewidth=.5)
    #            axs[ii,jj].plot(q_fit2[i][::8], q_true[i][::8], 'ko', fillstyle='none', linewidth=.5)
                axs[ii,jj].text(2, 70, llll[iii] + ' ' + r'$\tau_{cw}$ = ' + latex_float(xs[i]), fontsize=plt_format.fontsize)
                axs[ii,jj].set_xlim([0,80])
                axs[ii,jj].set_ylim([0,80])
                axs[ii,jj].set_aspect('equal', adjustable='box')
                
                iii+=1
        ax.set_ylabel('Measured ' + plt_format.ssc_label, labelpad=-10)
        ax.set_xlabel('Lognormal fit of ' + plt_format.ssc_label, labelpad=20)
        fig.tight_layout()

        sum_qq = 0
        for i in range(24):
#            print(r2_score(q_true[i], q_fit1[i]), r2_score(q_true[i], q_fit2[i]),r2_score(q_true[i], q_fit1[i])-r2_score(q_true[i], q_fit2[i]))
            sum_qq += r2_score(q_true[i], q_fit1[i])-r2_score(q_true[i], q_fit2[i])
    
    
        if SAVE_FIGURES:
            plt.savefig(figure_dir + '/qq_%d.png' % loc_indx_run, dpi=plt_format.dpi)
    
    


#%%


if DO_FULL_PDF:
    indx = indx_fundamental & (D_orig > 1)
    ssc = ssc_orig[indx]
    
    plt.figure(figsize=(5,4))
    xfit = np.linspace(0,200,1000)
        

    bins = np.linspace(0,102,100)
    pdf, edge = np.histogram(ssc, bins, density=True)
    plt.plot(bins[0:-1],pdf, 'k--',linewidth=plt_format.thin_linewidth)
    param = lognorm.fit(ssc, loc=0)
    #param2 = lognorm.fit(np.random.choice(ssc, int(np.floor(len(ssc)/2))), loc=0)
    
    pdf_fit = lognorm.pdf(bins,*param)
    plt.plot(bins, pdf_fit, 'k-',linewidth=plt_format.thin_linewidth)
    plt.plot(np.mean(ssc)*np.ones((1000)),np.linspace(0,1,1000),'k:', linewidth=plt_format.thin_linewidth)
    plt.plot(np.median(ssc)*np.ones((1000)),np.linspace(0,1,1000),'k-.', linewidth=plt_format.thin_linewidth)
            
    plt.plot(lognorm.mean(*param), lognorm.pdf(lognorm.mean(*param), *param), 'ko')
    plt.plot(lognorm.median(*param), lognorm.pdf(lognorm.median(*param), *param), 'ks')
    plt.xlim([0,100])
    plt.ylim([0,.05])
    print(param)
    order = [0,2,1,3]
#    handles, legend_labels = plt.gca().get_legend_handles_labels()

#    plt.legend([handles[idx] for idx in order],[legend_labels[idx] for idx in order], fontsize=plt_format.fontsize)
#    plt.legend(patch_leg, label_leg,fontsize=plt_format.fontsize)
    plt.xlabel(plt_format.ssc_label,fontsize=plt_format.fontsize)
    plt.ylabel(r'Probability density', fontsize=plt_format.fontsize)
    plt.tight_layout()
    if SAVE_FIGURES:
        plt.savefig(figure_dir + '/all_ssc_pdf.png',dpi=plt_format.paper_dpi)

if DO_PARTIAL_PDF:
    wave = Hs_orig > .125
    nowave = Hs_orig < .1
    indx = indx_fundamental & (D_orig > 1)
    indxw = indx & wave
    indxn = indx & nowave
    sscw = ssc_orig[indxw]
    sscn = ssc_orig[indxn]
    plt.figure(figsize=(5,4))
    xfit = np.linspace(0,200,1000)
        

    bins = np.linspace(0,102,50)
    pdfw, edgen = np.histogram(sscw, bins, density=True)
    pdfn, edgen = np.histogram(sscn, bins, density=True)
    
    plt.plot(bins[0:-1],pdfw, 'k--',linewidth=plt_format.thin_linewidth)
    plt.plot(bins[0:-1],pdfn, 'k-.',linewidth=plt_format.thin_linewidth)
    
    paramw = lognorm.fit(sscw, loc=0)
    paramn = lognorm.fit(sscn, loc=0)
    
    pdf_fitw = lognorm.pdf(bins,*paramw)
    plt.plot(bins, pdf_fitw, 'k-',linewidth=plt_format.thin_linewidth)
    print(paramw)
    print(paramn)
    
    pdf_fitn = lognorm.pdf(bins,*paramn)
    plt.plot(bins, pdf_fitn, 'k:',linewidth=plt_format.thin_linewidth)
    
    plt.xlim([0,100])
    plt.ylim([0,.05])
    
    order = [0,2,1,3]
#    handles, legend_labels = plt.gca().get_legend_handles_labels()

#    plt.legend([handles[idx] for idx in order],[legend_labels[idx] for idx in order], fontsize=plt_format.fontsize)
#    plt.legend(patch_leg, label_leg,fontsize=plt_format.fontsize)
    plt.xlabel(plt_format.ssc_label,fontsize=plt_format.fontsize)
    plt.ylabel(r'Probability density', fontsize=plt_format.fontsize)
    plt.tight_layout()
    if SAVE_FIGURES:
        plt.savefig(figure_dir + '/wave_ssc_pdf.png',dpi=plt_format.paper_dpi)


        
    









#%%


#Bin and filter
nbins = 20
min_stress = 10**-3
max_stress = 1
            
def func_top(samp):
    if len(samp) < MIN_SAMPLE:
        return np.nan
    return np.percentile(samp, 75)

def func_bot(samp):
    if len(samp) < MIN_SAMPLE:
        return np.nan
    return np.percentile(samp, 25)

def plot_stress_curve(tau, indx, loc, names, resultstring, fig, dofit=True):
    markertype = ['o','s']
    linetype = ['k-','k--']
    k = 0
    for xdata in [tau[indx] + 1e-8]:#, tau_c_orig[indx] + 1e-8]:
        ydata = ssc_orig[indx]
    
        
        xs = np.logspace(np.log10(min_stress) ,np.log10(max_stress), nbins, endpoint=False)
        xs = np.log10(xs)
        digitized = np.digitize(np.log10(xdata), xs, right=True)
        
        digilist = [ydata[(digitized == i)] for i in range(0,nbins)] 
        
        
        
        with Pool(8) as pool:
            ys = np.array(pool.map(func, digilist))
            y_top = np.array(pool.map(func_top, digilist))
            y_bot = np.array(pool.map(func_bot, digilist))
            
        notnanind = ~np.isnan(ys)
        xs = xs[notnanind]
        ys = ys[notnanind]
        y_top = y_top[notnanind]
        y_bot = y_bot[notnanind]
        
        xcurve = np.linspace(-5,1,10000)
#        print(len(xs), len(y_top), len(y_bot))
        #Plot
        ax = fig.add_subplot(len(names),1,loc+1)
        ax.scatter(10**xs, ys, color='k', marker=markertype[k], facecolor='w', 
                   linewidth=.5, label='test')
        ax.scatter(10**xs, y_top, color='k', marker='_', linewidth=.5)
        ax.scatter(10**xs, y_bot, color='k', marker='_', linewidth=.5)
      
        if dofit:
            #Curve fit
            def this_piece_error(_p):
                err = piecewise_linear(xs,*_p) - ys
                return np.linalg.norm(err[~np.isnan(err)],1)
#                return huber_norm(err[~np.isnan(err)], 2)
    
            bnd_1 = np.mean(ys[:5])
            
#            bools = (ys-bnd_1)/bnd_1>.25
            bools = ((ys-bnd_1)/bnd_1) > (6/bnd_1)
            
            
            trues = [i for i, x in enumerate(bools) if x]
#            r = 0np.random.normal(1)/20*dofit
            bnd_2_a = -6#xs[trues[0]-2] 
            bnd_2_b = np.log10(.5)#xs[trues[0]] 
            bnds = ((bnd_2_a, bnd_2_b), (bnd_1, bnd_1), (0, 100))
#            print('curve fit')
            p_full = optimize.minimize(this_piece_error, x0=[-1, 15, 10], bounds=bnds)
#            print('curve fit done')
            this_p = p_full.x
#            p , e = optimize.curve_fit(piecewise_linear, xs[xs<-.4], ys[xs<-.4], method='lm')#'dogbox')#,bounds=bounder)
#            print(len(ys), bnd_1, (bnd_2_a, bnd_2_b), this_p[0])


            r2 = r2_score(ys,piecewise_linear(xs,*this_p))
            r2string = '%.2f' % r2
#            print(this_p[0])
            resultstring += names[loc] + ': ' + str(np.round(np.mean(ssc_orig[indx]),2)) + ', ' + str(np.round(10**this_p[0],3)) 
            resultstring += ', ' + r2string + '\n'
            ax.plot(10**xcurve, piecewise_linear(xcurve, *this_p), linetype[k], 
                    linewidth=1)
            ax.plot(10**this_p[0],piecewise_linear(this_p[0],*this_p),'kx', 
                    markersize=12)
            
        k += 1
        
    #Add axis labels
    ax.set_xlim([10**-3, 1])
    ax.set_ylim([0,60])
    ax.set_xscale('log')
#        ax.set_yscale('log') 
    if not loc ==len(names) -1:
        ax.get_xaxis().set_ticks([])

    return ax, resultstring



#%%
     
if DO_CRIT_STRESS_SEASON:
    nplots = 2
    
    winter = (month >= 12 )| (month <= 3)
    summer = (month > 4) & (month < 11) 

    axis_space = [[20,35],[10,30],[20,50],[30,50]]
    fig = plt.figure(figsize=(6,4))
    axall = plt.subplot(111)
    axall.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
        
    z0 = 6.661e-4
    try:
        recent_data
    except NameError:
        recent_data = {'z0':None,'indx':None}
    if recent_data == None or not ((recent_data['z0'] == z0) and all(recent_data['indx'] == indx_fundamental)):
        print('uhoh gotta do some computations')
        recent_data = get_taus(z0)
    
    
    #Get raw data
    tau_cw_orig = recent_data['cw']
    tau_c_orig = recent_data['c']
    tau_w_orig = recent_data['w']
    
    season_indx = [winter, summer]
    labels = ['Winter','Summer']
    for loc in range(0,nplots):
    #Filter (keep relevant data)
    #    loc=2
        indx = indx_fundamental & ~np.isnan(tau_cw_orig) 
        
        indx = indx & (D_orig < shoal_max_depth) & (D_orig > shoal_min_depth)
        
        indx = indx & season_indx[loc] #& (location_indx[2]|location_indx[0])
        indx = indx & (location_indx[2]| location_indx[0])
        
        ax, resultstring = plot_stress_curve(tau, indx, loc, labels, resultstring,fig)
        
        ymin, ymax = ax.get_ylim()
        yalign = (ymax - ymin)*.95 + ymin
        ax.text(1.1*10**-3, yalign, llll[loc] + ' ' +labels[loc],fontsize=axsize, horizontalalignment='left', verticalalignment='top')

#        
    #Make plot pretty
    axall.set_xlabel(r'$\tau_{cw}$ (Pa)',fontsize=axsize)
    axall.set_ylabel(r'SSC (mg L$^{-1}$)', fontsize=axsize)
    
    fig.tight_layout()
       
    #Save figure    
    if SAVE_FIGURES:
        plt.savefig(figure_dir + '/season_crit_stress.png', dpi=plt_format.dpi)    
#%%

     
if DO_CRIT_STRESS_WAVES:
    nplots = 2
    
    wave = Hs_orig > .125
    nowave = Hs_orig < .1

    axis_space = [[20,35],[10,30],[20,50],[30,50]]
    fig = plt.figure(figsize=(6,4))
    axall = plt.subplot(111)
    axall.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
        
    z0 = 6.661e-4
    try:
        recent_data
    except NameError:
        recent_data = {'z0':None,'indx':None}
    if recent_data == None or not ((recent_data['z0'] == z0) and all(recent_data['indx'] == indx_fundamental)):
        print('uhoh gotta do some computations')
        recent_data = get_taus(z0)
    
    
    #Get raw data
    tau_cw_orig = recent_data['cw']
    tau_c_orig = recent_data['c']
    tau_w_orig = recent_data['w']
    
    season_indx = [wave, nowave]
    labels = ['Waves', 'No Waves']
    
    for loc in range(0,nplots):
    #Filter (keep relevant data)
    #    loc=2
        indx = indx_fundamental & ~np.isnan(tau_cw_orig) & ~np.isnan(tau_c_orig) & ~np.isnan(tau_w_orig)
        
        indx = indx  & location_indx[0]#(location_indx[2] | location_indx[0]) 
        indx = indx & (D_orig < shoal_max_depth) & (D_orig > shoal_min_depth)#& (D_orig > 1)
        
        indx = indx & season_indx[loc]
#        indx = indx & (location_indx[2])# | location_indx[0])
        ax, resultstring = plot_stress_curve(tau_cw_orig, indx, loc, labels, resultstring,fig)

        
        ymin, ymax = ax.get_ylim()
        yalign = (ymax - ymin)*.95 + ymin
        plt.text(1.1*10**-3, yalign, llll[loc] + ' ' + labels[loc],fontsize=axsize, horizontalalignment='left', verticalalignment='top')

    #Make plot pretty
    axall.set_xlabel(r'$\tau_{c}$ (Pa)',fontsize=axsize)
    axall.set_ylabel(r'SSC (mg L$^{-1}$)', fontsize=axsize)
    
    fig.tight_layout()
       
    #Save figure    
    if SAVE_FIGURES:
        plt.savefig(figure_dir + '/wave_crit_stress.png', dpi=plt_format.dpi)  

 


#%%
llll = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']

#DO_CRIT_STRESS_LOCATION = True
if DO_CRIT_STRESS_LOCATION:
    axis_space = [[20,35],[10,35],[20,90],[30,80]]
    fig = plt.figure(figsize=(6,6))
    axall = plt.subplot(111)
    axall.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
        
    z0 = 6.661e-4
    try:
        recent_data
    except NameError:
        recent_data = {'z0':None,'indx':None}
    if recent_data == None or not ((recent_data['z0'] == z0) and all(recent_data['indx'] == indx_fundamental)):
        print('uhoh gotta do some computations')
        recent_data = get_taus(z0)
    
    
    #Get raw data
    tau_cw_orig = recent_data['cw']
    tau_c_orig = recent_data['c']
    tau_w_orig = recent_data['w']
    
    for loc in range(0,4):
#        if not loc == 1:
#            continue
    #Filter (keep relevant data)
    #    loc=2
        indx = indx_fundamental & ~np.isnan(tau_cw_orig)  #& ~np.isnan(tau_c_orig) & ~np.isnan(tau_w_orig)
        indx = indx #& nowave
#        indx = indx & (D_orig < 3) #& (D_orig > 1)
                
        indx = indx & (D_orig > 5)#(D_orig < shoal_max_depth)&  (D_orig > shoal_min_depth)
        indx = indx & location_indx[loc]
        resultstring +=  str(np.round(np.mean(ssc_orig[indx]),2)) + ' & '

#        continue
        
        if not loc == 4:
            ax, resultstring = plot_stress_curve(tau, indx, loc, locationlabels, resultstring,fig)
        else:
            ax, resultstring = plot_stress_curve(tau, indx, loc, locationlabels, resultstring,fig, dofit=False)
        
        ymin, ymax = ax.get_ylim()
        
        if not loc == 4:
            yalign = (ymax - ymin)*.95 + ymin
            plt.text(1.1*10**-3, yalign, llll[loc] + ' '+ locationlabels[loc],fontsize=axsize, horizontalalignment='left', verticalalignment='top')
        else:
            yalign = (ymax - ymin)*.20 + ymin
            plt.text(1.1*10**-3, yalign, llll[loc] + ' '+locationlabels[loc],fontsize=axsize, horizontalalignment='left', verticalalignment='top')


    #Make plot pretty
    axall.set_xlabel(r'$\tau_{cw}$ (Pa)',fontsize=axsize)
    axall.set_ylabel(r'SSC (mg L$^{-1}$)', fontsize=axsize)
    
    fig.tight_layout()
       
    #Save figure    
    if SAVE_FIGURES:
        plt.savefig(figure_dir + '/location_crit_stress.png', dpi=plt_format.dpi)    

#DO_CRIT_STRESS_LOCATION = True
if DO_CRIT_STRESS_LOCATION_WITH_FLOW:
    resultstring += 'Dayflow  shoals \n'
    axis_space = [[20,35],[10,35],[20,90],[30,80]]
    fig = plt.figure(figsize=(6,6))
    axall = plt.subplot(111)
    axall.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
        
    z0 = 6.661e-4
    try:
        recent_data
    except NameError:
        recent_data = {'z0':None,'indx':None}
    if recent_data == None or not ((recent_data['z0'] == z0) and all(recent_data['indx'] == indx_fundamental)):
        print('uhoh gotta do some computations')
        recent_data = get_taus(z0)
    
    
    #Get raw data
    tau_cw_orig = recent_data['cw']
    tau_c_orig = recent_data['c']
    tau_w_orig = recent_data['w']
    
    
    for loc in range(0,4):
        
#        if not loc == 1:
#            continue
    #Filter (keep relevant data)
    #    loc=2
        indx = indx_fundamental & ~np.isnan(tau_cw_orig)  #& ~np.isnan(tau_c_orig) & ~np.isnan(tau_w_orig)
        indx = indx & (dayflow/35.314666212661 < 819.3876115308075)

#        indx = indx & (D_orig > 7) #& (D_orig > 1)
#        indx = indx & winter \
#        indx = indx & nowave
        indx = indx & (D_orig < shoal_max_depth) &  (D_orig > shoal_min_depth)
        indx = indx & location_indx[loc]

#        continue
        
        if not loc == 4:
            ax, resultstring = plot_stress_curve(tau, indx, loc, locationlabels, resultstring,fig)
        else:
            ax, resultstring = plot_stress_curve(tau, indx, loc, locationlabels, resultstring,fig, dofit=False)
        
        ymin, ymax = ax.get_ylim()
        
        if not loc == 4:
            yalign = (ymax - ymin)*.95 + ymin
            plt.text(1.1*10**-3, yalign, llll[loc] + ' '+ locationlabels[loc],fontsize=axsize, horizontalalignment='left', verticalalignment='top')
        else:
            yalign = (ymax - ymin)*.20 + ymin
            plt.text(1.1*10**-3, yalign, llll[loc] + ' '+locationlabels[loc],fontsize=axsize, horizontalalignment='left', verticalalignment='top')


    #Make plot pretty
    axall.set_xlabel(r'$\tau_{cw}$ (Pa)',fontsize=axsize)
    axall.set_ylabel(r'SSC (mg L$^{-1}$)', fontsize=axsize)
    
    fig.tight_layout()
       
    #Save figure    
    if SAVE_FIGURES:
        plt.savefig(figure_dir + '/location_crit_stress_dayflow.png', dpi=plt_format.dpi)    

#resultstring+='\n'

print(resultstring)
