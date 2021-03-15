
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
import scipy as sp
import scipy.stats
import scipy.io as sio
import scipy.optimize as optimize
from scipy.optimize import leastsq


import matplotlib.pyplot as plt

from plot_formatter import PlotFormatter
plt_format = PlotFormatter(plt)

import os


#%% LOAD CONSTANTS
gm_data_dir = '/Users/jadelson/Dropbox/phdResearch/AllOptical/sfbrspy/grant_madsen_convert'
figure_dir = '/Users/jadelson/Dropbox/phdResearch/AllOptical/sfbrspy/figures/nov2019'

LOAD_FROM_FILE = False
LOAD_SINGLE_STRESS_SET = False
READ_FROM_MEMORY= False
MAKE_SAVE_FILES = False
PLOT_ALL_DATA = False
ANALYZE_CRIT_STRESS = False
ANALYZE_CRIT_STRESS_BY_BIN = False
MAKE_PDF = False
USGS_RS_COMPARE = False
USGS_RS_STATS = False
SAVE_FIGURES = False
MAKE_LOCATION_PDF = False
MAKE_LOCATION_WAVE_PDF = False
MAKE_LOCATION_QQ = False

if __name__ == '__main__':
    MAKE_PDF = False
    MAKE_LOCATION_PDF = False
    MAKE_LOCATION_QQ = True
    MAKE_LOCATION_WAVE_PDF = False
    USGS_RS_COMPARE = False
    USGS_RS_STATS = True
    SAVE_FIGURES = False

density = 1025

plt.close('all')


#%% DEFINE STATIC FUNCTIONS
#Nechad function for RS SPM
def nechad(c,rho):
    spm = c[0]*rho/(1 - rho/0.170991)+c[1]
    return spm



#Sigmoid fit for crit shear stress
def sigmoid(x, x0, C, A, B, k):
    y = A /(B+np.exp(-k*(x-x0))) + C
    return y

def exponential_modified(x,c0,c1,c2, c3, c4):
    return c0*np.power(x,c1) + c3+(x > c2)*(c0*np.power(c2,c1) - c0*np.power(x,c1))
    
def critpoint(x, x0, C, A, B, k):
#    a = np.log(-(-2+np.sqrt(3))*np.exp(x0*k)/B)/k    
    return np.log((0.5*(4*B*np.exp(x0 *k) -  4*np.sqrt(1*np.square(B)*np.exp(2*x0 *k) -  0.25*np.square(B)*np.exp(2*x0 *k))))/np.square(B))/k#Sigmoid jacobian


#Get the mode for fitting    
def get_log_normal_param(s):
    m = np.mean(s)
    sd = np.std(s)
    v = sd*sd
    mu = np.log(m/np.sqrt(1+v/m/m))
    sigma = np.sqrt(np.log(1+v/m/m))
    
    return mu, sigma
    
def get_mode(s):
#    sigma, loc, scale =  sp.stats.lognorm.fit(s)
#    mu = np.log(scale)
    mu, sigma = get_log_normal_param(s)
    mode = np.exp(mu-sigma*sigma)
    return mode

def rayleigh_pdf(x, sigma):
    pdf = x/np.square(sigma)*np.exp(-1*np.square(x)/2/np.square(sigma))
    return pdf
    
def gen_rayleigh_pdf(x, alpha, lamb):
    pdf = 2*alpha*np.square(lamb)*x
    pdf = pdf*np.exp(-1*np.square(lamb*x))
    pdf = pdf*np.power(1 - np.exp(-1*np.square(lamb*x)),alpha-1)
    return pdf

def get_rayleigh_params(x):
    sigma = np.mean(x)/np.sqrt(np.pi/2)
    return sigma
    
def log_normal_pdf(x, mu, sigma):
    return 1/x/sigma/np.sqrt(2*np.pi)*np.exp(-1*np.square(np.log(x)- mu)/(2*np.square(sigma)))
 
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
        
        
stat_functions = {'mode':get_mode, 'median':np.median, 'mean':np.mean, 'std':np.std,'count':len, '25th' :lambda a: np.percentile(a, 25),'75th':lambda a: np.percentile(a, 75),'max':np.max,'min':np.min}
plot_type =  {'mean':'+', 'median':'^', 'mode':'s','std':''}
stat_label_names = {'mean':'Mean', 'median':'Median', 'mode':'Mode', 'std':'Std. Dev.', 'count':'Count', '25th':'25th Percentile','75th':'75th Percentile','min':'Minimum','max':'Maximum'}

fit_func = exponential_modified

#%% LOAD DATA

dataset_name = 'l7_full_dataset.dk'
if (not 'd_load' in locals()) or (not d_load == dataset_name):
    with open(dataset_name,'rb') as f:
        d = pickle.load(f)
        d_load = dataset_name
    
    with open('agg_histogram.dat','rb') as f:
        agg_histogram = pickle.load(f)
    
    with open('all_histogram.dat','rb') as f:
        all_histogram = pickle.load(f)
    #phase
#    with open('tide_phase.dk','rb') as f:
#        tl = pickle.load(f)
    
    #Tau
    tau_orig = d['tau_b']
    
    #SPM
    c = [3.96675038e+02,  -2.77015965e-02]
    c = [4.02979661e+02, 0]
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
    

#%% Establish subsets (if you want to rerun this: subsets_established = False)
if (not 'subsets_established' in locals()) or (not subsets_established):
    subsets_established = True 
    
    #SSC Limit
    indx_fundamental = d['RHOW_661'] < .15
    
    #Locations subssets
    lat1 = 4.176e6
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
    winter = (month > 12 )| (month <= 3)
    summer = (month > 4) & (month < 10) 
    
    seasonlabels = ['Winter','Summer']
    
#%% 
class Digitizer():
    def get_popt(self, indx_fundamental, tau_cw_orig, tau_c_orig, tau_w_orig, nbins = 80, testbintypes = False, min_c = 1):  
        for i in [0]:
            indx = indx_fundamental & ~np.isnan(tau_cw_orig) & ~np.isnan(tau_c_orig) & ~np.isnan(tau_w_orig)
#            indx = indx_fundamental & (D_orig >= 1) & (D_orig < 5)

            local_location = [l[indx] for l in location_indx]
            
            tau_cw = tau_cw_orig[indx]
            tau_c = tau_c_orig[indx]
            tau_w = tau_w_orig[indx]
            ssc = ssc_orig[indx]
            D = D_orig[indx]
            
            
            taus = [tau_cw, tau_c, tau_w]

            self.D = D
            self.xfit = []
#            self.yfit = {}

            
            self.x_plot = []
            self.y_plot = {}
            self.count = []
            self.rouse = {}
            self.c_plot = []
            
            self.popt = {}
            self.pcov = {}
            
            self.err = []
            
            
            for k in stat_functions.keys():
                self.y_plot[k] = []
#                self.yfit[k] = []
                self.popt[k] = []
                self.pcov[k] = []
#                self.err = []
                
#            depth_indx = [(D < 3) & (D > 2), (D > 3) & (D < 5), (D > 5) & (D < 7)]
            depth_indx = [(D < 10)]# & (D > 1.5)]
            for j in range(0,3):   

                err_up = []
                err_down = []
                #Digitize1
                if testbintypes:
                    xdata = taus[0]
#                    xdata = rouse
                else:
#                    xdata = np.sqrt(taus[0])/density
#                    xdata = np.sqrt(taus[j])/density
                    xdata = taus[0]
#                    xdata = rouse

                ydata = ssc
                    
#                min_stress = 2.5
                min_stress = .4

                low_stress = xdata < min_stress
                indx = low_stress &   depth_indx[0]& local_location[j]

                
                xdata = xdata[indx]
                ydata = ydata[indx]

#                bins = np.logspace(-4 ,np.log10(min_stress), nbins)
#                bins = np.array([np.mean(x) for x in np.array_split(np.sort(xdata[xdata>0]),2000)])
                
                if testbintypes:
                    if j == 0:
                        bins = np.array([np.mean(x) for x in np.array_split(np.sort(xdata[xdata>0]),nbins)])
                    elif j == 1:
                        bins = np.linspace(0, min_stress,nbins)
                    elif j == 2:
                        bins = np.logspace(-4 ,np.log10(min_stress), nbins)
                else:
                    bins = np.logspace(-4 ,np.log10(min_stress), nbins)
                    bins = np.logspace(-4 ,np.log10(min_stress), nbins)

                    
                x_filtered = bins
                digitized = np.digitize(xdata, x_filtered, right=True)
                print(sum(indx))
                count =  np.array([np.sum(digitized == i) for i in range(0, len(bins))])


                c_filtered = np.log10(.1 + np.array([np.sum(digitized == i) for i in range(0, len(bins))]))

                x_clean = x_filtered[~np.isnan(x_filtered)&(c_filtered>min_c)]
                c_clean = c_filtered[(c_filtered>min_c)]
                
                xfit = x_clean
                                
                self.x_plot.append(x_clean)
                self.c_plot.append(c_clean)
                self.xfit.append(xfit)
                
                for k in ['mean']:#stat_functions.keys():
                    if k == 'count' or k == 'std':
                        continue
                    y_filtered = np.array([stat_functions[k](ydata[digitized == i]) for i in range(0, len(bins))])
#                    print([sum([digitized == i]) for i in range(0, len(bins))])

                    y_clean = y_filtered[~np.isnan(y_filtered )& (c_filtered>min_c)]
                    self.y_plot[k].append(y_clean)

#                

                err_down = np.array([np.percentile(ydata[digitized == i],25) if c_filtered[i]>min_c else np.nan for i in range(0, len(bins))])
                err_up = np.array([np.percentile(ydata[digitized == i],75) if c_filtered[i]>min_c else np.nan for i in range(0, len(bins))])
                    
                self.err.append([err_down[~np.isnan(err_down)], err_up[~np.isnan(err_up)]])
                self.count.append(count)
            return self.popt['mode'], self.popt['median'], self.popt['mean']

        


#%%
def get_taus(z0, subindex=None):
    filepath = gm_data_dir + "/gm_con_%s_.dat" % str(z0)
    if os.path.isfile(filepath):
        with open(filepath,'rb') as f:
            data_array = pickle.load(f)
        tau_cw_orig = data_array[1][0]
        tau_c_orig = data_array[1][1]
        tau_w_orig = data_array[1][2]
        return {'z0':z0, 'cw':tau_cw_orig, 'c':tau_c_orig, 'w':tau_w_orig, 'indx':indx_fundamental}                     
    else:
        return compute_taus(z0, True, subindex)
    
#%%
def compute_taus(z0, save_file, subindex=None):
    re_ustr_cw =[]# np.zeros(len(indx_fundamental))
    re_ustr_c =[]# np.zeros(len(indx_fundamental))
    re_ustr_w =[]# np.zeros(len(indx_fundamental))
    
    kb = 30*z0
    ustrc_est = np.sqrt(u_orig*u_orig+v_orig*v_orig)*.41/(D_orig*(np.log(D_orig/z0) + z0/D_orig- 1))
    zr = D_orig/2
    u_mag_zr = ustrc_est/.41*np.log(zr/z0)
    u_mag_zr[u_mag_zr<0] = 0
    ub_ = d['ub'].tolist()
    omega_ = d['omega'].tolist()
    phi_c_ = d['phi_c'].tolist()
    
    
    #    u_mag_zr_ = u_mag_zr.tolist()
    #    zr_ = zr.tolist()
    #    ustrc_est_ = ustrc_est.tolist()
    if not subindex is None: 
        indx_to_use = subindex
    else:
        indx_to_use = indx_fundanmental
        
    for k in range(0,len(tau_orig)):
        if indx_to_use[k]:
    
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
        filepath = gm_data_dir + "/gm_con_%s_.dat" % str(k)
        print(filepath)
        data_struct = [z0, [tau_cw_orig, tau_c_orig, tau_w_orig]]
        with open(filepath,'wb') as f:
            pickle.dump(data_struct, f)
    
    return {'z0':z0, 'cw':tau_cw_orig, 'c':tau_c_orig, 'w':tau_w_orig, 'indx':indx_fundamental}                
        

  

#%%                
def get_curves():
    POPT_MODE = []
    POPT_MEDIAN = []
    POPT_MEAN = []
    Z0 = []
    data_filter = Digitizer()
    for z0 in all_data.keys():
#            print(z0)
        data_array = all_data[z0]
#        kb = 30*z0
        tau_cw_orig = data_array[0]
        tau_c_orig = data_array[1]
        tau_w_orig = data_array[2]
        
        popt_mode, popt_median, popt_mean = data_filter.get_popt(indx_fundamental, tau_cw_orig, tau_c_orig, tau_w_orig)
#                data_filter.plot_tri_curve()
        POPT_MODE.append(popt_mode)
        POPT_MEDIAN.append(popt_median)
        POPT_MEAN.append(popt_mean)
        
        Z0.append(z0)


    POPT_MODE = np.array(POPT_MODE)
    POPT_MEDIAN = np.array(POPT_MEDIAN)
    POPT_MEAN = np.array(POPT_MEAN)
    Z0 = np.array(Z0)
    
    POPT_ORIG = {'mean':POPT_MEAN, 'median':POPT_MEDIAN, 'mode':POPT_MODE}
    return Z0, POPT_ORIG
      
#%%                
def get_curves_by_bin(z0= 6.52e-5):
    POPT_MODE = []
    POPT_MEDIAN = []
    POPT_MEAN = []
    BINS = []
    data_filter = Digitizer()
    Z0 = np.array(list(all_data.keys()))

    key = np.argmin(np.abs(z0 - Z0))
    
    for nbins in np.logspace(np.log10(10),np.log10(200),20):

        data_array = all_data[Z0[key]]

        tau_cw_orig = data_array[0]
        tau_c_orig = data_array[1]
        tau_w_orig = data_array[2]
        
        popt_mode, popt_median, popt_mean = data_filter.get_popt(indx_fundamental, tau_cw_orig, tau_c_orig, tau_w_orig, nbins=nbins, testbintypes=True)
#                data_filter.plot_tri_curve()
        POPT_MODE.append(popt_mode)
        POPT_MEDIAN.append(popt_median)
        POPT_MEAN.append(popt_mean)
        
        BINS.append(nbins)
        
    POPT_MODE = np.array(POPT_MODE)
    POPT_MEDIAN = np.array(POPT_MEDIAN)
    POPT_MEAN = np.array(POPT_MEAN)
    BINS = np.array(BINS)
    
    POPT_ORIG = {'mean':POPT_MEAN, 'median':POPT_MEDIAN, 'mode':POPT_MODE}
    return BINS, POPT_ORIG
      

print('Script starting') 
#%%    
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
            if z0 > .01 or z0 < 1e-5:
                continue

            
            with open(gm_data_dir + '/' + filename,'rb') as f:
                data_array = pickle.load(f)
            z0 = data_array[0]
            Z0.append(z0)
            all_data[z0] = data_array[1]
    Z0 = np.sort(np.array(Z0))

#%%
if LOAD_SINGLE_STRESS_SET:
    z0 = 3.61*(10**-5)
    
    try:
        recent_data
    except NameError:
        print('exception')
        recent_data = {'z0':None,'indx':None}

    if recent_data == None or not ((recent_data['z0'] == z0) and all(recent_data['indx'] == indx_fundamental)):
        recent_data = get_taus(z0)
            
#%%                
if READ_FROM_MEMORY:
    Z0, POPT_ORIG = get_curves()
#    BINS, POPT_ORIG_BIN = get_curves_by_bin()

      
#%%        
if MAKE_SAVE_FILES:
    import multiprocessing
    print('this may take a while...')
    done = [float(x.split('_')[2]) for x in os.listdir(gm_data_dir) if 'gm_con' in x]

    Z0 = np.logspace(-6,-3,300)
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

    
    
#%% Plotting tau_crit vs z0
if ANALYZE_CRIT_STRESS:
    fig_crit = plt.figure()
    plt.rc('text', usetex=True)
    plt.rc('font', family='Times New Roman')
    
    sort_indx = sorted(range(len(Z0)),key=lambda x:Z0[x])
#    sort_indx = sort_indx[-1*POPT[:,i,0]<1]
    plot_indx = sort_indx[::1]
    
    
    
    power_law = lambda c,x: c[0]*np.power(x,c[1])+c[2]
    power_error = lambda c,x,y: power_law(c,x) - y
    get_yeq1_intercept = lambda c: np.power((.1-c[2])/c[0],1/c[1])
   
    plt.semilogx(np.logspace(-8,1,100),np.logspace(-8,1,100)*0+.1,'k--',alpha=1, linewidth=.5)        
    colors = ['k','r','g']
    for i in range(0,1):
        color = colors[i]
#        plot_type =  {'mean':color+'+', 'median':color+'^', 'mode':color+'s'}
        for k in POPT_ORIG.keys():
            POPT = np.array(POPT_ORIG[k])
            tau_crit = -1*POPT[:,i,0]
            _Z0 = Z0[tau_crit < 1]
            tau_crit = tau_crit[tau_crit < 1]
             
            sort_indx = sorted(range(len(_Z0)),key=lambda x:_Z0[x])
            plot_indx = sort_indx[::8]
    
            c, success = leastsq(power_error, [1,2,tau_crit[0]], args=(_Z0, tau_crit))
            interc = get_yeq1_intercept(c)

            plt.semilogx(_Z0[plot_indx],tau_crit[plot_indx],'k'+ plot_type[k], fillstyle='none',label=stat_label_names[k])
            plt.semilogx(_Z0[sort_indx], power_law(c,_Z0[sort_indx]),color+'-', linewidth=.8)
    
            print(stat_label_names[k] + ':\t& $' + format(interc, '.2e').replace("e-0", "e-").replace('e',' \\times 10^{') + '}$ \\\\')
            
        plt.xlim(min(Z0),max(Z0))
        plt.legend(title='Filtering Method', fontsize=plt_format.fontsize)
        plt.xlabel(r'$z_0$ [m]', fontsize=plt_format.fontsize)
        plt.ylabel(r'$\tau_{\textrm{sig}}$ [kg m$^{-1}$ s$^{-2}$] ', fontsize=plt_format.fontsize)
        fig_crit.tight_layout()
        if SAVE_FIGURES:
            plt.savefig(figure_dir + '/critical_vs_z0.png',dpi=plt_format.paper_dpi)
            

#%% Plotting ssc vs tau_crit

if ANALYZE_CRIT_STRESS_BY_BIN:
        
    fig_crit = plt.figure()
    plt.rc('text', usetex=True)
    plt.rc('font', family='Times New Roman')
    
#    sort_indx = sorted(range(len(BINS)),key=lambda x:BINS[x])
#    plot_indx = sort_indx[::2]
    
    
    power_law = lambda c,x: c[0]*np.power(x,c[1])+c[2]
    power_error = lambda c,x,y: power_law(c,x) - y
    get_yeq1_intercept = lambda c: np.power((.1-c[2])/c[0],1/c[1])
   
    bin_plot = ['o','+','s','>']
    
    z0list = [10**-5,10**-4,10**-3]
    colors = ['r','b','k']
    color_labels = ['Equal Frequency', 'Linear Spacing', 'Logrithmic Spacing']
    shapes = ['o','+','s']
    shape_labels = [r'$z_0 = 10^{-5}$',r'$z_0 = 10^{-4}$',r'$z_0 = 10^{-3}$']

    def get_binning_data():
        BINS_ALL = []
        POPT_BIN_ALL = []   
        for z0 in z0list:
            BINS, POPT_ORIG_BIN = get_curves_by_bin(z0)
            BINS_ALL.append(BINS)
            POPT_BIN_ALL.append(POPT_ORIG_BIN)
        return BINS_ALL, POPT_BIN_ALL
    
    BINS_ALL, POPT_BIN_ALL = get_binning_data()
    
    for BINS, POPT_ORIG_BIN, z0, bin_plot, bin_label in zip(BINS_ALL, POPT_BIN_ALL, z0list,shapes,shape_labels):
        POPT = np.array(POPT_ORIG_BIN['mean'])
        for j in [2]:

            
            tau_crit = -1*POPT[:,j,0]
#            c, success = leastsq(power_error, [1,2,tau_crit[0]], args=(BINS, tau_crit))
#            interc = get_yeq1_intercept(c)
            
            
            
            plt.semilogx(BINS,tau_crit,colors[j]+bin_plot, fillstyle='none',label=bin_label)
    #            plt.plot(BINS[sort_indx], power_law(c,BINS[sort_indx]),color+'-', linewidth=.8)
    
    #        print(stat_label_names[k] + ':\t& $' + format(interc, '.2e').replace("e-0", "e-").replace('e',' \\times 10^{') + '}$ \\\\')
            
    plt.xlim(9,250)
    plt.legend( fontsize=plt_format.fontsize)
    plt.xlabel(r'$N$ Bins', fontsize=plt_format.fontsize)
    plt.ylabel(r'$\tau_{\textrm{crit}}$ [kg m$^{-1}$ s$^{-2}$] ', fontsize=plt_format.fontsize)
    fig_crit.tight_layout()
    if SAVE_FIGURES:
        plt.savefig(figure_dir + '/critical_vs_bins.png',dpi=plt_format.paper_dpi)
            

 #%%  
 
if MAKE_PDF:
    
    def getgeneric(x, sample, pdftype, returnll=False):
        param = pdftype.fit(sample,loc=0)
        yhat = pdftype.pdf(x,*param)
        if returnll:
            return yhat, pdftype.logpdf(sample,*param).sum(), param
        return yhat

    def getfittedpdf(x,sample,returnll=False):    
        from scipy.stats import mielke as pdftype
        return getgeneric(x, sample, pdftype, returnll)

    histlabels = locationlabels
    histlabels = ['Remotely Sensed SSC']
    shoallabels = ['Channels', 'Shoals']
    histlabels = shoallabels
    titlelabels = seasonlabels
    fig_hist = plt.figure(figsize=(6,4))
    p1 = []
    p2 = []
    colors = ['k','b','r','g']
    patch_leg = []
    label_leg = []
    indx1 = [(ssc_orig < 200) & (D_orig > 1)] #[(D_orig > 10), (D_orig <= 5)]
#    indx1 = indx1 & winter & location_indx[0]
    linetype = ['k-']
    for i in range(1):
        plt.rc('text', usetex=True)
        plt.rc('font', family='Times New Roman')
        
        indx = indx_fundamental & indx1[i] # & ~location_indx[1] 
        ssc = ssc_orig[indx]
        
        xfit = np.linspace(0,200,1000)
        

        bins = np.linspace(0,102,100)
        pdf, edge = np.histogram(ssc, bins, density=True)
        plt.plot(bins[0:-1],pdf,linetype[i],linewidth=plt_format.thin_linewidth)
        
#        plt.xlim([0,80])
#        plt.ylim([0,.04])
        ax = plt.gca()
        ymin, ymax = ax.get_ylim()
        yalign = (ymax - ymin)*.95 + ymin
        

        from scipy.stats import mielke, lognorm
        ci=0
        for pdftype in [mielke]:#[lognorm]:#, mielke]:

            yfit, ll, param = getgeneric(xfit, ssc, pdftype, returnll=True)
            print(ll)
            p2, = plt.plot(xfit, yfit, 'k--', linewidth=plt_format.thin_linewidth)
            ci+=1
            
            def plot_point(x, shape):
                plt.plot(x,pdftype.pdf(x,*param), shape, markersize=15, fillstyle='none',label=stat_label_names[k])
            
            intervals = pdftype.interval(.5,*param)
            stat_values = {'mode':xfit[np.argmax(yfit)], 'median':pdftype.median(*param), 'mean':pdftype.mean(*param), 'std':pdftype.std(*param),'count':len(ssc), '25th' : intervals[0],'75th': intervals[1]}
    
#            for k in stat_values.keys():
#                print(stat_label_names[k] + ':\t& ' + str(stat_values[k]) )
#                if not k in ['std','count','25th','75th']:
#                    plot_point(stat_values[k], 'k'+plot_type[k])
#            
            plt.plot(np.mean(ssc)*np.ones((1000)),np.linspace(0,1,1000),'k:', linewidth=plt_format.thin_linewidth)
            plt.plot(np.median(ssc)*np.ones((1000)),np.linspace(0,1,1000),'k-.', linewidth=plt_format.thin_linewidth)
            
 
    plt.xlim([0,100])
    plt.ylim([0,.05])
    
    order = [0,2,1,3]
#    handles, legend_labels = plt.gca().get_legend_handles_labels()

#    plt.legend([handles[idx] for idx in order],[legend_labels[idx] for idx in order], fontsize=plt_format.fontsize)
#    plt.legend(patch_leg, label_leg,fontsize=plt_format.fontsize)
    plt.xlabel(plt_format.ssc_label,fontsize=plt_format.fontsize)
    plt.ylabel(r'Probability density', fontsize=plt_format.fontsize)
    fig_hist.tight_layout()
    if SAVE_FIGURES:
        plt.savefig(figure_dir + '/all_ssc_pdf.png',dpi=plt_format.paper_dpi)

#%%
if MAKE_LOCATION_PDF:
    fig_hist = plt.figure(figsize=(6,5))
    
# Location hist
        
    histlabels = locationlabels

#    fig_loc_hist = plt.figure(figsize=(9,6))
#    plt.subplot(111)
    axall = plt.gca()
    axall.tick_params(labelcolor='k', top='off', bottom='off', left='off', right='off')
#    for spine in plt.gca().spines.values():
#        spine.set_visible(False)


    linetype = ['k-','k--','k:','k-.']
    llll = ['a','b','c','d','e','f','g','h','i','j','k','l']

    patch_leg = []
    label_leg = []
    indx1 = location_indx

    for i in range(len(indx1)):
        indx2 = [summer, winter]
        indx3 = [D_orig > 7, (D_orig < 3) & (D_orig > .5)]
        if i == 3:
            indx3 = [D_orig > 7,  (D_orig < 5) & (D_orig > .5)]
        for j in range(len(indx2)):
            ax = plt.subplot(4,1,i+1)
            for k in range(len(indx3)):
                plt.rc('text', usetex=True)
                plt.rc('font', family='Times New Roman')
                indx = indx_fundamental    & indx1[i] & indx2[j] &indx3[k]
                ssc = ssc_orig[indx]
        
                bins = (np.linspace(0,40,40)**2)/40**2*200
#                bins = np.linspace(0,200,200)
                pdf, edge = np.histogram(ssc, bins, density=True)
                plt.plot(bins[0:-1],pdf,linetype[k+j*2],linewidth=1)

                ax = plt.gca()
                ymin, ymax = ax.get_ylim()
                yalign = (ymax - ymin)*.95 + ymin
#        plt.text(95, yalign, locationlabels[i],fontsize=plt_format.fontsize, horizontalalignment='right', verticalalignment='top')
           
        top = .05
        right = 80
        if i == 1:
            top = .1
            right = 50
        if i == 2:
            right = 100
        if i == 3:
            right = 160
#        top = 1
            
        plt.xlim([0,right])
        plt.ylim([0,top])
        ax.text(78/80.*right, .065/.075*top, '(' + llll[i] + ') ' + histlabels[i],fontsize=plt_format.fontsize, horizontalalignment='right', verticalalignment='top')

    fig_hist.tight_layout()
    fig_hist.text(0.02, 0.5, r'Probability density', fontsize=plt_format.fontsize, va='center', rotation='vertical')
    plt.xlabel(plt_format.ssc_label,fontsize=plt_format.fontsize)
    fig_hist.subplots_adjust(bottom=0.12)
    fig_hist.subplots_adjust(left=0.13)

    if SAVE_FIGURES:
        plt.savefig(figure_dir + '/location_pdf.png',dpi=plt_format.paper_dpi)

#%%
if MAKE_LOCATION_QQ:
    fig_qq = plt.figure(figsize=(6,4))
    
    from scipy.stats import lognorm
# Location hist
        
    histlabels = locationlabels

    indx1 = location_indx

    indx2 = (D_orig < 5) & (D_orig > .5)


    plt.rc('text', usetex=True)
    plt.rc('font', family='Times New Roman')
    indx = indx_fundamental & indx2 # & indx1[0] & indx2
    ssc = ssc_orig[indx]
        
    dist = lognorm.fit(ssc)
    
    q1 = np.array([np.percentile(ssc,i) for i in range(1,100)])
    q2 = lognorm.ppf(np.linspace(.01,.99,99),*dist)
    
    ax = fig_qq.subplots(1, 1)
    ax.plot(q1, q2 , 'ko', alpha=.7)
    ax.plot(np.linspace(0,100), np.linspace(0,100), 'k-', alpha=.6)
    ax.set_xlim([0,100])
    ax.set_ylim([0,100])
    ax.set_xlabel('Remotely sensed SSC (mg L$^{-1}$)')
    ax.set_ylabel('Lognormal distribution (mg L$^{-1}$)')
    ax.set_aspect('equal')
    
    if SAVE_FIGURES:
        fig_qq.savefig(figure_dir + '/south_bay_qq.png',dpi=plt_format.paper_dpi)


#%%
        
if MAKE_LOCATION_WAVE_PDF:
    fig_hist = plt.figure(figsize=(6,3))
    
# Location hist
        
    histlabels = locationlabels

#    fig_loc_hist = plt.figure(figsize=(9,6))
#    plt.subplot(111)
    axall = plt.gca()
    axall.tick_params(labelcolor='k', top='off', bottom='off', left='off', right='off')
#    for spine in plt.gca().spines.values():
#        spine.set_visible(False)


    linetype = ['k-','k--','k:','k-.']
    llll = ['a','b','c','d','e','f','g','h','i','j','k','l']

    patch_leg = []
    label_leg = []
    indx1 = location_indx
    pltnum = 1
    for i in range(len(indx1)):
        if i == 1 or i == 3:
            continue
        indx2 = [(Hs_orig <= .1), (Hs_orig > 0.2)]
        indx3 = [D_orig > 7, (D_orig < 3) & (D_orig > .5)]
        for j in range(len(indx2)):
            ax = plt.subplot(2,1,pltnum)
            
            for k in range(len(indx3)):
                plt.rc('text', usetex=True)
                plt.rc('font', family='Times New Roman')
                indx = indx_fundamental    & indx1[i] & indx2[j] &indx3[k]
                ssc = ssc_orig[indx]
        
                bins = (np.linspace(0,40,40)**2)/40**2*200
#                bins = np.linspace(0,200,200)
                pdf, edge = np.histogram(ssc, bins, density=True)
                plt.plot(bins[0:-1],pdf,linetype[k+j*2],linewidth=1)
                
                

                ax = plt.gca()
                ymin, ymax = ax.get_ylim()
                yalign = (ymax - ymin)*.95 + ymin
#        plt.text(95, yalign, locationlabels[i],fontsize=plt_format.fontsize, horizontalalignment='right', verticalalignment='top')
        pltnum += 1   
        top = .05
        right = 80
        if i == 1:
            top = .1
            right = 50
        if i == 2:
            right = 100
        if i == 3:
            right = 160
#        top = 1
            
        plt.xlim([0,right])
        plt.ylim([0,top])
        ax.text(78/80.*right, .065/.075*top, '(' + llll[pltnum-2] + ') ' + histlabels[i],fontsize=plt_format.fontsize, horizontalalignment='right', verticalalignment='top')

    fig_hist.tight_layout()
    fig_hist.text(0.02, 0.5, r'Probability density', fontsize=plt_format.fontsize, va='center', rotation='vertical')
    plt.xlabel(plt_format.ssc_label,fontsize=plt_format.fontsize)
    fig_hist.subplots_adjust(bottom=0.15)
    fig_hist.subplots_adjust(left=0.13)

    if SAVE_FIGURES:
        plt.savefig(figure_dir + '/location_wave_pdf.png',dpi=plt_format.paper_dpi)
        

#%%
colors = ['r','b']
if USGS_RS_COMPARE:
    linetype = ['k-','k--','k:','k-.']
    llll = ['a','b','c','d','e','f','g','h','i','j','k','l']
    import pandas as pd
    locs = [36, 25, 15, 9, 3]
    histlabels1 = locationlabels
    histlabels2 = ['Remotely Sensed', 'In Situ']
    histlabels3 = ['RS', 'In Situ']
#    histlabels = np.empty((4,2),dtype=str)
#    for i in range(4):
#        for j in range(2):
#            histlabels[i,j] = 
    filename = '/Users/jadelson/Dropbox/phdResearch/AllOptical/polaris/rousecopy.csv'
    dataset = pd.read_csv(filename)
    fig_polaris_pdfs =plt.figure(figsize=(6,5))

    stats = {'Remotely Sensed':{}, 'In Situ':{}}
    
    
    
    for i in range(4):
        indx = ~np.isnan(dataset['Calculated_SPM']) & (dataset['Depth'] <= 2) 
        indx = indx & (dataset['Station_Number'] <= locs[i]) & (dataset['Station_Number'] > locs[i+1]) 
        indx = indx & (dataset['Days_since_01011990'] > 4000)
        indx = indx & (dataset['Calculated_SPM'] < 100)
        spm = dataset['Calculated_SPM'][indx]
        indx = indx_fundamental & (D_orig > 10) & location_indx[i] & (D_orig < 50) #& (Hs_orig < .3)
        ssc2 = ssc_orig[indx]

        ax = plt.subplot(4,1,i+1)
        patch_leg = []
        label_leg = []
        
        ymax = .05
        
        from scipy.stats import ttest_ind as ttest
        print(ttest(spm,ssc2),np.median(ssc2),np.median(spm))
        for j in range(2):

            if j == 0:
                ssc = ssc2
            else:
                ssc = spm      
            
            xfit = np.linspace(0,200,1000)

            bins = (np.linspace(0,40,40)**2)/40**2*200

            pdf, edge = np.histogram(ssc, bins, density=True)
            plt.plot(bins[0:-1],pdf,linetype[j],linewidth=plt_format.thin_linewidth)

        
        top = .06
        right = 80
        if i == 1:
            top = .08
            
        plt.xlim([0,right])
        plt.ylim([0,top])
        ax.text(78/80.*right, .065/.075*top, '(' + llll[i] + ') ' + histlabels1[i],fontsize=plt_format.fontsize, horizontalalignment='right', verticalalignment='top')

    fig_polaris_pdfs.tight_layout()
    fig_polaris_pdfs.text(0.02, 0.5, r'Probability density', fontsize=plt_format.fontsize, va='center', rotation='vertical')
    plt.xlabel(plt_format.ssc_label,fontsize=plt_format.fontsize)
    fig_polaris_pdfs.subplots_adjust(bottom=0.10)
    fig_polaris_pdfs.subplots_adjust(left=0.12)
    
    if SAVE_FIGURES:
        plt.savefig(figure_dir + '/USGS_comparison_bylocation.png',dpi=plt_format.paper_dpi)


#%%
if USGS_RS_STATS:
    import pandas as pd
    locs = [36, 25, 15, 9, 3]

    filename = '/Users/jadelson/Dropbox/phdResearch/AllOptical/polaris/rousecopy.csv'
    dataset = pd.read_csv(filename)        
       
    indx_usgs = ~np.isnan(dataset['Calculated_SPM'])&  (dataset['Station_Number'] <= locs[0]) &  (dataset['Station_Number'] > 3) 
    indx_usgs = indx_usgs & ((dataset['Station_Number'] > locs[1]) | (dataset['Station_Number'] <= locs[2]) )
    indx_usgs = indx_usgs & (dataset['Depth'] <= 2) 
    indx_usgs = indx_usgs & (dataset['Days_since_01011990'] > 6000)
    
    spm_is = dataset['Calculated_SPM'][indx_usgs]
    indx3 =  indx_fundamental & ~location_indx[1] & (D_orig > 10)
    indx_rs = [indx_fundamental , indx_fundamental & (D_orig < 5) & ~location_indx[1] ,indx3]
    spm_rs = []
    print('& RS SF Bay & RS Shoals & RS Channels & In Situ \\\\')
    for indx in indx_rs:
        spm_rs.append(ssc_orig[indx])
        
    mus = []
    sigmas = []
    for spm in spm_rs:
        
        mu,sigma = get_log_normal_param(spm)
        mus.append(mu)
        sigmas.append(sigma)
        
    mu,sigma = get_log_normal_param(spm_is)
    mus.append(mu)
    sigmas.append(sigma)
    

    def plot_point(x, shape):
        plt.plot(x,log_normal_pdf(x, mu, sigma), shape, markersize=10, fillstyle='none',label=stat_label_names[k])
    
    
    for k in stat_functions.keys():
#             plot_point(stat_functions[k](ssc), plot_type[k])
        string = stat_label_names[k] + ' & '
        for spm in spm_rs:
            string = string +  '%.2f & ' % stat_functions[k](spm)
        string = string + '%.2f \\\\' % stat_functions[k](spm_is)
        print(string  )

    print(scipy.stats.ks_2samp(spm,spm_is))
#    string = '$\\sigma$ '
#    for sigma in sigmas:
#        string = string + '& %.2f ' % sigma
#    print(string + '\\\\')
#
#    string = '$\\mu$ '
#    for mu in mus:
#        string = string + '& %.2f ' % mu
#    print(string + '\\\\')
#

    