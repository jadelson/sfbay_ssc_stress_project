import pandas as pd
from scipy.optimize import minimize
from scipy.stats import variation as cv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


from plot_formatter import PlotFormatter


plt_format = PlotFormatter(plt)
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor


figure_dir = 'figures'


def my_plot_predictor(y,y_hat,title=None, savename=None):
    fig,ax = plt.subplots()
    ax.plot(y_hat,y,'o',y_hat,y_hat,'b-')
    ax.set_xlabel(r'Predicted $\hat{y}$')
    ax.set_ylabel(r'Measured $y$')
    ax.get_xaxis().get_major_formatter().set_useOffset(False)
    ax.set_title(title)
    if savename != None:
        fig.save(savename)

def nechad(c,rho):
    denom = c[2]
    denom = .171
    spm = c[0]*rho/(1 - rho/denom)+c[1]
    return spm

def nechad3(c,rho):
    denom = c[2]
#    denom = .21
    spm = c[0]*rho/(1 - rho/denom)+c[1]
    return spm

def nechadfull(c,rho):
    denom = c[2]
    spm = c[0]*rho/(1 - rho/denom)+c[1]
    return spm

def huber_norm(x,delta):
    y = np.power(x[np.abs(x) <= delta],2)/2/delta
    z = np.abs(x[np.abs(x) > delta]) - delta/2
    return sum(y) + sum(z)
    
def multilinear_func(c,X):
    return np.dot(X,c[0:-1])+c[-1]

def worker_func_nechad(X, y, n):
    rho = X[:,n]
    def nechad_min(c):
        return huber_norm(y-nechad(c,rho), 10)
        
    min_value = minimize(nechad_min,[1,1,.178], bounds=((300,500),(-10,10),(0.16,0.18)), options={'disp': False})
    c = min_value.x
#    y_hat = nechad(c,rho)
    return c
 
def worker_func_nechad3(X, y, n):
    rho = X[:,n]
    def nechad_min(c):
        return huber_norm(y-nechad3(c,rho), 10)
        
    min_value = minimize(nechad_min,[1,0,.21], bounds=((1500,2500),(-2,2),(0.19,0.23)), options={'disp': False})
    c = min_value.x
#    y_hat = nechad(c,rho)
    return c
        

def worker_func_nechad2(X, y, n):
    rho = X[:,n]
    def nechad_min(c):
        return np.linalg.norm(y-nechad(c,rho), np.inf)
        
    min_value = minimize(nechad_min,[1,0,.17], bounds=((300,500),(-2,2),(0.14,0.18)), options={'disp': False})
    c = min_value.x
#    y_hat = nechad(c,rho)
    return c
    
def worker_func_nechad4(X, y, n):
    rho = X[:,n]
    def nechad_min(c):
        return np.linalg.norm(y-nechad3(c,rho), 2)
        
    min_value = minimize(nechad_min,[1,0,.21], bounds=((1500,2500),(-2,2),(0.19,0.23)), options={'disp': False})
    c = min_value.x
#    y_hat = nechad(c,rho)
    return c

def mse(err):
    return np.round(np.sqrt(np.sum(np.square(err)))/len(err),2)

def mae(err):
    return np.round(np.mean(np.abs(err)),2)

def worker_func_multiliniear2(X, y):
    m,n = X.shape
    def multiliniear_min(c):
        return np.linalg.norm(y-multilinear_func(c,X) , 2) 
        
    min_value = minimize(multiliniear_min,[np.ones((n+1,))], options={'disp': False})
    c_hat = min_value.x
    
    return c_hat

def worker_func_multiliniear(X, y):
    m,n = X.shape
    def multiliniear_min(c):
        return huber_norm(y-multilinear_func(c,X) , 10) 
        
    min_value = minimize(multiliniear_min,[np.ones((n+1,))], options={'disp': False})
    c_hat = min_value.x
    
    return c_hat


def power_c(c,x):
    return c[1]*np.exp(x*c[0])

def worker_func_power(x,y):
    def power_min(c):
        return np.linalg.norm(y - power_c(c,x))
    min_value = minimize(power_min,[0.29525658, 1.72556876])
    return min_value.x

def worker_func_power2(x,y):
    def power_min(c):
        return huber_norm(y - power_c(c,x), 10) #np.linalg.norm(y - power_c(c,x))
    min_value = minimize(power_min,[0.29525658, 1.72556876])
    return min_value.x
    
def worker_func_ransac(X,y):
    ransac = linear_model.RANSACRegressor()
    ransac.fit(X, y)    
    
# Function for creating a nechad curve
def make_example_nechad_plot(c,n):
    x = np.linspace(0,0.140991,10000)
    y = nechad(c,x)
    
    if sat == '7':
        landsatnames = ['479','561','661','835'] #LANDSAT 7
        channelnames = [1,2,3,4]
    elif sat == '8':
        landsatnames = ['FLAGS','443','483','561','655','865'] #LANDSAT 8
        channelnames = [0,1,2,3,4,5]
        
    wl = landsatnames[n]
    ch = channelnames[n]
    #    c = np.round(c,2)
    c1 = '%.2f' % c[1]
    if c[1] > 0:
        c1 = '+ ' + c1
    elif c[1] == 0:
        c1 = ''
        
    plt.figure(num=None, figsize=(12, nfolds), dpi=300, facecolor='w')
    plt.plot(x,y,'k')
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.title(r'$SPM$ Inference Function (Nechad et al. 2010)',fontsize=11)
    plt.xlabel(r'Landsat 7 Channel %d Reflectance, $\displaystyle\rho$, at $\lambda$ = %s nm' % (ch,wl),fontsize=11)
    plt.ylabel(r'$SPM$ (mgL$\displaystyle^{-1}$)',fontsize=11)
    plt.text(.003, 200, r'$\displaystyle SPM =  \frac{%.2f \rho}{1 - \rho/0.170991}%s$'% (c[0],c1), fontsize=11)
    
# Data is in data_landsat8_xxminute.csv files in the landsat_spm_data directory
# main calls different worker functions
def main():
    pass

nfolds = 4
plot = False
plt.close('all')


final_string = ''
alt_string = ''

if True:
    save_directory = 'landsat_spm_data/'
    sat= '7'
    simple_channel = 2
    channel = 2
    channel2 = 3
    
    model_indx = [0,2,9,11,12]
    model_indx = [0,1,2,3,4,5,6,7,8,9,11,12]
    MINUTES = ['15','30','60','120']
        
    llll = ['a','b','c','d','e','f','g','h','i','j','k','l']
    lll = ['i','ii','iii','iv','v','vi']
    
    PLOT_60_MIN = True
    PLOT_KFOLDS = False
    
#    rs = 1239
    kf = KFold(n_splits=nfolds,shuffle=True)

    if sat == '7':
        LANDSAT_NAMES = ['RHOW_479','RHOW_561','RHOW_661','RHOW_835'] #LANDSAT 7
        plot_names = ['479','561','661','835']
    elif sat == '8':
        LANDSAT_NAMES = ['FLAGS','RHOW_443','RHOW_483','RHOW_561','RHOW_655','RHOW_865'] #LANDSAT 8

    ynames = ['Tabulated Nechad', 'Nechad LSR', 'Nechad Huber', 'Nechad NIR LSR', 'Nechad NIR Huber', 'ML LSR', 'ML Huber', 'Linear Red', 'Linear NIR', 'Log Red', 'Power Red', 'Power Red Huber', 'Neural Network']
    ynames2 = ['TabulatedNechad', 'NechadLSR', 'NechadHuber', 'NechadNIRLSR', 'NechadNIRHuber', 'MLLSR', 'MLHuber', 'LinearRed', 'LinearNIR', 'LogRed', 'PowerRed', 'NeuralNetwork']
    
    text_names1_orig = ['N10 Red', 'N10 Red', 'N10 Red', 'N10 NIR', 'N10 NIR', 'MultiLin.', 'MultiLin.', 'Lin. Red', 'Lin. NIR', 'R01 Red', 'Exp. Red', 'Exp. Red', 'Neural']
    text_names2_orig = ['Tabulated', 'LSR', 'Huber', 'LSR', 'Huber', 'LSR', 'Huber', 'Huber', 'Huber', 'LSR', 'LSR', 'Huber', 'Network']
    
        
    for minutes in MINUTES:
        
        save_filename = 'data_landsat'+sat+'_'+minutes+'minute.csv'

        dataset = pd.read_csv(save_directory + save_filename, index_col=0)
        
        X = dataset.loc[:,LANDSAT_NAMES].values
        m,n = X.shape
        n2 = n*(1+(n-1))

        X_wratio = np.zeros((m,n2))
        X_wratio[:,0:n] = X
        k = n
        for i in range(0,n):
      
            for j in range(0,n):
                if i == j:
                    continue
                else:
                    X_wratio[:,k] = X[:,i]/X[:,j]
                    k += 1
        
#        n2 = 4   
        X = X_wratio           
        y = dataset.loc[:,'Calculated SPM'].values
        
        m = len(y)
    
        choice = np.random.choice(m,np.round(m))
        indx = np.asarray(m*[True])
        indx[choice] = False
    
        

        fold = 0
        n_models = len(model_indx)
        figs = [None]*n_models
        axs = [None]*n_models
        errors = np.zeros((nfolds,n_models))
        bias_errors = np.zeros((nfolds,n_models))
        weights = np.zeros((nfolds,n_models))
        pws = np.zeros((nfolds,n_models))
        

        fold_str = ''
        fold_str_mae = ''
        fold_str_bias = ''
        fold_str_cv = ''
        fold_str_pw = ''
        
        for train, test in kf.split(X):
            
            c_n_s = [328.17,1.68,.171]
            c_n_h_k = worker_func_nechad(X[train,:], y[train], channel)
            c_n_l_k =  worker_func_nechad2(X[train,:], y[train], channel)
            
            c_nir_h_k = worker_func_nechad3(X[train,:], y[train], channel2)
            c_nir_l_k = worker_func_nechad4(X[train,:], y[train], channel2)
            

            
            c_s_k = worker_func_multiliniear(np.array([X[train,simple_channel]]).transpose(), y[train])
            c_snir_k = worker_func_multiliniear(np.array([X[train,channel2]]).transpose(), y[train])
            c_sl_k = worker_func_multiliniear2(np.array([X[train,simple_channel]]).transpose(), np.log(y[train]) )
            
            c_p_l_k = worker_func_power(X[train,channel], y[train])
            c_p_h_k = worker_func_power2(X[train,channel], y[train])
            
            yhat_n_s = nechadfull(c_n_s,X[:,channel])
            yhat_n_l = nechad(c_n_l_k, X[:,channel]) #nechadhuber
            yhat_n_h = nechad(c_n_h_k, X[:,channel]) #nechadnorm

            yhat_s_l = multilinear_func(c_s_k,np.array([X[:,simple_channel]]).transpose()) #simple
            yhat_snir_l =  multilinear_func(c_snir_k,np.array([X[:,channel2]]).transpose()) #simple
            yhat_sl_l = np.exp(multilinear_func(c_sl_k,np.array([X[:,simple_channel]]).transpose()))
            
            
            yhat_p_l = power_c(c_p_l_k,X[:,simple_channel])
            yhat_p_h = power_c(c_p_h_k,X[:,simple_channel])
            
            yhat_nnir_h = nechad3(c_nir_h_k, X[:,channel2])
            yhat_nnir_l = nechad3(c_nir_l_k, X[:,channel2])
            
            yhatnfolds = np.exp(multilinear_func(c_sl_k,np.array([X[:,simple_channel]]).transpose())) #simplelog
            
            scaler = StandardScaler()
             
            X2 = X[:,:]
            
            scaler.fit(X2[train,:])  
            X2 = scaler.transform(X2)
#            u,s,v = np.linalg.svd(X2)
#            X2 = u[:,:4]
            
            c_m_k = worker_func_multiliniear2(X2[train,:], y[train] )
            c_mh_k = worker_func_multiliniear(X2[train,:], y[train])
            
            yhat_m_l = multilinear_func(c_m_k,X2) #multilinearnorm
            yhat_m_h = multilinear_func(c_mh_k,X2) #multilinearhuber

            X2 = X[:,:4]
            
            scaler.fit(X2[train,:])  
            X2 = scaler.transform(X2)
                    
            clf = MLPRegressor(solver='lbfgs',hidden_layer_sizes=(5,),alpha=1e-5,random_state=1)
            clf.fit(X2[train,:],y[train])
            yhat_net = clf.predict(X2)


            mae_print = lambda yhat : np.round(np.mean(np.abs(y[test]-yhat[test])),2)
            bias_print = lambda yhat : np.round(np.mean(y[test]-yhat[test]),2)
            cv_print = lambda yhat : np.round(cv(y[test]-yhat[test]),2)
#                    mae_print = lambda yhat : np.round(np.sqrt(np.sum(np.square(y[test]-yhat[test]))),2)
            mae_print = lambda yhat : np.round(r2_score(y[test], yhat[test]),2)
 
            yhats = [yhat_n_s, yhat_n_l, yhat_n_h, yhat_nnir_l, yhat_nnir_h, yhat_m_l, yhat_m_h, yhat_s_l, yhat_snir_l, yhat_sl_l, yhat_p_l, yhat_p_h, yhat_net]

            yhats = [yhats[iiii] for iiii in model_indx]
            text_names1 = [text_names1_orig[iiii] for iiii in model_indx]
            text_names2 = [text_names2_orig[iiii] for iiii in model_indx]

            percent_wins = []
            for ii in range(len(yhats)):
                wins = 0
                compares = 0
                for jj in range(len(yhats)):
                    if ii == jj:
                        continue
                    for kk in test:
                        compares += 1
                        if np.abs(yhats[ii][kk] - y[kk]) <= np.abs(yhats[jj][kk] - y[kk]):
                            wins+=1
                percent_wins.append(np.round(wins*100./compares,1))

                    
            if fold == 0:
                
                strr = ''
                for yn in text_names1:
                    strr += ' & \\textbf{' + yn + '}'
                fold_str += strr + '\\\\\n'

                strr = '\\textbf{K-folds}'
                for yn in text_names2:
                    strr += ' & \\textbf{' + yn + '}'

                fold_str += strr + '\\\\\n'
                fold_str += '\\cline{2-13}\n'
                
            strr = 'Fold-' + str(fold + 1) + ' MAE'
            for yh in yhats:
                strr += ' & %.2f' % (mae_print(yh) + 0)
            fold_str_mae += strr+' \\\\\n' 

            strr = 'Fold-' + str(fold + 1) + ' Bias'
            for yh in yhats:
                strr += ' & %.2f' % (bias_print(yh) + 0)
            fold_str_bias += strr+' \\\\\n' 


            strr = 'Fold-' + str(fold + 1) + ' \% wins'
            for pw in percent_wins:
                strr += ' & %.2f' % pw
            fold_str_pw += strr +' \\\\'
            if not fold + 1 == 4:
                fold_str_pw += '\n'

            
            errors[fold,:] = np.array([mae(yh[test]-y[test]) for yh in yhats])
            bias_errors[fold,:] = np.array([bias_print(yh) for yh in yhats])
            weights[fold,:] = np.array([len(test) for yh in yhats])
            pws[fold,:] = np.array(percent_wins)    
 
            hyatplots = yhats#[yhat_n_l, yhat_n_h, yhat_m_l, yhat_m_h, yhat_s_l, yhatnfolds]
            pltfontsize=10
            
                
            if PLOT_KFOLDS:
                for ll in range(n_models):
                    if fold == 0:
                        plt.rc('text', usetex=True)
#                                plt.rc('font', family='serif')
                        plt.rcParams["font.family"] = "Times New Roman"
#                                plt.figure(ll, dpi=200)
                        figs[ll],axs[ll] = plt.subplots(int(nfolds/2),2,figsize=(2.8,2.1),dpi=300)
                    ax1 = axs[ll]
                    fig1 = figs[ll]
                    yy = np.linspace(-20,100,1000)
                    yhatplot = hyatplots[ll]
                    aa3, = ax1[int(fold/2),np.mod (fold,2)].plot(yy,yy,'k:',linewidth=.5,alpha=.5)
                    aa1 = ax1[int (fold/2),np.mod (fold,2)].scatter(y[train],yhatplot[train], s=40, marker='o',facecolors='none', edgecolors='gray',label='Training Data')
                    aa2 = ax1[int (fold/2),np.mod (fold,2)].scatter(y[test],yhatplot[test], s=55, marker='x',facecolors='k', edgecolors='k',label='Testing Data')
                       
                    if fold >= 2:
                        ax1[int(fold/2),np.mod(fold,2)].set_xticks([0,30,60,90])
                    else:
                        ax1[int(fold/2),np.mod(fold,2)].set_xticks([])
                    if np.mod (fold,2)== 0: 
                        ax1[int(fold/2),np.mod(fold,2)].set_yticks([0,30,60,90])
                    else:
                        
                        ax1[int(fold/2),np.mod(fold,2)].set_yticks([])
                        
                    ax1[int(fold/2),np.mod(fold,2)].text(2, 76, lll[fold] + ') Fold-' + str(i+1),fontsize=pltfontsize)
                    ax1[int(fold/2),np.mod(fold,2)].set_xlim([-5,100])
                    ax1[int(fold/2),np.mod(fold,2)].set_ylim([-5,100])
                    
                     
                    
                    if fold == 3:
                        plt.setp(ax1[1,0].xaxis.get_majorticklabels(), ha="right")
                        plt.setp(ax1[1,1].xaxis.get_majorticklabels(), ha="center")
                        plt.setp(ax1[0,0].yaxis.get_majorticklabels(), va="center")
                        plt.setp(ax1[1,0].yaxis.get_majorticklabels(), va="top")
                        ax = fig1.add_subplot(111)
#                                fig1.set_size_inches(4,3)
                        ax.set_xlabel(r'In Situ SSC [mg L$^{-1}$]', labelpad=20, fontsize=pltfontsize)
                        ax.set_ylabel(r'Remotely sensed SSC [mg L$^{-1}$]', labelpad=20, fontsize=pltfontsize)
                        ax.set_facecolor('None')
                        ax.spines["top"].set_visible(False)
                        ax.spines["bottom"].set_visible(False)
                        ax.spines["left"].set_visible(False)
                        ax.spines["right"].set_visible(False)
                        ax.set_xticks([])
                        ax.set_yticks([])
                        ax.text(-0.31,.95,'('+llll[ll] + ')')
                        fig1.tight_layout()
                        fig1.subplots_adjust(bottom=0.21)
                        fig1.subplots_adjust(hspace=.08)
                        fig1.subplots_adjust(wspace=.08)
                        fig1.subplots_adjust(left=0.23)
                        fig1.subplots_adjust(top=.965)
                        fig1.subplots_adjust(right=.97)
                        
                        fig1.savefig(figure_dir + '/kmeans_'+ynames2[ll]+'.png',dpi=200)
                        
            fold += 1

                
        c_n_h = worker_func_nechad(X, y, channel) 
        c_n_l = worker_func_nechad2(X, y, channel)
    
        c_nnir_h = worker_func_nechad3(X, y, channel2)
        c_nnir_l = worker_func_nechad4(X, y, channel2)

        c_nir_l = worker_func_multiliniear(np.array([X[:,channel2]]).transpose(), y)
        c_s_l = worker_func_multiliniear(np.array([X[:,simple_channel]]).transpose(), y)
        c_sl= worker_func_multiliniear2(np.array([X[:,simple_channel]]).transpose(), np.log(y) )
        c_p_l = worker_func_power(X[:,simple_channel],y)
        c_p_h = worker_func_power2(X[:,simple_channel],y)
    #                print(c_p)
        
        yhat_n_s = nechadfull(c_n_s,X[:,channel])
        yhat_n_h = nechad(c_n_h,X[:,channel])
        yhat_n_l = nechad(c_n_l,X[:,channel])
        yhat_nnir_h = nechad3(c_nnir_h,X[:,channel2])
        yhat_nnir_l = nechad3(c_nnir_l,X[:,channel2])
        

        
        yhat_s_l = multilinear_func(c_s_l,np.array([X[:,simple_channel]]).transpose())
        yhat_snir_l = multilinear_func(c_nir_l,np.array([X[:,channel2]]).transpose())
        yhat_log_l = np.exp(multilinear_func(c_sl,np.array([X[:,simple_channel]]).transpose()))
        yhat_p_l = power_c(c_p_l,X[:,simple_channel])
        yhat_p_h = power_c(c_p_h,X[:,simple_channel])
    
        X2 = X[:,:]
        #                    X2 = X2.reshape(-1,1)
        

        scaler.fit(X2[:,:])  
        X2 = scaler.transform(X2)
#        u,s,v = np.linalg.svd(X2)
#        
#        X2 = u[:,:4]
        
        c_ml_h = worker_func_multiliniear(X2, y)
        c_ml_l = worker_func_multiliniear2(X2, y)
 
    
        yhat_ml_h = multilinear_func(c_ml_h,X2)
        yhat_ml_l = multilinear_func(c_ml_l,X2)
        
        X2 = X[:,:4]
        
        scaler.fit(X2[train,:])  
        X2 = scaler.transform(X2)
        #                    
        clf = MLPRegressor(solver='lbfgs',hidden_layer_sizes=(5,),alpha=1e-5,random_state=1)
        clf.fit(X2[:,:],y[:])
        yhat_net = clf.predict(X2)
        
        final_ys = [yhat_n_s, yhat_n_l, yhat_n_h, yhat_nnir_l, yhat_nnir_h, yhat_ml_l, yhat_ml_h, yhat_s_l, yhat_snir_l, yhat_log_l, yhat_p_l, yhat_p_h, yhat_net]
    
        final_ys = [final_ys[iiii] for iiii in model_indx]
            
        final_percent_wins = []
        for ii in range(len(final_ys)):
            wins = 0
            compares = 0
            for jj in range(len(final_ys)):
                if ii == jj:
                    continue
                for kk in range(len(final_ys[0])):
                    compares += 1
                    if np.abs(yhats[ii][kk] - y[kk]) <= np.abs(yhats[jj][kk] - y[kk]):
                        wins+=1
            final_percent_wins.append(np.round(wins*100./compares,1))
        s = ''
        if minutes == '15' or minutes == '60':# or minutes == '120':
            s += '\\begin{table}[!ht]\n'
            if not minutes == '15':
                s += '\\ContinuedFloat\n'
#            s += '\\begin{subtable}[t]{\\textwidth}\n'
            s += '\\scriptsize\n'
            s += '\\centering\n'
            s += '\\setlength\\tabcolsep{2pt}\n'
            s += '\\renewcommand{\\arraystretch}{1.2}\n'
            s += '\\begin{tabular}{l'
            for jj in range(n_models):
                s += ' | r'
            s += '}\n' 
        s += '\\hline\n\hline\n\\multicolumn{' + str(len(final_ys)) + '}{c}{\\textbf{'+ str(minutes)+ '-Minute threshold}}'
        
    #                for e in final_ys:
    #                    s += ' & '
        s += '\\\\\n\hline\n'
        s += fold_str       
#        s += '\\textbf{K-folds}'
#        for e in final_ys:
#            s += ' & '
#        s += '\\\\'
        final_string += s 
        
        final_string += fold_str_mae+'\hdashline\n'+fold_str_bias+'\hdashline\n'#+fold_str_pw+'\n\hdashline' + '\n'
        s = '\\textbf{Mean MAE}'
        for e in np.average(errors,0, weights=weights):
            s += ' & %.2f' % (np.round(e,2))
        s += '\\\\'
        final_string += s + '\n'
 
        s = '\\textbf{StdDev. MAE}'
        for e in np.std(errors,0):
            s += ' & %.2f' % (np.round(e,2))
        s += '\\\\'
        final_string += s + '\n'
        
        
        s = '\\textbf{Mean Bias}'
        for e in np.average(bias_errors,0, weights=weights):
            s += ' & %.2f' % (np.round(e,2))
        s += '\\\\'
        final_string += s + '\n'
        
        s = '\\textbf{StdDev. Bias}'
        for e in np.std(bias_errors,0):
            s += ' & %.2f' % (np.round(e,2))
        s += '\\\\'
        final_string += s + '\n'
        
    #                s = '\\textbf{Mean \\% Wins}'
    #                for e in np.median(pws,0):#,weights=weights):
    #                    s += ' & ' + str(np.round(e,2))
    #                s += '\\\\'
    #                final_string += s + '\n'
        final_string += '\\hline' + '\n'
        
        s = '\\textbf{Full dataset}'
        for e in final_ys:
            s += ' & '
        s += '\\\\\n'
        final_string += s
      
        s = 'MAE'
        for e in final_ys:
            s += ' & %.2f' % (np.round(r2_score(y,e), 2) + 0)
        s += '\\\\'
        final_string += s + '\n'
        
        s = 'Bias'
        for e in final_ys:
            s += ' & %.2f' % (np.round(np.mean(y-e),2) + 0)
        s += '\\\\'
        final_string += s + '\n'
        
    #                s = '\\% Wins'
    #                for e in final_percent_wins:
    #                    s += ' & %.2f' % e
    #                final_string += s + '\n'
        
        s = ''
        if minutes == '30' or minutes == '90' or minutes == '120':
            s += '\\end{tabular}\n'
            
#            s += '\\end{subtable}\n'
            s += '\\caption{'
            if not minutes == '30':
                s += '(continued) '
            s += 'Comparative results of K-folds calibration process for 12 models of remote sensing SSC.}\n'
            if not minutes == '30':
                s += '\\label{tab:calibration}\n'
            s += '\\end{table}\n'
            final_string += s + '\n'
        PLOT_EXAMPLE_THRESHOLD = False
        if PLOT_EXAMPLE_THRESHOLD:
            if minutes == '30':
                c30 = c_n_h
                x30 = X[:,channel]
                y30 = y
            if minutes == '60':
                x60 = X[:,channel]
                y60 = y
                c60 = c_n_h
            
    if minutes == '60' and PLOT_60_MIN:
        fignew = plt.figure(figsize=(6,6))
        plt.rc('text', usetex=True)
        plt.rc('text', usetex=True)
        plt.rcParams["font.family"] = "Times New Roman"
    
        
        for plotmodel in range(n_models):
            if plotmodel >= 12:
                continue
            ax = fignew.add_subplot(4, 3, plotmodel + 1)
            plt.plot(y, y, 'k', alpha=.5)
            plt.scatter(y, final_ys[plotmodel], marker='o',facecolors='none', edgecolors='k')
            plt.xlim([-5,90])
            plt.ylim([-5,90])
            
            ii = int(plotmodel/3)+1
            jj = np.mod(i,3)+1
    
            ax.set_xticks([])
            ax.set_yticks([])
            ax.text(2, 76, '(' + llll[plotmodel] +')',fontsize=pltfontsize)
    
            if ii == 4:
                ax.set_xticks([0,30,60,90])
    #                        ax.set_xlabel(r'In Situ SSC [mg L$^{-1}$]', labelpad=20, fontsize=pltfontsize)
                        
            if jj == 1:
                ax.set_yticks([0,30,60,90])
    #                        ax.set_ylabel(r'Remotely sensed SSC [mg L$^{-1}$]', labelpad=20, fontsize=pltfontsize)
                
    #                        plt.ylabel([])
            
            
        ax = fignew.add_subplot(111)
    #                                fig1.set_size_inches(4,3)
        ax.set_xlabel(r'In Situ SSC [mg L$^{-1}$]', labelpad=20, fontsize=pltfontsize)
        ax.set_ylabel(r'Remotely sensed SSC [mg L$^{-1}$]', labelpad=20, fontsize=pltfontsize)
        ax.set_facecolor('None')
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
    
        fignew.tight_layout()
        fignew.subplots_adjust(bottom=0.08)
    #                fig1.subplots_adjust(hspace=.08)
    #                fi1.subplots_adjust(wspace=.08)
        fignew.subplots_adjust(left=0.08)
        fignew.subplots_adjust(top=.98)
        fignew.subplots_adjust(right=.98)
        plt.savefig(figure_dir + '/compare_calibration.png', dpi=plt_format.dpi)

        
                
    print(final_string)
        
    if PLOT_EXAMPLE_THRESHOLD:
        indx_30 = []
        for i in range(len(x30)):
            for j in range(len(x60)):
                if x30[i] == x60[j] and y30[i] == y60[j]:
                    indx_30.append(j)
                    continue
                
        x60remainder = np.delete(x60,indx_30)
        y60remainder = np.delete(y60,indx_30)
        
        plt.figure(figsize=(6,3))
        
        plt.rc('text', usetex=True)
        plt.rcParams["font.family"] = "Times New Roman"
    #

        ax = plt.subplot(1,2,1)
        plt.plot(y,y,'k', alpha=.5)
        plt.plot(y30, nechad(c30,x30), 'ko', fillstyle='none')
        plt.plot(y60remainder, nechad(c30,x60remainder), 'kx')
        plt.xlabel(r'In Situ SSC (mg L$^{-1}$)', fontsize=pltfontsize)
        plt.ylabel(r'Remotely sensed SSC (mg L$^{-1}$)', fontsize=pltfontsize)
        plt.xlim([0,90])
        plt.xticks([0,30,60,90])
        plt.ylim([0,90])
        plt.yticks([0,30,60,90])
        ax.text(2, 80, '(a)',fontsize=pltfontsize)
        
        ax = plt.subplot(1,2,2)
        plt.plot(y,y,'k', alpha=.5)
        plt.plot(y30, nechad(c60,x30), 'ko', fillstyle='none')
        plt.plot(y60remainder, nechad(c60,x60remainder), 'kx')        
        plt.xlabel(r'In Situ SSC (mg L$^{-1}$)', fontsize=pltfontsize)
#        plt.ylabel(r'Remotely sensed SSC (mg L$^{-1}$)', fontsize=pltfontsize)
        plt.xlim([0,90])
        plt.xticks([0,30,60,90])
        plt.ylim([0,90])
        plt.yticks([0,30,60,90])
        ax.text(2, 80, '(b)',fontsize=pltfontsize)
        
        plt.tight_layout()
        plt.savefig(figure_dir + '/effect_of_big_ssc.png', dpi=plt_format.dpi)
if __name__ == "__main__":
    main()