#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 11:07:06 2019

@author: mcarlu
"""

import numpy as np
from scipy.special import erf
import scipy.stats
import math

def gaussian(x,mean,std):
    return 1/(np.sqrt(2*math.pi*std**2))*math.exp(-(x-mean)**2/(2*std**2))

def deriv1(funct,v,numb_var,h):

    first_deriv=[]
    #vsec_plus=np.ones(numb_var)
    #vsec_minus=np.ones(numb_var)
    
    #for i in range(numb_var+numb_adapt): # initialization of perturbation vector
    vsec_plus=v.copy()
    vsec_minus=v.copy()
    
    for i in range(numb_var):
        vsec_plus[i]+=0.5*h # perturb one component
        vsec_minus[i]-=0.5*h
        first_deriv.append((funct(*vsec_plus)-funct(*vsec_minus))/h)
        vsec_plus[i]=v[i] # reset component
        vsec_minus[i]=v[i]
        
    return first_deriv

def deriv2(funct,v,numb_var,h):  
    # separate first and second deriv to avoid useless confusion
    sec_deriv=[]
    #vsec_plus=np.ones(numb_var)
    #vsec_minus=np.ones(numb_var)
    #first_deriv=deriv1(funct,v,numb_var,h)
    
    #for i in range(numb_var+numb_adapt): # initialization of perturbation vector
    vsec_plus=v.copy()
    vsec_minus=v.copy()
        
    for i in range(numb_var):
        deriv_2_temp=[]
        for j in range(numb_var):
            vsec_plus[j]+=0.5*h # perturb one component
            vsec_minus[j]-=0.5*h    
            deriv_2_temp.append((deriv1(funct,vsec_plus,numb_var,h)[i]-\
                                 deriv1(funct,vsec_minus,numb_var,h)[i])/h)
            vsec_plus[j]=v[j] # reset component
            vsec_minus[j]=v[j]
        
        sec_deriv.append(deriv_2_temp)
        
    return sec_deriv

def deriv1b(funct,v,numb_var,h,i):

    first_deriv=[]
    #vsec_plus=np.ones(numb_var)
    #vsec_minus=np.ones(numb_var)
    
    #for i in range(numb_var+numb_adapt): # initialization of perturbation vector
    vsec_plus=v.copy()
    vsec_minus=v.copy()
    
    #for i in range(numb_var):
    vsec_plus[i]+=0.5*h # perturb one component
    vsec_minus[i]-=0.5*h
    first_deriv=(funct(*vsec_plus)-funct(*vsec_minus))/h
    #vsec_plus[i]=v[i] # reset component
    #vsec_minus[i]=v[i]
        
    return first_deriv

def deriv2b(funct,v,numb_var,h,i,j):  
    # separate first and second deriv to avoid useless confusion
    #vsec_plus=np.ones(numb_var)
    #vsec_minus=np.ones(numb_var)
    #first_deriv=deriv1(funct,v,numb_var,h)
    
    #for i in range(numb_var+numb_adapt): # initialization of perturbation vector
    vsec_plus=v.copy()
    vsec_minus=v.copy()
    
    vsec_plus[j]+=0.5*h # perturb one component
    vsec_minus[j]-=0.5*h    
    sec_deriv=(deriv1b(funct,vsec_plus,numb_var,h,i)-deriv1b(funct,vsec_minus,numb_var,h,i))/h
    
    return sec_deriv

def deriv3b(funct,v,numb_var,h,i,j,k):
    
    vsec_plus=v.copy()
    vsec_minus=v.copy()
    
    vsec_plus[k]+=0.5*h # perturb one component
    vsec_minus[k]-=0.5*h    
    third_deriv=(deriv2(funct,vsec_plus,numb_var,h,i,j)-deriv2(funct,vsec_minus,numb_var,h,i,j))/h


        
def rk4_dN_dp_General(x0,params,time,func):

    p=params['numb_var']# // variables needed to build arrays
    #int lp = par[8];
    dx=np.zeros(p)  
    deriv=np.zeros(p)
    xt = np.zeros(p)

    for i in range(p):
        dx[i]=deriv[i]=xt[i]=0
    
    deriv,TF=func(x0, time, params)

    for i in range(p): 
        dx[i]=deriv[i] # x[i]=k1[i]
        xt[i]=x0[i]+0.5*deriv[i]*params['tstep']# argument for k2 (careful k1,2,3,4 in comment = k's of RK method, different from k1 & k2 of the code)

    time+=params['tstep']/2;

    deriv,TF=func(xt, time, params)
    
    for i in range(p):
        dx[i]+=2*deriv[i] # x[i]=k1[i]+2*k2[i]
        xt[i]=x0[i]+0.5*deriv[i]*params['tstep'] # argument for k3	


    deriv,TF=func(xt, time, params)


    for i in range(p):
        dx[i]+=2*deriv[i] 
        xt[i]=x0[i]+deriv[i]*params['tstep']# //argument for k4

    time+=params['tstep']/2;

    deriv,TF=func(xt, time, params)


    for i in range(p):
        dx[i]+=deriv[i] # x[i]=k1[i]+2*k2[i]
        xt[i]=x0[i]+deriv[i]*params['tstep'] 


    for i in range(p):
        x0[i]=x0[i]+dx[i]*params['tstep']/6.0
        
      
    return TF

def Euler_method_withstep(x0,params,time):
    
    p=params[1]
    deriv=MeanField_Opt(x0, params, time)
    
    for i in range(p): 
        x0[i]+=deriv[i]*params[0]

    time+=params[0]
    
def Euler_method_General(x0,params,time,func):
    
    p=params['numb_var']
    deriv,TF=func(x0, time,params)
    
    for i in range(p): 
        x0[i]+=deriv[i]*params['tstep'] 

    time+=params['tstep'] 
    
    return TF


def MeanField_dN_dp(x0,t,params):
    #print(' t = ',t)
    N0_e=8000
    N0_i=2000
    tstep=params['tstep']
    tot_numb=params['numb_var']
    tauw=params['tauw']
    a=params['a']
    b=params['b']
    El=params['El']
    N=params['size_subpop']
    numb_var=params['numb_pop']
    numb_adapt=params['numb_adapt']
    F=params['TF']
    h=params['h']
    T=params['T']
    mu=params['mu']
    ext_drive=params['ext_inp']
    input_rate=params['inp_funct']
    
    if (input_rate != 0): input_func=input_rate(t, *params['inp_par'])
    else : input_func=0
 
    
    ''' We keep the original TF framework'''
    
    numb_v_TF=int(2)
    numb_w_TF=int(1)
    
    '''Define the number of subpopulations, they are assumed to be evenly distributed'''
    
    numb_sub_exc=params['numb_sub_exc']
    numb_sub_inh=params['numb_sub_inh']
    
    
    Ntot_exc=0
    Ntot_inh=0
    
    for i in range(numb_sub_exc): Ntot_exc+=N[i]
    for i in range(numb_sub_inh): Ntot_inh+=N[i+numb_sub_exc] #convention 
    
    deriv=np.zeros(tot_numb)
    
    ''' Define vsec_vec, because of the way TF are defined and the different arguments they take 
    (adaptive versus non adaptive populations)'''
    
    vsec_vec=np.zeros((numb_var,numb_v_TF+numb_w_TF))  # Needed to pass different values to TF1 and TF2
    
    ''' Define dp and dN to account for variability in population size and connectivity '''
   
    p=params['p_conn']
    #dN=[Ntot_exc/N0_e,Ntot_inh/N0_i] # Ntot is taken because it is averaged over populations
   
   
   #for i in range(numb_sub_exc): dN.append(1+(Ntot_exc-N0_e)/N0_e) # Ntot_exc is taken because it is averaged over populations
   #for i in range(numb_sub_inh): dN.append(1+(Ntot_inh-N0_i)/N0_i)
    
    
    for i in range(numb_var):#receive
        for j in range(numb_sub_exc): ## excitatory input #sending
            vsec_vec[i][0]+=x0[j]*N[j]/N0_e*p[i][j]/(5e-2) # initially only divide by numb_sub_exc : true if populations are evenly distributed ! => /2= * 4000/8000 ##normalize the probility of connectivity #p1N1/p0N0 * TF1'
        
        ''' dividing by Ntot_exc is for averaging over subpops,  (1+dN[i]) for accounting for difference
        in initial population size w.r.t TF calculation '''
        
        for j in range(numb_sub_inh): ## inhibitory input
            vsec_vec[i][1]+=x0[j+numb_sub_exc]*N[j+numb_sub_exc]/N0_i*p[i][j+numb_sub_exc]/(5e-2)
        
    
    for i in range(numb_adapt):
        vsec_vec[i][numb_v_TF]=x0[numb_var*numb_var+numb_var+i]
    
    '''CAREFUL : the above assumes that all adaptive variables are put first'''        
    
    p_pois=params['p_pois']
    
    
    for i in range(numb_var): ## external drive
        vsec_vec[i][0]+=(ext_drive[i]+input_func)*p_pois[i]/(5e-2)*Ntot_exc/N0_e#+input_rate(t, *params[23]) #+input_rate(t, 2., 0.1, 0.1, 120, 1) #+input_rate(t,0.1,0.005,0.02,10)
    
    '''dN[0] is taken because generally Poisson input are taken with same pop size as excitatory neurons'''

    
    # Define the TF derivatives here to compute them only once #
    
    Delta1=np.zeros((numb_var,numb_var))
    Delta2=np.zeros((numb_var,numb_var,numb_var))
    
    #Delta1_old=np.zeros((numb_var,numb_var))
    #Delta2_old=np.zeros((numb_var,numb_var,numb_var))
    
    Eval_TF=np.zeros(numb_var)

    
    for i in range(numb_var):
        Eval_TF[i]=F[i](*vsec_vec[i])
        #print('evaluated TF',i,' = ', Eval_TF[i])

        for k in range(numb_sub_exc): 
            
            Delta1[i][k]=deriv1b(F[i],vsec_vec[i],numb_v_TF,h,0)*N[k]/N0_e*p[i][k]/(5e-2) # 0 is taken because TF takes only exc and inhib inputs, without discarding the subpops
            
            '''Be careful here, there might be a problem with the type of numb_sub_(exc/inh) (int/float)'''
                
        for k in range(numb_sub_inh):
            
            Delta1[i][k+numb_sub_exc]=deriv1b(F[i],vsec_vec[i],numb_v_TF,h,1)*N[k+numb_sub_exc]/N0_i*p[i][k+numb_sub_exc]/(5e-2)

        for k in range(numb_sub_exc):
            for j in range(numb_sub_exc):

                Delta2[i][k][j]=deriv2b(F[i],vsec_vec[i],numb_v_TF,h,0,0)*\
                N[k]/N0_e*N[j]/N0_e*p[i][k]/(5e-2)*p[i][j]/(5e-2)
                
            for j in range(numb_sub_inh):
                Delta2[i][k][j+numb_sub_exc]=deriv2b(F[i],vsec_vec[i],numb_v_TF,h,0,1)*\
                N[k]/N0_e*N[j+numb_sub_exc]/N0_i*p[i][k]/(5e-2)*p[i][j+numb_sub_exc]/(5e-2)
                
        for k in range(numb_sub_inh):
            for j in range(numb_sub_exc):
                    Delta2[i][k+numb_sub_exc][j]=deriv2b(F[i],vsec_vec[i],numb_v_TF,h,1,0)*\
                    N[k+numb_sub_exc]/N0_i*N[j]/N0_e*p[i][k+numb_sub_exc]/(5e-2)*p[i][j]/(5e-2)
            for j in range(numb_sub_inh):
                Delta2[i][k+numb_sub_exc][j+numb_sub_exc]=deriv2b(F[i],vsec_vec[i],numb_v_TF,h,1,1)*\
                N[k+numb_sub_exc]/N0_i*N[j+numb_sub_exc]/N0_i*p[i][k+numb_sub_exc]/(5e-2)*\
                p[i][j+numb_sub_exc]/(5e-2)
                #print(Delta2[i][k][j])
                
       
    for i in range(numb_var):
        deriv[i]=(Eval_TF[i]-x0[i])/T

        deriv[numb_var*i+i+numb_var]+=(Eval_TF[i]*(1./T-Eval_TF[i])/N[i])/T
        for j in range(numb_var):
            index=numb_var*i+j+numb_var # define here the location of the considered c variable and its derivative
            for k in range(numb_var): 
                deriv[i]+=(0.5*Delta2[i][j][k]*x0[numb_var*j+k+numb_var])/T
                index_ik=numb_var*i+k+numb_var
                index_jk=numb_var*j+k+numb_var
                deriv[index]+=(x0[index_jk]*Delta1[i][k]+x0[index_ik]*Delta1[j][k])/T
                
            deriv[index]+=((Eval_TF[i]-x0[i])*(Eval_TF[j]-x0[j])-2*x0[index])/T
  
                
    for i in range(numb_adapt):
        deriv[numb_var**2+numb_var+i]=-x0[numb_var**2+numb_var+i]/tauw[i]+b[i]*x0[i]+a[i]*(mu(x0,params,i,t)-El[i])/tauw[i]
        
      # DEFINE MU !!!!#
    
    
    ''' Printing to debug '''
    
    
    #print('\n','\n','TIME = ', t, '\n','\n', 'Delta1 = ', Delta1,'\n','Delta1 Old = ', Delta1_old, '\n', 'Delta2 = ', Delta2,'\n', 'Delta2 Old= ', Delta2_old,'\n')
                
    return deriv,Eval_TF

def MeanField_dN_dp_NoTF(x0,t,params):
    #print(' t = ',t)
    N0_e=8000
    N0_i=2000
    tstep=params['tstep']
    tot_numb=params['numb_var']
    tauw=params['tauw']
    a=params['a']
    b=params['b']
    El=params['El']
    N=params['size_subpop']
    numb_var=params['numb_pop']
    numb_adapt=params['numb_adapt']
    F=params['TF']
    h=params['h']
    T=params['T']
    mu=params['mu']
    ext_drive=params['ext_inp']
    input_rate=params['inp_funct']
    
    if (input_rate != 0): input_func=input_rate(t, *params['inp_par'])
    else : input_func=0
 
    
    ''' We keep the original TF framework'''
    
    numb_v_TF=int(2)
    numb_w_TF=int(1)
    
    '''Define the number of subpopulations, they are assumed to be evenly distributed'''
    
    numb_sub_exc=params['numb_sub_exc']
    numb_sub_inh=params['numb_sub_inh']
    
    
    Ntot_exc=0
    Ntot_inh=0
    
    for i in range(numb_sub_exc): Ntot_exc+=N[i]
    for i in range(numb_sub_inh): Ntot_inh+=N[i+numb_sub_exc]
    
    deriv=np.zeros(tot_numb)
    
    ''' Define vsec_vec, because of the way TF are defined and the different arguments they take 
    (adaptive versus non adaptive populations)'''
    
    vsec_vec=np.zeros((numb_var,numb_v_TF+numb_w_TF))  # Needed to pass different values to TF1 and TF2
    
    ''' Define dp and dN to account for variability in population size and connectivity '''
   
    p=params['p_conn']
    #dN=[Ntot_exc/N0_e,Ntot_inh/N0_i] # Ntot is taken because it is averaged over populations
   
   
   #for i in range(numb_sub_exc): dN.append(1+(Ntot_exc-N0_e)/N0_e) # Ntot_exc is taken because it is averaged over populations
   #for i in range(numb_sub_inh): dN.append(1+(Ntot_inh-N0_i)/N0_i)
    
    
    for i in range(numb_var):
        for j in range(numb_sub_exc): ## excitatory input
            vsec_vec[i][0]+=x0[j]*N[j]/N0_e*p[i][j]/(5e-2) # initially only divide by numb_sub_exc : true if populations are evenly distributed ! => /2= * 4000/8000
        
        ''' dividing by Ntot_exc is for averaging over subpops,  (1+dN[i]) for accounting for difference
        in initial population size w.r.t TF calculation '''
        
        for j in range(numb_sub_inh): ## inhibitory input
            vsec_vec[i][1]+=x0[j+numb_sub_exc]*N[j+numb_sub_exc]/N0_i*p[i][j+numb_sub_exc]/(5e-2)
        
    
    for i in range(numb_adapt):
        vsec_vec[i][numb_v_TF]=x0[numb_var*numb_var+numb_var+i]
    
    '''CAREFUL : the above assumes that all adaptive variables are put first'''        
    
    p_pois=params['p_pois']
    
    
    for i in range(numb_var): ## external drive
        vsec_vec[i][0]+=(ext_drive[i]+input_func)*p_pois[i]/(5e-2)*Ntot_exc/N0_e#+input_rate(t, *params[23]) #+input_rate(t, 2., 0.1, 0.1, 120, 1) #+input_rate(t,0.1,0.005,0.02,10)
    
    '''dN[0] is taken because generally Poisson input are taken with same pop size as excitatory neurons'''

    
    # Define the TF derivatives here to compute them only once #
    
    Delta1=np.zeros((numb_var,numb_var))
    Delta2=np.zeros((numb_var,numb_var,numb_var))
    
    #Delta1_old=np.zeros((numb_var,numb_var))
    #Delta2_old=np.zeros((numb_var,numb_var,numb_var))
    
    Eval_TF=np.zeros(numb_var)

    
    for i in range(numb_var):
        Eval_TF[i]=F[i](*vsec_vec[i])
        #print('evaluated TF',i,' = ', Eval_TF[i])

        for k in range(numb_sub_exc): 
            
            Delta1[i][k]=deriv1b(F[i],vsec_vec[i],numb_v_TF,h,0)*N[k]/N0_e*p[i][k]/(5e-2) # 0 is taken because TF takes only exc and inhib inputs, without discarding the subpops
            
            '''Be careful here, there might be a problem with the type of numb_sub_(exc/inh) (int/float)'''
                
        for k in range(numb_sub_inh):
            
            Delta1[i][k+numb_sub_exc]=deriv1b(F[i],vsec_vec[i],numb_v_TF,h,1)*N[k+numb_sub_exc]/N0_i*p[i][k+numb_sub_exc]/(5e-2)

        for k in range(numb_sub_exc):
            for j in range(numb_sub_exc):

                Delta2[i][k][j]=deriv2b(F[i],vsec_vec[i],numb_v_TF,h,0,0)*\
                N[k]/N0_e*N[j]/N0_e*p[i][k]/(5e-2)*p[i][j]/(5e-2)
                
            for j in range(numb_sub_inh):
                Delta2[i][k][j+numb_sub_exc]=deriv2b(F[i],vsec_vec[i],numb_v_TF,h,0,1)*\
                N[k]/N0_e*N[j+numb_sub_exc]/N0_i*p[i][k]/(5e-2)*p[i][j+numb_sub_exc]/(5e-2)
                
        for k in range(numb_sub_inh):
            for j in range(numb_sub_exc):
                    Delta2[i][k+numb_sub_exc][j]=deriv2b(F[i],vsec_vec[i],numb_v_TF,h,1,0)*\
                    N[k+numb_sub_exc]/N0_i*N[j]/N0_e*p[i][k+numb_sub_exc]/(5e-2)*p[i][j]/(5e-2)
            for j in range(numb_sub_inh):
                Delta2[i][k+numb_sub_exc][j+numb_sub_exc]=deriv2b(F[i],vsec_vec[i],numb_v_TF,h,1,1)*\
                N[k+numb_sub_exc]/N0_i*N[j+numb_sub_exc]/N0_i*p[i][k+numb_sub_exc]/(5e-2)*\
                p[i][j+numb_sub_exc]/(5e-2)
                #print(Delta2[i][k][j])
                
       
    for i in range(numb_var):
        deriv[i]=(Eval_TF[i]-x0[i])/T

        deriv[numb_var*i+i+numb_var]+=(Eval_TF[i]*(1./T-Eval_TF[i])/N[i])/T
        for j in range(numb_var):
            index=numb_var*i+j+numb_var # define here the location of the considered c variable and its derivative
            for k in range(numb_var): 
                deriv[i]+=(0.5*Delta2[i][j][k]*x0[numb_var*j+k+numb_var])/T
                index_ik=numb_var*i+k+numb_var
                index_jk=numb_var*j+k+numb_var
                deriv[index]+=(x0[index_jk]*Delta1[i][k]+x0[index_ik]*Delta1[j][k])/T
                
            deriv[index]+=((Eval_TF[i]-x0[i])*(Eval_TF[j]-x0[j])-2*x0[index])/T
  
                
    for i in range(numb_adapt):
        deriv[numb_var**2+numb_var+i]=-x0[numb_var**2+numb_var+i]/tauw[i]+b[i]*x0[i]+a[i]*(mu(x0,params,i,t)-El[i])/tauw[i]
        
      # DEFINE MU !!!!#
    
    
    ''' Printing to debug '''
    
    
    #print('\n','\n','TIME = ', t, '\n','\n', 'Delta1 = ', Delta1,'\n','Delta1 Old = ', Delta1_old, '\n', 'Delta2 = ', Delta2,'\n', 'Delta2 Old= ', Delta2_old,'\n')
                
    return deriv


def MeanField_dN_dp_Hybrid_noise(x0,t,params):
    #print(' t = ',t)
    N0_e=8000
    N0_i=2000
    tstep=params['tstep']
    tot_numb=params['numb_var']
    tauw=params['tauw']
    a=params['a']
    b=params['b']
    El=params['El']
    N=params['size_subpop']
    numb_var=params['numb_pop']
    numb_adapt=params['numb_adapt']
    F=params['TF']
    h=params['h']
    T=params['T']
    mu=params['mu']
    noise=params['noise']
    ext_drive=params['ext_inp']
    input_rate=params['inp_funct']
    
    if (input_rate != 0): input_func=input_rate(t, *params['inp_par'])
    else : input_func=0
 
    
    ''' We keep the original TF framework'''
    
    numb_v_TF=int(2)
    numb_w_TF=int(1)
    
    '''Define the number of subpopulations, they are assumed to be evenly distributed'''
    
    numb_sub_exc=params['numb_sub_exc']
    numb_sub_inh=params['numb_sub_inh']
    
    
    Ntot_exc=0
    Ntot_inh=0
    
    for i in range(numb_sub_exc): Ntot_exc+=N[i]
    for i in range(numb_sub_inh): Ntot_inh+=N[i+numb_sub_exc]
    
    deriv=np.zeros(tot_numb)
    
    ''' Define vsec_vec, because of the way TF are defined and the different arguments they take 
    (adaptive versus non adaptive populations)'''
    
    vsec_vec=np.zeros((numb_var,numb_v_TF+numb_w_TF))  # Needed to pass different values to TF1 and TF2
    
    ''' Define dp and dN to account for variability in population size and connectivity '''
   
    p=params['p_conn']
    #dN=[Ntot_exc/N0_e,Ntot_inh/N0_i] # Ntot is taken because it is averaged over populations
   
   
   #for i in range(numb_sub_exc): dN.append(1+(Ntot_exc-N0_e)/N0_e) # Ntot_exc is taken because it is averaged over populations
   #for i in range(numb_sub_inh): dN.append(1+(Ntot_inh-N0_i)/N0_i)
    
    
    for i in range(numb_var):
        for j in range(numb_sub_exc): ## excitatory input
            vsec_vec[i][0]+=x0[j]*N[j]/N0_e*p[i][j]/(5e-2) # initially only divide by numb_sub_exc : true if populations are evenly distributed ! => /2= * 4000/8000
        
        ''' dividing by Ntot_exc is for averaging over subpops,  (1+dN[i]) for accounting for difference
        in initial population size w.r.t TF calculation '''
        
        for j in range(numb_sub_inh): ## inhibitory input
            vsec_vec[i][1]+=x0[j+numb_sub_exc]*N[j+numb_sub_exc]/N0_i*p[i][j+numb_sub_exc]/(5e-2)
        
    
    for i in range(numb_adapt):
        vsec_vec[i][numb_v_TF]=x0[numb_var*numb_var+numb_var+i]
    
    '''CAREFUL : the above assumes that all adaptive variables are put first'''        
    
    p_pois=params['p_pois']
    
    vsec_vec_first=vsec_vec.copy()
    
    for i in range(numb_var): ## external drive
         vsec_vec[i][0]+=(ext_drive[i]+input_func)*p_pois[i]/(5e-2)*Ntot_exc/N0_e#+input_rate(t, *params[23]) #+input_rate(t, 2., 0.1, 0.1, 120, 1) #+input_rate(t,0.1,0.005,0.02,10)
         vsec_vec_first[i][0]+=(ext_drive[i]+input_func+noise[i])*p_pois[i]/(5e-2)*Ntot_exc/N0_e#+input_rate(t, *params[23]) #+input_rate(t, 2., 0.1, 0.1, 120, 1) #+input_rate(t,0.1,0.005,0.02,10)   
    
    '''dN[0] is taken because generally Poisson input are taken with same pop size as excitatory neurons'''

    
    # Define the TF derivatives here to compute them only once #
    
    Delta1=np.zeros((numb_var,numb_var))
    Delta2=np.zeros((numb_var,numb_var,numb_var))
    
    #Delta1_old=np.zeros((numb_var,numb_var))
    #Delta2_old=np.zeros((numb_var,numb_var,numb_var))
    
    Eval_TF=np.zeros(numb_var)

    
    for i in range(numb_var):
        Eval_TF[i]=F[i](*vsec_vec_first[i])
        #print('evaluated TF',i,' = ', Eval_TF[i])

        for k in range(numb_sub_exc): 
            
            Delta1[i][k]=deriv1b(F[i],vsec_vec_first[i],numb_v_TF,h,0)*N[k]/N0_e*p[i][k]/(5e-2) # 0 is taken because TF takes only exc and inhib inputs, without discarding the subpops
            
            '''Be careful here, there might be a problem with the type of numb_sub_(exc/inh) (int/float)'''
                
        for k in range(numb_sub_inh):
            
            Delta1[i][k+numb_sub_exc]=deriv1b(F[i],vsec_vec_first[i],numb_v_TF,h,1)*N[k+numb_sub_exc]/N0_i*p[i][k+numb_sub_exc]/(5e-2)

        for k in range(numb_sub_exc):
            for j in range(numb_sub_exc):

                Delta2[i][k][j]=deriv2b(F[i],vsec_vec[i],numb_v_TF,h,0,0)*\
                N[k]/N0_e*N[j]/N0_e*p[i][k]/(5e-2)*p[i][j]/(5e-2)
                
            for j in range(numb_sub_inh):
                Delta2[i][k][j+numb_sub_exc]=deriv2b(F[i],vsec_vec[i],numb_v_TF,h,0,1)*\
                N[k]/N0_e*N[j+numb_sub_exc]/N0_i*p[i][k]/(5e-2)*p[i][j+numb_sub_exc]/(5e-2)
                
        for k in range(numb_sub_inh):
            for j in range(numb_sub_exc):
                    Delta2[i][k+numb_sub_exc][j]=deriv2b(F[i],vsec_vec[i],numb_v_TF,h,1,0)*\
                    N[k+numb_sub_exc]/N0_i*N[j]/N0_e*p[i][k+numb_sub_exc]/(5e-2)*p[i][j]/(5e-2)
            for j in range(numb_sub_inh):
                Delta2[i][k+numb_sub_exc][j+numb_sub_exc]=deriv2b(F[i],vsec_vec[i],numb_v_TF,h,1,1)*\
                N[k+numb_sub_exc]/N0_i*N[j+numb_sub_exc]/N0_i*p[i][k+numb_sub_exc]/(5e-2)*\
                p[i][j+numb_sub_exc]/(5e-2)
                #print(Delta2[i][k][j])
                
       
    for i in range(numb_var):
        deriv[i]=(Eval_TF[i]-x0[i])/T

        deriv[numb_var*i+i+numb_var]+=(Eval_TF[i]*(1./T-Eval_TF[i])/N[i])/T
        for j in range(numb_var):
            index=numb_var*i+j+numb_var # define here the location of the considered c variable and its derivative
            for k in range(numb_var): 
                deriv[i]+=(0.5*Delta2[i][j][k]*x0[numb_var*j+k+numb_var])/T
                index_ik=numb_var*i+k+numb_var
                index_jk=numb_var*j+k+numb_var
                deriv[index]+=(x0[index_jk]*Delta1[i][k]+x0[index_ik]*Delta1[j][k])/T
                
            deriv[index]+=((Eval_TF[i]-x0[i])*(Eval_TF[j]-x0[j])-2*x0[index])/T
  
                
    for i in range(numb_adapt):
        deriv[numb_var**2+numb_var+i]=-x0[numb_var**2+numb_var+i]/tauw[i]+b[i]*x0[i]+a[i]*(mu(x0,params,i,t)-El[i])/tauw[i]
        
      # DEFINE MU !!!!#
    
    
    ''' Printing to debug '''
    
    
    #print('\n','\n','TIME = ', t, '\n','\n', 'Delta1 = ', Delta1,'\n','Delta1 Old = ', Delta1_old, '\n', 'Delta2 = ', Delta2,'\n', 'Delta2 Old= ', Delta2_old,'\n')
                
    return deriv,Eval_TF


def MeanField_dN_dp_FO(x0,t,params):
    #print(' t = ',t)
    N0_e=8000
    N0_i=2000
    tstep=params['tstep']
    tot_numb=params['numb_var']
    tauw=params['tauw']
    a=params['a']
    b=params['b']
    El=params['El']
    N=params['size_subpop']
    numb_var=params['numb_pop']
    numb_adapt=params['numb_adapt']
    F=params['TF']
    h=params['h']
    T=params['T']
    mu=params['mu']
    ext_drive=params['ext_inp']
    input_rate=params['inp_funct']
    
    if (input_rate != 0): input_func=input_rate(t, *params['inp_par'])
    else : input_func=0
 
    
    ''' We keep the original TF framework'''
    
    numb_v_TF=int(2)
    numb_w_TF=int(1)
    
    '''Define the number of subpopulations, they are assumed to be evenly distributed'''
    
    numb_sub_exc=params['numb_sub_exc']
    numb_sub_inh=params['numb_sub_inh']
 
    
    Ntot_exc=0
    Ntot_inh=0
    
    for i in range(numb_sub_exc): Ntot_exc+=N[i]
    for i in range(numb_sub_inh): Ntot_inh+=N[i+numb_sub_exc]
    
    deriv=np.zeros(tot_numb)
    
    ''' Define vsec_vec, because of the way TF are defined and the different arguments they take 
    (adaptive versus non adaptive populations)'''
    
    vsec_vec=np.zeros((numb_var,numb_v_TF+numb_w_TF))  # Needed to pass different values to TF1 and TF2
    
    ''' Define dp and dN to account for variability in population size and connectivity '''
   
    p=params['p_conn']
    #dN=[Ntot_exc/N0_e,Ntot_inh/N0_i] # Ntot is taken because it is averaged over populations
   
   
   #for i in range(numb_sub_exc): dN.append(1+(Ntot_exc-N0_e)/N0_e) # Ntot_exc is taken because it is averaged over populations
   #for i in range(numb_sub_inh): dN.append(1+(Ntot_inh-N0_i)/N0_i)
    
    
    for i in range(numb_var):
        for j in range(numb_sub_exc): ## excitatory input
            vsec_vec[i][0]+=x0[j]*N[j]/N0_e*p[i][j]/(5e-2) # initially only divide by numb_sub_exc : true if populations are evenly distributed ! => /2= * 4000/8000
        
        ''' dividing by Ntot_exc is for averaging over subpops,  (1+dN[i]) for accounting for difference
        in initial population size w.r.t TF calculation '''
        
        for j in range(numb_sub_inh): ## inhibitory input
            vsec_vec[i][1]+=x0[j+numb_sub_exc]*N[j+numb_sub_exc]/N0_i*p[i][j+numb_sub_exc]/(5e-2)
        
    
    for i in range(numb_adapt):
        vsec_vec[i][numb_v_TF]=x0[numb_var+i]
    
    '''CAREFUL : the above assumes that all adaptive variables are put first'''        
    
    p_pois=params['p_pois']
    
    
    for i in range(numb_var): ## external drive
        vsec_vec[i][0]+=(ext_drive[i]+input_func)*p_pois[i]/(5e-2)*Ntot_exc/N0_e#+input_rate(t, *params[23]) #+input_rate(t, 2., 0.1, 0.1, 120, 1) #+input_rate(t,0.1,0.005,0.02,10)
    
    '''dN[0] is taken because generally Poisson input are taken with same pop size as excitatory neurons'''

    
    # Define the TF derivatives here to compute them only once #
    
    Delta1=np.zeros((numb_var,numb_var))
    Delta2=np.zeros((numb_var,numb_var,numb_var))
    
    #Delta1_old=np.zeros((numb_var,numb_var))
    #Delta2_old=np.zeros((numb_var,numb_var,numb_var))
    
    Eval_TF=np.zeros(numb_var)

    
    for i in range(numb_var):
        Eval_TF[i]=F[i](*vsec_vec[i])
        #print('evaluated TF',i,' = ', Eval_TF[i])

        for k in range(numb_sub_exc): 
            
            Delta1[i][k]=deriv1b(F[i],vsec_vec[i],numb_v_TF,h,0)*N[k]/N0_e*p[i][k]/(5e-2) # 0 is taken because TF takes only exc and inhib inputs, without discarding the subpops
            
            '''Be careful here, there might be a problem with the type of numb_sub_(exc/inh) (int/float)'''
                
        for k in range(numb_sub_inh):
            
            Delta1[i][k+numb_sub_exc]=deriv1b(F[i],vsec_vec[i],numb_v_TF,h,1)*N[k+numb_sub_exc]/N0_i*p[i][k+numb_sub_exc]/(5e-2)

        for k in range(numb_sub_exc):
            for j in range(numb_sub_exc):

                Delta2[i][k][j]=deriv2b(F[i],vsec_vec[i],numb_v_TF,h,0,0)*\
                N[k]/N0_e*N[j]/N0_e*p[i][k]/(5e-2)*p[i][j]/(5e-2)
                
            for j in range(numb_sub_inh):
                Delta2[i][k][j+numb_sub_exc]=deriv2b(F[i],vsec_vec[i],numb_v_TF,h,0,1)*\
                N[k]/N0_e*N[j+numb_sub_exc]/N0_i*p[i][k]/(5e-2)*p[i][j+numb_sub_exc]/(5e-2)
                
        for k in range(numb_sub_inh):
            for j in range(numb_sub_exc):
                    Delta2[i][k+numb_sub_exc][j]=deriv2b(F[i],vsec_vec[i],numb_v_TF,h,1,0)*\
                    N[k+numb_sub_exc]/N0_i*N[j]/N0_e*p[i][k+numb_sub_exc]/(5e-2)*p[i][j]/(5e-2)
            for j in range(numb_sub_inh):
                Delta2[i][k+numb_sub_exc][j+numb_sub_exc]=deriv2b(F[i],vsec_vec[i],numb_v_TF,h,1,1)*\
                N[k+numb_sub_exc]/N0_i*N[j+numb_sub_exc]/N0_i*p[i][k+numb_sub_exc]/(5e-2)*\
                p[i][j+numb_sub_exc]/(5e-2)
                #print(Delta2[i][k][j])
                
       
    for i in range(numb_var):
        deriv[i]=(Eval_TF[i]-x0[i])/T

        #deriv[numb_var*i+i+numb_var]+=(Eval_TF[i]*(1./T-Eval_TF[i])/N[i])/T
        for j in range(numb_var):
            index=numb_var*i+j+numb_var # define here the location of the considered c variable and its derivative
            for k in range(numb_var): 
                #deriv[i]+=(0.5*Delta2[i][j][k]*x0[numb_var*j+k+numb_var])/T
                index_ik=numb_var*i+k+numb_var
                index_jk=numb_var*j+k+numb_var
                #deriv[index]+=(x0[index_jk]*Delta1[i][k]+x0[index_ik]*Delta1[j][k])/T
                
            #deriv[index]+=((Eval_TF[i]-x0[i])*(Eval_TF[j]-x0[j])-2*x0[index])/T
  
                
    for i in range(numb_adapt):
        deriv[numb_var+i]=-x0[numb_var+i]/tauw[i]+b[i]*x0[i]+a[i]*(mu(x0,params,i,t)-El[i])/tauw[i]
        
      # DEFINE MU !!!!#
    
    
    ''' Printing to debug '''
    
    
    #print('\n','\n','TIME = ', t, '\n','\n', 'Delta1 = ', Delta1,'\n','Delta1 Old = ', Delta1_old, '\n', 'Delta2 = ', Delta2,'\n', 'Delta2 Old= ', Delta2_old,'\n')
                
    return deriv,Eval_TF

def MeanField_dN_dp_FO_NoTF(x0,t,params):
    #print(' t = ',t)
    N0_e=8000
    N0_i=2000
    tstep=params['tstep']
    tot_numb=params['numb_var']
    tauw=params['tauw']
    a=params['a']
    b=params['b']
    El=params['El']
    N=params['size_subpop']
    numb_var=params['numb_pop']
    numb_adapt=params['numb_adapt']
    F=params['TF']
    h=params['h']
    T=params['T']
    mu=params['mu']
    ext_drive=params['ext_inp']
    input_rate=params['inp_funct']
    
    if (input_rate != 0): input_func=input_rate(t, *params['inp_par'])
    else : input_func=0
 
    
    ''' We keep the original TF framework'''
    
    numb_v_TF=int(2)
    numb_w_TF=int(1)
    
    '''Define the number of subpopulations, they are assumed to be evenly distributed'''
    
    numb_sub_exc=params['numb_sub_exc']
    numb_sub_inh=params['numb_sub_inh']
 
    
    Ntot_exc=0
    Ntot_inh=0
    
    for i in range(numb_sub_exc): Ntot_exc+=N[i]
    for i in range(numb_sub_inh): Ntot_inh+=N[i+numb_sub_exc]
    
    deriv=np.zeros(tot_numb)
    
    ''' Define vsec_vec, because of the way TF are defined and the different arguments they take 
    (adaptive versus non adaptive populations)'''
    
    vsec_vec=np.zeros((numb_var,numb_v_TF+numb_w_TF))  # Needed to pass different values to TF1 and TF2
    
    ''' Define dp and dN to account for variability in population size and connectivity '''
   
    p=params['p_conn']
    #dN=[Ntot_exc/N0_e,Ntot_inh/N0_i] # Ntot is taken because it is averaged over populations
   
   
   #for i in range(numb_sub_exc): dN.append(1+(Ntot_exc-N0_e)/N0_e) # Ntot_exc is taken because it is averaged over populations
   #for i in range(numb_sub_inh): dN.append(1+(Ntot_inh-N0_i)/N0_i)
    
    
    for i in range(numb_var):
        for j in range(numb_sub_exc): ## excitatory input
            vsec_vec[i][0]+=x0[j]*N[j]/N0_e*p[i][j]/(5e-2) # initially only divide by numb_sub_exc : true if populations are evenly distributed ! => /2= * 4000/8000
        
        ''' dividing by Ntot_exc is for averaging over subpops,  (1+dN[i]) for accounting for difference
        in initial population size w.r.t TF calculation '''
        
        for j in range(numb_sub_inh): ## inhibitory input
            vsec_vec[i][1]+=x0[j+numb_sub_exc]*N[j+numb_sub_exc]/N0_i*p[i][j+numb_sub_exc]/(5e-2)
        
    
    for i in range(numb_adapt):
        vsec_vec[i][numb_v_TF]=x0[numb_var+i]
    
    '''CAREFUL : the above assumes that all adaptive variables are put first'''        
    
    p_pois=params['p_pois']
    
    
    for i in range(numb_var): ## external drive
        vsec_vec[i][0]+=(ext_drive[i]+input_func)*p_pois[i]/(5e-2)*Ntot_exc/N0_e#+input_rate(t, *params[23]) #+input_rate(t, 2., 0.1, 0.1, 120, 1) #+input_rate(t,0.1,0.005,0.02,10)
    
    '''dN[0] is taken because generally Poisson input are taken with same pop size as excitatory neurons'''

    
    # Define the TF derivatives here to compute them only once #
    
    Delta1=np.zeros((numb_var,numb_var))
    Delta2=np.zeros((numb_var,numb_var,numb_var))
    
    #Delta1_old=np.zeros((numb_var,numb_var))
    #Delta2_old=np.zeros((numb_var,numb_var,numb_var))
    
    Eval_TF=np.zeros(numb_var)

    
    for i in range(numb_var):
        Eval_TF[i]=F[i](*vsec_vec[i])
        #print('evaluated TF',i,' = ', Eval_TF[i])

        for k in range(numb_sub_exc): 
            
            Delta1[i][k]=deriv1b(F[i],vsec_vec[i],numb_v_TF,h,0)*N[k]/N0_e*p[i][k]/(5e-2) # 0 is taken because TF takes only exc and inhib inputs, without discarding the subpops
            
            '''Be careful here, there might be a problem with the type of numb_sub_(exc/inh) (int/float)'''
                
        for k in range(numb_sub_inh):
            
            Delta1[i][k+numb_sub_exc]=deriv1b(F[i],vsec_vec[i],numb_v_TF,h,1)*N[k+numb_sub_exc]/N0_i*p[i][k+numb_sub_exc]/(5e-2)

        for k in range(numb_sub_exc):
            for j in range(numb_sub_exc):

                Delta2[i][k][j]=deriv2b(F[i],vsec_vec[i],numb_v_TF,h,0,0)*\
                N[k]/N0_e*N[j]/N0_e*p[i][k]/(5e-2)*p[i][j]/(5e-2)
                
            for j in range(numb_sub_inh):
                Delta2[i][k][j+numb_sub_exc]=deriv2b(F[i],vsec_vec[i],numb_v_TF,h,0,1)*\
                N[k]/N0_e*N[j+numb_sub_exc]/N0_i*p[i][k]/(5e-2)*p[i][j+numb_sub_exc]/(5e-2)
                
        for k in range(numb_sub_inh):
            for j in range(numb_sub_exc):
                    Delta2[i][k+numb_sub_exc][j]=deriv2b(F[i],vsec_vec[i],numb_v_TF,h,1,0)*\
                    N[k+numb_sub_exc]/N0_i*N[j]/N0_e*p[i][k+numb_sub_exc]/(5e-2)*p[i][j]/(5e-2)
            for j in range(numb_sub_inh):
                Delta2[i][k+numb_sub_exc][j+numb_sub_exc]=deriv2b(F[i],vsec_vec[i],numb_v_TF,h,1,1)*\
                N[k+numb_sub_exc]/N0_i*N[j+numb_sub_exc]/N0_i*p[i][k+numb_sub_exc]/(5e-2)*\
                p[i][j+numb_sub_exc]/(5e-2)
                #print(Delta2[i][k][j])
                
       
    for i in range(numb_var):
        deriv[i]=(Eval_TF[i]-x0[i])/T

        #deriv[numb_var*i+i+numb_var]+=(Eval_TF[i]*(1./T-Eval_TF[i])/N[i])/T
        for j in range(numb_var):
            index=numb_var*i+j+numb_var # define here the location of the considered c variable and its derivative
            for k in range(numb_var): 
                #deriv[i]+=(0.5*Delta2[i][j][k]*x0[numb_var*j+k+numb_var])/T
                index_ik=numb_var*i+k+numb_var
                index_jk=numb_var*j+k+numb_var
                #deriv[index]+=(x0[index_jk]*Delta1[i][k]+x0[index_ik]*Delta1[j][k])/T
                
            #deriv[index]+=((Eval_TF[i]-x0[i])*(Eval_TF[j]-x0[j])-2*x0[index])/T
  
                
    for i in range(numb_adapt):
        deriv[numb_var+i]=-x0[numb_var+i]/tauw[i]+b[i]*x0[i]+a[i]*(mu(x0,params,i,t)-El[i])/tauw[i]
        
      # DEFINE MU !!!!#
    
    
    ''' Printing to debug '''
    
    
    #print('\n','\n','TIME = ', t, '\n','\n', 'Delta1 = ', Delta1,'\n','Delta1 Old = ', Delta1_old, '\n', 'Delta2 = ', Delta2,'\n', 'Delta2 Old= ', Delta2_old,'\n')
                
    return deriv

def MeanField_dN_dp_CADEX(x0,t,params):
    #print(' t = ',t)
    N0_e=8000 # this shall always remain unchanged !
    N0_i=2000
    tstep=params['tstep']
    tot_numb=params['numb_var']
    El=params['El']
    N=params['size_subpop']
    numb_var=params['numb_pop']
    numb_adapt=params['numb_adapt']
    F=params['TF']
    h=params['h']
    T=params['T']
    mu=params['mu']
    ext_drive=params['ext_inp']
    input_rate=params['inp_funct']
    
    Ea=params['Ea']
    ga_bar=params['ga_bar']
    dga=params['dga']
    tau_a=params['tau_a']
    Delta_a=params['delta_a']
    Va=params['Va']
    
    
    if (input_rate != 0): input_func=input_rate(t, *params['inp_par'])
    else : input_func=0
 
    
    ''' We keep the original TF framework'''
    
    numb_v_TF=int(2)
    numb_w_TF=int(1)
    
    '''Define the number of subpopulations, they are assumed to be evenly distributed'''
    
    numb_sub_exc=params['numb_sub_exc']
    numb_sub_inh=params['numb_sub_inh']
    
    
    Ntot_exc=0
    Ntot_inh=0
    
    for i in range(numb_sub_exc): Ntot_exc+=N[i]
    for i in range(numb_sub_inh): Ntot_inh+=N[i+numb_sub_exc]
    
    deriv=np.zeros(tot_numb)
    
    ''' Define vsec_vec, because of the way TF are defined and the different arguments they take 
    (adaptive versus non adaptive populations)'''
    
    vsec_vec=np.zeros((numb_var,numb_v_TF+numb_w_TF))  # Needed to pass different values to TF1 and TF2
    
    ''' Define dp and dN to account for variability in population size and connectivity '''
   
    p=params['p_conn']    #dN=[Ntot_exc/N0_e,Ntot_inh/N0_i] # Ntot is taken because it is averaged over populations
   
   
   #for i in range(numb_sub_exc): dN.append(1+(Ntot_exc-N0_e)/N0_e) # Ntot_exc is taken because it is averaged over populations
   #for i in range(numb_sub_inh): dN.append(1+(Ntot_inh-N0_i)/N0_i)
    
    
    for i in range(numb_var):
        for j in range(numb_sub_exc): ## excitatory input
            vsec_vec[i][0]+=x0[j]*N[j]/N0_e*p[i][j]/(5e-2) # initially only divide by numb_sub_exc : true if populations are evenly distributed ! => /2= * 4000/8000
        
        ''' dividing by Ntot_exc is for averaging over subpops,  (1+dN[i]) for accounting for difference
        in initial population size w.r.t TF calculation '''
        
        for j in range(numb_sub_inh): ## inhibitory input
            vsec_vec[i][1]+=x0[j+numb_sub_exc]*N[j+numb_sub_exc]/N0_i*p[i][j+numb_sub_exc]/(5e-2)
        
    
    for i in range(numb_adapt):
        vsec_vec[i][numb_v_TF]=x0[numb_var*numb_var+numb_var+i]
    
    '''CAREFUL : the above assumes that all adaptive variables are put first'''        
    
    p_pois=params['p_pois']
    
    for i in range(numb_var): ## external drive
        vsec_vec[i][0]+=(ext_drive+input_func)*p_pois[i]/(5e-2)*Ntot_exc/N0_e#+input_rate(t, *params[23]) #+input_rate(t, 2., 0.1, 0.1, 120, 1) #+input_rate(t,0.1,0.005,0.02,10)
    
    '''dN[0] is taken because generally Poisson input are taken with same pop size as excitatory neurons'''

    
    # Define the TF derivatives here to compute them only once #
    
    Delta1=np.zeros((numb_var,numb_var))
    Delta2=np.zeros((numb_var,numb_var,numb_var))
    
    #Delta1_old=np.zeros((numb_var,numb_var))
    #Delta2_old=np.zeros((numb_var,numb_var,numb_var))
    
    Eval_TF=np.zeros(numb_var)

    
    for i in range(numb_var):
        Eval_TF[i]=F[i](*vsec_vec[i])
        #print('evaluated TF',i,' = ', Eval_TF[i])

        for k in range(numb_sub_exc): 
            
            Delta1[i][k]=deriv1b(F[i],vsec_vec[i],numb_v_TF,h,0)*N[k]/N0_e*p[i][k]/(5e-2) # 0 is taken because TF takes only exc and inhib inputs, without discarding the subpops
            
            '''Be careful here, there might be a problem with the type of numb_sub_(exc/inh) (int/float)'''
                
        for k in range(numb_sub_inh):
            
            Delta1[i][k+numb_sub_exc]=deriv1b(F[i],vsec_vec[i],numb_v_TF,h,1)*N[k+numb_sub_exc]/N0_i*p[i][k+numb_sub_exc]/(5e-2)

        for k in range(numb_sub_exc):
            for j in range(numb_sub_exc):

                Delta2[i][k][j]=deriv2b(F[i],vsec_vec[i],numb_v_TF,h,0,0)*\
                N[k]/N0_e*N[j]/N0_e*p[i][k]/(5e-2)*p[i][j]/(5e-2)
                
            for j in range(numb_sub_inh):
                Delta2[i][k][j+numb_sub_exc]=deriv2b(F[i],vsec_vec[i],numb_v_TF,h,0,1)*\
                N[k]/N0_e*N[j+numb_sub_exc]/N0_i*p[i][k]/(5e-2)*p[i][j+numb_sub_exc]/(5e-2)
                
        for k in range(numb_sub_inh):
            for j in range(numb_sub_exc):
                    Delta2[i][k+numb_sub_exc][j]=deriv2b(F[i],vsec_vec[i],numb_v_TF,h,1,0)*\
                    N[k+numb_sub_exc]/N0_i*N[j]/N0_e*p[i][k+numb_sub_exc]/(5e-2)*p[i][j]/(5e-2)
            for j in range(numb_sub_inh):
                Delta2[i][k+numb_sub_exc][j+numb_sub_exc]=deriv2b(F[i],vsec_vec[i],numb_v_TF,h,1,1)*\
                N[k+numb_sub_exc]/N0_i*N[j+numb_sub_exc]/N0_i*p[i][k+numb_sub_exc]/(5e-2)*\
                p[i][j+numb_sub_exc]/(5e-2)
                #print(Delta2[i][k][j])
                
       
    for i in range(numb_var):
        deriv[i]=(Eval_TF[i]-x0[i])/T

        deriv[numb_var*i+i+numb_var]+=(Eval_TF[i]*(1./T-Eval_TF[i])/N[i])/T
        for j in range(numb_var):
            index=numb_var*i+j+numb_var # define here the location of the considered c variable and its derivative
            for k in range(numb_var): 
                deriv[i]+=(0.5*Delta2[i][j][k]*x0[numb_var*j+k+numb_var])/T
                index_ik=numb_var*i+k+numb_var
                index_jk=numb_var*j+k+numb_var
                deriv[index]+=(x0[index_jk]*Delta1[i][k]+x0[index_ik]*Delta1[j][k])/T
                
            deriv[index]+=((Eval_TF[i]-x0[i])*(Eval_TF[j]-x0[j])-2*x0[index])/T
  
                
    for i in range(numb_adapt):
        deriv[numb_var**2+numb_var+i]=dga[i]*x0[i]+(-x0[numb_var**2+numb_var+i]+ga_bar[i]/(1+math.exp((Va[i]-mu(x0,params,i,t))/Delta_a[i])))/tau_a[i]
        
      # DEFINE MU !!!!#
    
    
    ''' Printing to debug '''
    
    
    #print('\n','\n','TIME = ', t, '\n','\n', 'Delta1 = ', Delta1,'\n','Delta1 Old = ', Delta1_old, '\n', 'Delta2 = ', Delta2,'\n', 'Delta2 Old= ', Delta2_old,'\n')
                
    return deriv,Eval_TF


def MeanField_dN_dp_CADEX_FO(x0,t,params):
    #print(' t = ',t)
    N0_e=8000 # this shall always remain unchanged !
    N0_i=2000
    tstep=params['tstep']
    tot_numb=params['numb_var']
    El=params['El']
    N=params['size_subpop']
    numb_var=params['numb_pop']
    numb_adapt=params['numb_adapt']
    F=params['TF']
    h=params['h']
    T=params['T']
    mu=params['mu']
    ext_drive=params['ext_inp']
    input_rate=params['inp_funct']
    
    Ea=params['Ea']
    ga_bar=params['ga_bar']
    dga=params['dga']
    tau_a=params['tau_a']
    Delta_a=params['delta_a']
    Va=params['Va']
    
    if (input_rate != 0): input_func=input_rate(t, *params['inp_par'])
    else : input_func=0
 
    
    ''' We keep the original TF framework'''
    
    numb_v_TF=int(2)
    numb_w_TF=int(1)
    
    '''Define the number of subpopulations, they are assumed to be evenly distributed'''
    
    numb_sub_exc=params['numb_sub_exc']
    numb_sub_inh=params['numb_sub_inh']
    
    
    Ntot_exc=0
    Ntot_inh=0
    
    for i in range(numb_sub_exc): Ntot_exc+=N[i]
    for i in range(numb_sub_inh): Ntot_inh+=N[i+numb_sub_exc]
    
    deriv=np.zeros(tot_numb)
    
    ''' Define vsec_vec, because of the way TF are defined and the different arguments they take 
    (adaptive versus non adaptive populations)'''
    
    vsec_vec=np.zeros((numb_var,numb_v_TF+numb_w_TF))  # Needed to pass different values to TF1 and TF2
    
    ''' Define dp and dN to account for variability in population size and connectivity '''
   
    p=params['p_conn']    #dN=[Ntot_exc/N0_e,Ntot_inh/N0_i] # Ntot is taken because it is averaged over populations
   
   
   #for i in range(numb_sub_exc): dN.append(1+(Ntot_exc-N0_e)/N0_e) # Ntot_exc is taken because it is averaged over populations
   #for i in range(numb_sub_inh): dN.append(1+(Ntot_inh-N0_i)/N0_i)
    
    
    for i in range(numb_var):
        for j in range(numb_sub_exc): ## excitatory input
            vsec_vec[i][0]+=x0[j]*N[j]/N0_e*p[i][j]/(5e-2) # initially only divide by numb_sub_exc : true if populations are evenly distributed ! => /2= * 4000/8000
        
        ''' dividing by Ntot_exc is for averaging over subpops,  (1+dN[i]) for accounting for difference
        in initial population size w.r.t TF calculation '''
        
        for j in range(numb_sub_inh): ## inhibitory input
            vsec_vec[i][1]+=x0[j+numb_sub_exc]*N[j+numb_sub_exc]/N0_i*p[i][j+numb_sub_exc]/(5e-2)
        
    
    for i in range(numb_adapt):
        vsec_vec[i][numb_v_TF]=x0[numb_var+i]
    
    '''CAREFUL : the above assumes that all adaptive variables are put first'''        
    
    p_pois=params['p_pois']
    
    
    for i in range(numb_var): ## external drive
        vsec_vec[i][0]+=(ext_drive+input_func)*p_pois[i]/(5e-2)*Ntot_exc/N0_e#+input_rate(t, *params[23]) #+input_rate(t, 2., 0.1, 0.1, 120, 1) #+input_rate(t,0.1,0.005,0.02,10)
    
    '''dN[0] is taken because generally Poisson input are taken with same pop size as excitatory neurons'''

    
    # Define the TF derivatives here to compute them only once #
    
    Delta1=np.zeros((numb_var,numb_var))
    Delta2=np.zeros((numb_var,numb_var,numb_var))
    
    #Delta1_old=np.zeros((numb_var,numb_var))
    #Delta2_old=np.zeros((numb_var,numb_var,numb_var))
    
    Eval_TF=np.zeros(numb_var)

    
    for i in range(numb_var):
        Eval_TF[i]=F[i](*vsec_vec[i])
        #print('evaluated TF',i,' = ', Eval_TF[i])

        for k in range(numb_sub_exc): 
            
            Delta1[i][k]=deriv1b(F[i],vsec_vec[i],numb_v_TF,h,0)*N[k]/N0_e*p[i][k]/(5e-2) # 0 is taken because TF takes only exc and inhib inputs, without discarding the subpops
            
            '''Be careful here, there might be a problem with the type of numb_sub_(exc/inh) (int/float)'''
                
        for k in range(numb_sub_inh):
            
            Delta1[i][k+numb_sub_exc]=deriv1b(F[i],vsec_vec[i],numb_v_TF,h,1)*N[k+numb_sub_exc]/N0_i*p[i][k+numb_sub_exc]/(5e-2)

        for k in range(numb_sub_exc):
            for j in range(numb_sub_exc):

                Delta2[i][k][j]=deriv2b(F[i],vsec_vec[i],numb_v_TF,h,0,0)*\
                N[k]/N0_e*N[j]/N0_e*p[i][k]/(5e-2)*p[i][j]/(5e-2)
                
            for j in range(numb_sub_inh):
                Delta2[i][k][j+numb_sub_exc]=deriv2b(F[i],vsec_vec[i],numb_v_TF,h,0,1)*\
                N[k]/N0_e*N[j+numb_sub_exc]/N0_i*p[i][k]/(5e-2)*p[i][j+numb_sub_exc]/(5e-2)
                
        for k in range(numb_sub_inh):
            for j in range(numb_sub_exc):
                    Delta2[i][k+numb_sub_exc][j]=deriv2b(F[i],vsec_vec[i],numb_v_TF,h,1,0)*\
                    N[k+numb_sub_exc]/N0_i*N[j]/N0_e*p[i][k+numb_sub_exc]/(5e-2)*p[i][j]/(5e-2)
            for j in range(numb_sub_inh):
                Delta2[i][k+numb_sub_exc][j+numb_sub_exc]=deriv2b(F[i],vsec_vec[i],numb_v_TF,h,1,1)*\
                N[k+numb_sub_exc]/N0_i*N[j+numb_sub_exc]/N0_i*p[i][k+numb_sub_exc]/(5e-2)*\
                p[i][j+numb_sub_exc]/(5e-2)
                #print(Delta2[i][k][j])
                
       
    for i in range(numb_var):
        deriv[i]=(Eval_TF[i]-x0[i])/T

        #deriv[numb_var*i+i+numb_var]+=(Eval_TF[i]*(1./T-Eval_TF[i])/N[i])/T
        for j in range(numb_var):
            index=numb_var*i+j+numb_var # define here the location of the considered c variable and its derivative
            for k in range(numb_var): 
                #deriv[i]+=(0.5*Delta2[i][j][k]*x0[numb_var*j+k+numb_var])/T
                index_ik=numb_var*i+k+numb_var
                index_jk=numb_var*j+k+numb_var
                #deriv[index]+=(x0[index_jk]*Delta1[i][k]+x0[index_ik]*Delta1[j][k])/T
                
            #deriv[index]+=((Eval_TF[i]-x0[i])*(Eval_TF[j]-x0[j])-2*x0[index])/T
  
                
    for i in range(numb_adapt):
        deriv[numb_var+i]=dga[i]*x0[i]+(-x0[numb_var+i]+ga_bar[i]/(1+math.exp((Va[i]-mu(x0,params,i,t))/Delta_a[i])))/tau_a[i] 
           
    return deriv,Eval_TF




def MeanField_multi_dp_TEST(x0,t,params):
    

    #print(' t = ',t)
    N0_e=8000 # this shall always remain unchanged !
    N0_i=2000
    tstep=params[0]
    tot_numb=params[1]
    tauw=params[2]
    a=params[3]
    b=params[4]
    El=params[5]
    N=params[6]
    numb_var=params[7]
    numb_adapt=params[8]
    F=params[9]
    h=params[10]
    T=params[11]
    mu=params[20]
    ext_drive=params[21]
    input_rate=params[22]
    
    if (params[22] != 0): input_func=params[22](t, *params[23])
    else : input_func=0
 
    
    ''' We keep the original TF framework'''
    
    numb_v_TF=int(2)
    numb_w_TF=int(1)
    
    '''Define the number of subpopulations, they are assumed to be evenly distributed'''
    
    numb_sub_exc=params[24]
    numb_sub_inh=params[25]
    
    
    Ntot_exc=N0_e
    Ntot_inh=N0_i
    
    #for i in range(numb_sub_exc): Ntot_exc+=N[i]
    #for i in range(numb_sub_inh): Ntot_inh+=N[i+numb_sub_exc]
    
    deriv=np.zeros(tot_numb)
    
    ''' Define vsec_vec, because of the way TF are defined and the different arguments they take 
    (adaptive versus non adaptive populations)'''
    
    vsec_vec=np.zeros((numb_var,numb_v_TF+numb_w_TF))  # Needed to pass different values to TF1 and TF2
    
    ''' Define dp and dN to account for variability in population size and connectivity '''
   
    p=params[26]
    dp_e=params[27][0] ## dp measures the difference in p for subpopulations => discretization in p_e
    dp_i=params[27][1]
    
    #dN=[Ntot_exc/N0_e,Ntot_inh/N0_i] # Ntot is taken because it is averaged over populations
   
   
   #for i in range(numb_sub_exc): dN.append(1+(Ntot_exc-N0_e)/N0_e) # Ntot_exc is taken because it is averaged over populations
   #for i in range(numb_sub_inh): dN.append(1+(Ntot_inh-N0_i)/N0_i)
   
   
  
    
    
    ''' Let's first normalize the total probabilities, otherwise the averaging makes no sense
        Thus what we want to achieve here is keeping the right proportions, from a real gaussian to a discrete
        n_d mesh'''
    
    prob_tot_e=np.zeros(numb_var)
    prob_loc_e=params[29][0]
    
    prob_tot_i=np.zeros(numb_var)
    prob_loc_i=params[29][1]
    
    #print('\n','\n')
    for i in range(numb_var):
        for j in range(numb_sub_exc): ## excitatory input
            #prob_loc[i][j]=gaussian(p[i][j]+dp[j],p[i][j],math.sqrt(p[i][j]/N[j])) ## dividing by N[j] because std of # is multiplied by N[j], and it needs to be normalized for p
            prob_tot_e[i]+=prob_loc_e[i][j]
            #print(i,j,'   ', prob_tot_e[i], '   ', prob_loc_e[i][j])
        for j in range(numb_sub_inh): ## excitatory input
            #prob_loc[i][j]=gaussian(p[i][j]+dp[j],p[i][j],math.sqrt(p[i][j]/N[j])) ## dividing by N[j] because std of # is multiplied by N[j], and it needs to be normalized for p
            prob_tot_i[i]+=prob_loc_i[i][j]
            
    x_e=np.zeros(numb_var)
    x_i=np.zeros(numb_var)
    
    for i in range(numb_var):
        for j in range(numb_sub_exc): ## excitatory input
            x_e[i]+=x0[j]*prob_loc_e[i][j]/prob_tot_e[i]# First average over all subspops
        
        #for j in range(numb_sub_exc): # only then, reinject in inputs accounting for different p
        vsec_vec[i][0]=x_e[i]*(p[i][0]+dp_e[i])/(5e-2) # [i][0] is taken for the moment because each subpop is supposed to have same p
        
        ''' dividing by Ntot_exc is for averaging over subpops,  (1+dN[i]) for accounting for difference
        in initial population size w.r.t TF calculation
        The gaussian multiplicative factor has average p and stdev sqrt(Np) 
        Here the philosophy is different : before, every input was weighted by its relative population size
        But now, they are pooled together in an average overall frequency'''
        
        
        for j in range(numb_sub_inh): ## excitatory input
            x_i[i]+=x0[j+numb_sub_exc]*prob_loc_i[i][j]/prob_tot_i[i]# First average over all subspops
            
        #for j in range(numb_sub_inh): ## inhibitory input
            #vsec_vec[i][1]+=x0[j+numb_sub_exc]*p[i][j+numb_sub_exc]/(5e-2)
        
        vsec_vec[i][1]=x_i[i]*(p[i][numb_sub_exc]+dp_i[i])/(5e-2)
    
    for i in range(numb_adapt):
        vsec_vec[i][numb_v_TF]=x0[numb_var*numb_var+numb_var+i]
    
    '''CAREFUL : the above assumes that all adaptive variables are put first'''        
    
    p_pois=params[28]
    
    
    for i in range(numb_var): ## external drive
        vsec_vec[i][0]+=(ext_drive+input_func)*p_pois[i]/(5e-2)#+input_rate(t, *params[23]) #+input_rate(t, 2., 0.1, 0.1, 120, 1) #+input_rate(t,0.1,0.005,0.02,10)
    
    '''dN[0] is taken because generally Poisson input are taken with same pop size as excitatory neurons'''

    
    # Define the TF derivatives here to compute them only once #
    
    Delta1=np.zeros((numb_var,numb_var))
    Delta2=np.zeros((numb_var,numb_var,numb_var))
    
    #Delta1_old=np.zeros((numb_var,numb_var))
    #Delta2_old=np.zeros((numb_var,numb_var,numb_var))
    
    Eval_TF=np.zeros(numb_var)

    
    for i in range(numb_var):
        Eval_TF[i]=F[i](*vsec_vec[i])
        #print('evaluated TF',i,' = ', Eval_TF[i])

        for k in range(numb_sub_exc): 
            
            Delta1[i][k]=deriv1b(F[i],vsec_vec[i],numb_v_TF,h,0)*(p[i][k]+dp_e[i])/(5e-2)*prob_loc_e[i][k]/prob_tot_e[i] 
            # 0 is taken because TF takes only exc and inhib inputs, without discarding the subpops
            
            '''KEEP IN MIND HERE THAT SO FAR, EVERYTHING IS SYMMETRIC => THE p[i][j] are superfluous '''
                
        for k in range(numb_sub_inh):
            
            Delta1[i][k+numb_sub_exc]=deriv1b(F[i],vsec_vec[i],numb_v_TF,h,1)*N[k+numb_sub_exc]/N0_i*(p[i][k+numb_sub_exc]+dp_i[i])/(5e-2)*\
            prob_loc_i[i][k]/prob_tot_i[i]

        for k in range(numb_sub_exc):
            for j in range(numb_sub_exc):

                Delta2[i][k][j]=deriv2b(F[i],vsec_vec[i],numb_v_TF,h,0,0)*\
                N[k]/N0_e*N[j]/N0_e*(p[i][k]+dp_e[i])/(5e-2)*(p[i][j]+dp_e[i])/(5e-2)*\
                prob_loc_e[i][k]/prob_tot_e[i]*prob_loc_e[i][j]/prob_tot_e[i]
                
            for j in range(numb_sub_inh):
                Delta2[i][k][j+numb_sub_exc]=deriv2b(F[i],vsec_vec[i],numb_v_TF,h,0,1)*\
                N[k]/N0_e*N[j+numb_sub_exc]/N0_i*(p[i][k]+dp_e[i])/(5e-2)*(p[i][j+numb_sub_exc]+dp_i[i])/(5e-2)*\
                prob_loc_e[i][k]/prob_tot_e[i]*prob_loc_i[i][j]/prob_tot_i[i]
                
        for k in range(numb_sub_inh):
            for j in range(numb_sub_exc):
                    Delta2[i][k+numb_sub_exc][j]=deriv2b(F[i],vsec_vec[i],numb_v_TF,h,1,0)*\
                    N[k+numb_sub_exc]/N0_i*N[j]/N0_e*(p[i][k+numb_sub_exc]+dp_i[i])/(5e-2)*(p[i][j]+dp_e[i])/(5e-2)*\
                    prob_loc_e[i][j]/prob_tot_e[i]*prob_loc_i[i][k]/prob_tot_i[i]
            for j in range(numb_sub_inh):
                Delta2[i][k+numb_sub_exc][j+numb_sub_exc]=deriv2b(F[i],vsec_vec[i],numb_v_TF,h,1,1)*\
                N[k+numb_sub_exc]/N0_i*N[j+numb_sub_exc]/N0_i*(p[i][k+numb_sub_exc]+dp_i[i])/(5e-2)*\
                (p[i][j+numb_sub_exc]+dp_i[i])/(5e-2)*prob_loc_i[i][k]/prob_tot_i[i]*prob_loc_i[i][j]/prob_tot_i[i]
                #print(Delta2[i][k][j])
                
       
    for i in range(numb_var):
        deriv[i]=(Eval_TF[i]-x0[i])/T

        deriv[numb_var*i+i+numb_var]+=(Eval_TF[i]*(1./T-Eval_TF[i])/N[i])/T
        for j in range(numb_var):
            index=numb_var*i+j+numb_var # define here the location of the considered c variable and its derivative
            for k in range(numb_var): 
                deriv[i]+=(0.5*Delta2[i][j][k]*x0[numb_var*j+k+numb_var])/T
                index_ik=numb_var*i+k+numb_var
                index_jk=numb_var*j+k+numb_var
                deriv[index]+=(x0[index_jk]*Delta1[i][k]+x0[index_ik]*Delta1[j][k])/T
                
            deriv[index]+=((Eval_TF[i]-x0[i])*(Eval_TF[j]-x0[j])-2*x0[index])/T
  
                
    for i in range(numb_adapt):
        deriv[numb_var**2+numb_var+i]=-x0[numb_var**2+numb_var+i]/tauw[i]+b[i]*x0[i]+a[i]*(mu(x0,params,i,t)-El[i])/tauw[i]
        
      # DEFINE MU !!!!#
    
    
    ''' Printing to debug '''
    
    
    #print('\n','\n','TIME = ', t, '\n','\n', 'Delta1 = ', Delta1,'\n','Delta1 Old = ', Delta1_old, '\n', 'Delta2 = ', Delta2,'\n', 'Delta2 Old= ', Delta2_old,'\n')
                
    return deriv,Eval_TF


def MeanField_multi_dp_TEST_FO(x0,t,params):
    
    ''' FOR THE MOMENT, CUT AT FIRST ORDER ! 
    
    
    This test only works for three subpop in p_e '''
    #print(' t = ',t)
    
    #print(' t = ',t)
    N0_e=8000 # this shall always remain unchanged !
    N0_i=2000
    tstep=params[0]
    tot_numb=params[1]
    tauw=params[2]
    a=params[3]
    b=params[4]
    El=params[5]
    N=params[6]
    numb_var=params[7]
    numb_adapt=params[8]
    F=params[9]
    h=params[10]
    T=params[11]
    mu=params[20]
    ext_drive=params[21]
    input_rate=params[22]
    
    if (params[22] != 0): input_func=params[22](t, *params[23])
    else : input_func=0
 
    
    ''' We keep the original TF framework'''
    
    numb_v_TF=int(2)
    numb_w_TF=int(1)
    
    '''Define the number of subpopulations, they are assumed to be evenly distributed'''
    
    numb_sub_exc=params[24]
    numb_sub_inh=params[25]
    
    
    Ntot_exc=N0_e
    Ntot_inh=N0_i
    
    #for i in range(numb_sub_exc): Ntot_exc+=N[i]
    #for i in range(numb_sub_inh): Ntot_inh+=N[i+numb_sub_exc]
    
    deriv=np.zeros(tot_numb)
    
    ''' Define vsec_vec, because of the way TF are defined and the different arguments they take 
    (adaptive versus non adaptive populations)'''
    
    vsec_vec=np.zeros((numb_var,numb_v_TF+numb_w_TF))  # Needed to pass different values to TF1 and TF2
    
    ''' Define dp and dN to account for variability in population size and connectivity '''
   
    p=params[26]
    dp_e=params[27][0] ## dp measures the difference in p for subpopulations => discretization in p_e
    dp_i=params[27][1]
    
    #dN=[Ntot_exc/N0_e,Ntot_inh/N0_i] # Ntot is taken because it is averaged over populations
   
   
   #for i in range(numb_sub_exc): dN.append(1+(Ntot_exc-N0_e)/N0_e) # Ntot_exc is taken because it is averaged over populations
   #for i in range(numb_sub_inh): dN.append(1+(Ntot_inh-N0_i)/N0_i)
   
   
  
    
    
    ''' Let's first normalize the total probabilities, otherwise the averaging makes no sense
        Thus what we want to achieve here is keeping the right proportions, from a real gaussian to a discrete
        n_d mesh'''
    
    prob_tot_e=np.zeros(numb_var)
    prob_loc_e=params[29][0]
    
    prob_tot_i=np.zeros(numb_var)
    prob_loc_i=params[29][1]
    
    #print('\n','\n')
    for i in range(numb_var):
        for j in range(numb_sub_exc): ## excitatory input
            #prob_loc[i][j]=gaussian(p[i][j]+dp[j],p[i][j],math.sqrt(p[i][j]/N[j])) ## dividing by N[j] because std of # is multiplied by N[j], and it needs to be normalized for p
            prob_tot_e[i]+=prob_loc_e[i][j]
            #print(i,j,'   ', prob_tot_e[i], '   ', prob_loc_e[i][j])
        for j in range(numb_sub_inh): ## excitatory input
            #prob_loc[i][j]=gaussian(p[i][j]+dp[j],p[i][j],math.sqrt(p[i][j]/N[j])) ## dividing by N[j] because std of # is multiplied by N[j], and it needs to be normalized for p
            prob_tot_i[i]+=prob_loc_i[i][j]
            
    x_e=np.zeros(numb_var)
    x_i=np.zeros(numb_var)
    
    for i in range(numb_var):
        for j in range(numb_sub_exc): ## excitatory input
            x_e[i]+=x0[j]*prob_loc_e[i][j]/prob_tot_e[i]# First average over all subspops
        
        #for j in range(numb_sub_exc): # only then, reinject in inputs accounting for different p
        vsec_vec[i][0]=x_e[i]*(p[i][0]+dp_e[i])/(5e-2) # [i][0] is taken for the moment because each subpop is supposed to have same p
        
        ''' dividing by Ntot_exc is for averaging over subpops,  (1+dN[i]) for accounting for difference
        in initial population size w.r.t TF calculation
        The gaussian multiplicative factor has average p and stdev sqrt(Np) 
        Here the philosophy is different : before, every input was weighted by its relative population size
        But now, they are pooled together in an average overall frequency'''
        
        
        for j in range(numb_sub_inh): ## excitatory input
            x_i[i]+=x0[j+numb_sub_exc]*prob_loc_i[i][j]/prob_tot_i[i]# First average over all subspops
            
        #for j in range(numb_sub_inh): ## inhibitory input
            #vsec_vec[i][1]+=x0[j+numb_sub_exc]*p[i][j+numb_sub_exc]/(5e-2)
        
        vsec_vec[i][1]=x_i[i]*(p[i][numb_sub_exc]+dp_i[i])/(5e-2)
    
    for i in range(numb_adapt):
        vsec_vec[i][numb_v_TF]=x0[numb_var+i]
    
    '''CAREFUL : the above assumes that all adaptive variables are put first'''        
    
    p_pois=params[28]
    
    
    for i in range(numb_var): ## external drive
        vsec_vec[i][0]+=(ext_drive+input_func)*p_pois[i]/(5e-2)#+input_rate(t, *params[23]) #+input_rate(t, 2., 0.1, 0.1, 120, 1) #+input_rate(t,0.1,0.005,0.02,10)
    
    '''dN[0] is taken because generally Poisson input are taken with same pop size as excitatory neurons'''

    
    # Define the TF derivatives here to compute them only once #
    
    Delta1=np.zeros((numb_var,numb_var))
    Delta2=np.zeros((numb_var,numb_var,numb_var))
    
    #Delta1_old=np.zeros((numb_var,numb_var))
    #Delta2_old=np.zeros((numb_var,numb_var,numb_var))
    
    Eval_TF=np.zeros(numb_var)

    
    for i in range(numb_var):
        Eval_TF[i]=F[i](*vsec_vec[i])
        #print('evaluated TF',i,' = ', Eval_TF[i])

        for k in range(numb_sub_exc): 
            
            Delta1[i][k]=deriv1b(F[i],vsec_vec[i],numb_v_TF,h,0)*(p[i][k]+dp_e[i])/(5e-2)*prob_loc_e[i][k]/prob_tot_e[i] 
            # 0 is taken because TF takes only exc and inhib inputs, without discarding the subpops
            
            '''KEEP IN MIND HERE THAT SO FAR, EVERYTHING IS SYMMETRIC => THE p[i][j] are superfluous '''
                
        for k in range(numb_sub_inh):
            
            Delta1[i][k+numb_sub_exc]=deriv1b(F[i],vsec_vec[i],numb_v_TF,h,1)*N[k+numb_sub_exc]/N0_i*(p[i][k+numb_sub_exc]+dp_i[i])/(5e-2)*\
            prob_loc_i[i][k]/prob_tot_i[i]

        for k in range(numb_sub_exc):
            for j in range(numb_sub_exc):

                Delta2[i][k][j]=deriv2b(F[i],vsec_vec[i],numb_v_TF,h,0,0)*\
                N[k]/N0_e*N[j]/N0_e*(p[i][k]+dp_e[i])/(5e-2)*(p[i][j]+dp_e[i])/(5e-2)*\
                prob_loc_e[i][k]/prob_tot_e[i]*prob_loc_e[i][j]/prob_tot_e[i]
                
            for j in range(numb_sub_inh):
                Delta2[i][k][j+numb_sub_exc]=deriv2b(F[i],vsec_vec[i],numb_v_TF,h,0,1)*\
                N[k]/N0_e*N[j+numb_sub_exc]/N0_i*(p[i][k]+dp_e[i])/(5e-2)*(p[i][j+numb_sub_exc]+dp_i[i])/(5e-2)*\
                prob_loc_e[i][k]/prob_tot_e[i]*prob_loc_i[i][j]/prob_tot_i[i]
                
        for k in range(numb_sub_inh):
            for j in range(numb_sub_exc):
                    Delta2[i][k+numb_sub_exc][j]=deriv2b(F[i],vsec_vec[i],numb_v_TF,h,1,0)*\
                    N[k+numb_sub_exc]/N0_i*N[j]/N0_e*(p[i][k+numb_sub_exc]+dp_i[i])/(5e-2)*(p[i][j]+dp_e[i])/(5e-2)*\
                    prob_loc_e[i][j]/prob_tot_e[i]*prob_loc_i[i][k]/prob_tot_i[i]
            for j in range(numb_sub_inh):
                Delta2[i][k+numb_sub_exc][j+numb_sub_exc]=deriv2b(F[i],vsec_vec[i],numb_v_TF,h,1,1)*\
                N[k+numb_sub_exc]/N0_i*N[j+numb_sub_exc]/N0_i*(p[i][k+numb_sub_exc]+dp_i[i])/(5e-2)*\
                (p[i][j+numb_sub_exc]+dp_i[i])/(5e-2)*prob_loc_i[i][k]/prob_tot_i[i]*prob_loc_i[i][j]/prob_tot_i[i]
                #print(Delta2[i][k][j])
                
       
    for i in range(numb_var):
        deriv[i]=(Eval_TF[i]-x0[i])/T

        #deriv[numb_var*i+i+numb_var]+=(Eval_TF[i]*(1./T-Eval_TF[i])/N[i])/T
        for j in range(numb_var):
            index=numb_var*i+j+numb_var # define here the location of the considered c variable and its derivative
            for k in range(numb_var): 
                #deriv[i]+=(0.5*Delta2[i][j][k]*x0[numb_var*j+k+numb_var])/T
                index_ik=numb_var*i+k+numb_var
                index_jk=numb_var*j+k+numb_var
                #deriv[index]+=(x0[index_jk]*Delta1[i][k]+x0[index_ik]*Delta1[j][k])/T
                
            #deriv[index]+=((Eval_TF[i]-x0[i])*(Eval_TF[j]-x0[j])-2*x0[index])/T
  
                
    for i in range(numb_adapt):
        deriv[numb_var+i]=-x0[numb_var+i]/tauw[i]+b[i]*x0[i]+a[i]*(mu(x0,params,i,t)-El[i])/tauw[i]
        
      # DEFINE MU !!!!#
    
    
    ''' Printing to debug '''
    
    
    #print('\n','\n','TIME = ', t, '\n','\n', 'Delta1 = ', Delta1,'\n','Delta1 Old = ', Delta1_old, '\n', 'Delta2 = ', Delta2,'\n', 'Delta2 Old= ', Delta2_old,'\n')
                
    return deriv



def MF_dissip_FO(x0,t,params):
    
    N0_e=8000
    N0_i=2000
    tstep=params['tstep']
    tot_numb=params['numb_var']
    tauw=params['tauw']
    a=params['a']
    b=params['b']
    El=params['El']
    N=params['size_subpop']
    numb_var=params['numb_pop']
    numb_adapt=params['numb_adapt']
    F=params['TF']
    h=params['h']
    T=params['T']
    mu=params['mu']
    ext_drive=params['ext_inp']
    input_rate=params['inp_funct']
    
    if (input_rate != 0): input_func=input_rate(t, *params['inp_par'])
    else : input_func=0
 
    
    ''' We keep the original TF framework'''
    
    numb_v_TF=int(2)
    numb_w_TF=int(1)
    
    '''Define the number of subpopulations, they are assumed to be evenly distributed'''
    
    numb_sub_exc=params['numb_sub_exc']
    numb_sub_inh=params['numb_sub_inh']
    
    
    Ntot_exc=0
    Ntot_inh=0
    
    for i in range(numb_sub_exc): Ntot_exc+=N[i]
    for i in range(numb_sub_inh): Ntot_inh+=N[i+numb_sub_exc]
    
    
    ''' Define vsec_vec, because of the way TF are defined and the different arguments they take 
    (adaptive versus non adaptive populations)'''
    
    vsec_vec=np.zeros((numb_var,numb_v_TF+numb_w_TF))  # Needed to pass different values to TF1 and TF2
    
    ''' Define dp and dN to account for variability in population size and connectivity '''
   
    p=params['p_conn']
    #dN=[Ntot_exc/N0_e,Ntot_inh/N0_i] # Ntot is taken because it is averaged over populations
   
   
   #for i in range(numb_sub_exc): dN.append(1+(Ntot_exc-N0_e)/N0_e) # Ntot_exc is taken because it is averaged over populations
   #for i in range(numb_sub_inh): dN.append(1+(Ntot_inh-N0_i)/N0_i)
    
    
    for i in range(numb_var):
        for j in range(numb_sub_exc): ## excitatory input
            vsec_vec[i][0]+=x0[j]*N[j]/N0_e*p[i][j]/(5e-2) # initially only divide by numb_sub_exc : true if populations are evenly distributed ! => /2= * 4000/8000
        
        ''' dividing by Ntot_exc is for averaging over subpops,  (1+dN[i]) for accounting for difference
        in initial population size w.r.t TF calculation '''
        
        for j in range(numb_sub_inh): ## inhibitory input
            vsec_vec[i][1]+=x0[j+numb_sub_exc]*N[j+numb_sub_exc]/N0_i*p[i][j+numb_sub_exc]/(5e-2)
        
    
    for i in range(numb_adapt):
        vsec_vec[i][numb_v_TF]=x0[numb_var+i]
        
    p_pois=params['p_pois']
    
    
    for i in range(numb_var): ## external drive
        vsec_vec[i][0]+=(ext_drive[i]+input_func)*p_pois[i]/(5e-2)*Ntot_exc/N0_e#+input_rate(t, *params[23]) #+input_rate(t, 2., 0.1, 0.1, 120, 1) #+input_rate(t,0.1,0.005,0.02,10)

    
    # Define the TF derivatives here to compute them only once #
    
    Delta1=np.zeros((numb_var,numb_var))
    delta_mu=np.zeros(numb_adapt)
     
    for i in range(numb_var):
       
        for k in range(numb_sub_exc): 
            
            Delta1[i][k]=deriv1b(F[i],vsec_vec[i],numb_v_TF,h,0)*N[k]/N0_e*p[i][k]/(5e-2) # 0 is taken because TF takes only exc and inhib inputs, without discarding the subpops
            
            '''Be careful here, there might be a problem with the type of numb_sub_(exc/inh) (int/float)'''
                
        for k in range(numb_sub_inh):
            
            Delta1[i][k+numb_sub_exc]=deriv1b(F[i],vsec_vec[i],numb_v_TF,h,1)*N[k+numb_sub_exc]/N0_i*p[i][k+numb_sub_exc]/(5e-2)

    for i in range(numb_adapt):
        delta_mu[i]=deriv_w_mu(mu,x0,params,i,t)
    
    ##################################################
            ## NOW LET'S CALCULATE DISSIPATION ##
    #################################################
    
    ''' Also create split dissipation vectors to track each component in time'''
        
    delta_dissip=np.zeros(numb_var)
    w_dissip=np.zeros(numb_adapt)
    
    total_dissip=0
        
    for i in range(numb_var):
        
        total_dissip+=(Delta1[i][i]-1)/T 
        delta_dissip[i]=(Delta1[i][i]-1)/T 
        
    for i in range(numb_adapt):
        
        total_dissip+= -1/tauw[i] +a[i]*delta_mu[i]/tauw[i]
        w_dissip[i]=-1/tauw[i] +a[i]*delta_mu[i]/tauw[i]
        
    
        
    return total_dissip,delta_dissip,w_dissip

def MeanField_dN_dp_delay(x0,t,params):
    #print(' t = ',t)
    N0_e=8000
    N0_i=2000
    tstep=params['tstep']
    tot_numb=params['numb_var']
    tauw=params['tauw']
    a=params['a']
    b=params['b']
    El=params['El']
    N=params['size_subpop']
    numb_var=params['numb_pop']
    numb_adapt=params['numb_adapt']
    F=params['TF']
    h=params['h']
    T=params['T']
    mu=params['mu']
    ext_drive=params['ext_inp']
    #noise=params['noise']
    input_rate=params['inp_funct']
    delays=params['delays']
    x_hist=params['x_hist']

    #for i in range(numb_var):
        #for j in range(numb_var):
           #delays[i][j]=int(delays[i][j]/tstep) ### each time step there is incrementation of the x
		
    meas_time=int(t/tstep)
   
    if (input_rate != 0): input_func=input_rate(t, *params['inp_par'])
    else : input_func=0
 
   
    ''' We keep the original TF framework'''
   
    numb_v_TF=int(2)
    numb_w_TF=int(1)
   
    '''Define the number of subpopulations, they are assumed to be evenly distributed'''
   
    numb_sub_exc=params['numb_sub_exc']
    numb_sub_inh=params['numb_sub_inh']
   
   
    Ntot_exc=0
    Ntot_inh=0
   
    for i in range(numb_sub_exc): Ntot_exc+=N[i]
    for i in range(numb_sub_inh): Ntot_inh+=N[i+numb_sub_exc]
   
    deriv=np.zeros(tot_numb)
   
    ''' Define vsec_vec, because of the way TF are defined and the different arguments they take
    (adaptive versus non adaptive populations)'''
   
    vsec_vec=np.zeros((numb_var,numb_v_TF+numb_w_TF))  # Needed to pass different values to TF1 and TF2
   
    ''' Define dp and dN to account for variability in population size and connectivity '''
   
    p=params['p_conn']
    #dN=[Ntot_exc/N0_e,Ntot_inh/N0_i] # Ntot is taken because it is averaged over populations
   
   
   #for i in range(numb_sub_exc): dN.append(1+(Ntot_exc-N0_e)/N0_e) # Ntot_exc is taken because it is averaged over populations
   #for i in range(numb_sub_inh): dN.append(1+(Ntot_inh-N0_i)/N0_i)
   
   
    for i in range(numb_var):
        for j in range(numb_sub_exc): ## excitatory input
            loc_time=int(meas_time-int(delays[i][j]/tstep))
            #print('\n i = ', i, '  j = ', j, ' loc time = ', loc_time, 'meas_time =', meas_time, 'delay =', delays[i][j])
            vsec_vec[i][0]+=x_hist[j][loc_time]*N[j]/N0_e*p[i][j]/(5e-2)
	     
	    
       
        ''' dividing by Ntot_exc is for averaging over subpops,  (1+dN[i]) for accounting for difference
        in initial population size w.r.t TF calculation '''
       
        for j in range(numb_sub_inh): ## inhibitory input
            loc_time=int(meas_time-int(delays[i][j+numb_sub_exc]/tstep))
            vsec_vec[i][1]+=x_hist[j+numb_sub_exc][loc_time]*N[j+numb_sub_exc]/N0_i*p[i][j+numb_sub_exc]/(5e-2)
       
   
    for i in range(numb_adapt):
        loc_time=int(meas_time-int(delays[i][i]/tstep))
        vsec_vec[i][numb_v_TF]=x_hist[numb_var*numb_var+numb_var+i][loc_time]
   
    '''CAREFUL : the above assumes that all adaptive variables are put first'''        
   
    p_pois=params['p_pois']

    #vsec_vec_first=vsec_vec.copy()
   
   
    for i in range(numb_var): ## external drive
        vsec_vec[i][0]+=(ext_drive[i]+input_func)*p_pois[i]/(5e-2)*Ntot_exc/N0_e#+input_rate(t, *params[23]) #+input_rate(t, 2., 0.1, 0.1, 120, 1) #+input_rate(t,0.1,0.005,0.02,10)
        #vsec_vec_first[i][0]+=(ext_drive[i]+input_func+noise[i])*p_pois[i]/(5e-2)*Ntot_exc/N0_e#+input_rate(t, *params[23]) #+input_rate(t, 2., 0.1, 0.1, 120, 1) #+input_rate(t,0.1,0.005,0.02,10)  
   
    '''dN[0] is taken because generally Poisson input are taken with same pop size as excitatory neurons'''

   
    # Define the TF derivatives here to compute them only once #
   
    Delta1=np.zeros((numb_var,numb_var))
    Delta2=np.zeros((numb_var,numb_var,numb_var))
   
    #Delta1_old=np.zeros((numb_var,numb_var))
    #Delta2_old=np.zeros((numb_var,numb_var,numb_var))
   
    Eval_TF=np.zeros(numb_var)

   
    for i in range(numb_var):
        Eval_TF[i]=F[i](*vsec_vec[i])
        #print('evaluated TF',i,' = ', Eval_TF[i])

        for k in range(numb_sub_exc):
           
            Delta1[i][k]=deriv1b(F[i],vsec_vec[i],numb_v_TF,h,0)*N[k]/N0_e*p[i][k]/(5e-2) # 0 is taken because TF takes only exc and inhib inputs, without discarding the subpops
           
            '''Be careful here, there might be a problem with the type of numb_sub_(exc/inh) (int/float)'''
               
        for k in range(numb_sub_inh):
           
            Delta1[i][k+numb_sub_exc]=deriv1b(F[i],vsec_vec[i],numb_v_TF,h,1)*N[k+numb_sub_exc]/N0_i*p[i][k+numb_sub_exc]/(5e-2)

        for k in range(numb_sub_exc):
            for j in range(numb_sub_exc):

                Delta2[i][k][j]=deriv2b(F[i],vsec_vec[i],numb_v_TF,h,0,0)*\
                N[k]/N0_e*N[j]/N0_e*p[i][k]/(5e-2)*p[i][j]/(5e-2)
               
            for j in range(numb_sub_inh):
                Delta2[i][k][j+numb_sub_exc]=deriv2b(F[i],vsec_vec[i],numb_v_TF,h,0,1)*\
                N[k]/N0_e*N[j+numb_sub_exc]/N0_i*p[i][k]/(5e-2)*p[i][j+numb_sub_exc]/(5e-2)
               
        for k in range(numb_sub_inh):
            for j in range(numb_sub_exc):
                    Delta2[i][k+numb_sub_exc][j]=deriv2b(F[i],vsec_vec[i],numb_v_TF,h,1,0)*\
                    N[k+numb_sub_exc]/N0_i*N[j]/N0_e*p[i][k+numb_sub_exc]/(5e-2)*p[i][j]/(5e-2)
            for j in range(numb_sub_inh):
                Delta2[i][k+numb_sub_exc][j+numb_sub_exc]=deriv2b(F[i],vsec_vec[i],numb_v_TF,h,1,1)*\
                N[k+numb_sub_exc]/N0_i*N[j+numb_sub_exc]/N0_i*p[i][k+numb_sub_exc]/(5e-2)*\
                p[i][j+numb_sub_exc]/(5e-2)
                #print(Delta2[i][k][j])
               
       
    for i in range(numb_var):
        deriv[i]=(Eval_TF[i]-x0[i])/T

        deriv[numb_var*i+i+numb_var]+=(Eval_TF[i]*(1./T-Eval_TF[i])/N[i])/T
        for j in range(numb_var):
            index=numb_var*i+j+numb_var # define here the location of the considered c variable and its derivative
            for k in range(numb_var):
                deriv[i]+=(0.5*Delta2[i][j][k]*x0[numb_var*j+k+numb_var])/T
                index_ik=numb_var*i+k+numb_var
                index_jk=numb_var*j+k+numb_var
                deriv[index]+=(x0[index_jk]*Delta1[i][k]+x0[index_ik]*Delta1[j][k])/T
               
            deriv[index]+=((Eval_TF[i]-x0[i])*(Eval_TF[j]-x0[j])-2*x0[index])/T
 
               
    for i in range(numb_adapt):
        deriv[numb_var**2+numb_var+i]=-x0[numb_var**2+numb_var+i]/tauw[i]+b[i]*x0[i]+a[i]*(mu(x0,params,i,t)-El[i])/tauw[i]
       
      # DEFINE MU !!!!#
   
   
    ''' Printing to debug '''
   
   
    #print('\n','\n','TIME = ', t, '\n','\n', 'Delta1 = ', Delta1,'\n','Delta1 Old = ', Delta1_old, '\n', 'Delta2 = ', Delta2,'\n', 'Delta2 Old= ', Delta2_old,'\n')
               
    return deriv,Eval_TF

def heaviside(x):
    return 0.5*(1+np.sign(x))

def smooth_heaviside(x):
    return 0.5*(1+erf(x))
   
def square_input(t, t1_exc, tau1_exc, tau2_exc, ampl_exc, plateau):
    # t1_exc=10. # time of the maximum of external stimulation
    # tau1_exc=20. # first time constant of perturbation = rising time
    # tau2_exc=50. # decaying time
    # ampl_exc=20. # amplitude of excitation
    inp = ampl_exc * (np.exp(-(t - t1_exc) ** 2 / (2. * tau1_exc ** 2)) * heaviside(-(t - t1_exc)) + \
                      heaviside(-(t - (t1_exc+plateau))) * heaviside(t - (t1_exc))+ \
                      np.exp(-(t - (t1_exc+plateau)) ** 2 / (2. * tau2_exc ** 2)) * heaviside(t - (t1_exc+plateau)))
    return inp

def alpha_inp(t,t1_exc,tau1_exc,tau2_exc,ampl_exc):
    #t1_exc=10. # time of the maximum of external stimulation 
    #tau1_exc=20. # first time constant of perturbation = rising time 
    #tau2_exc=50. # decaying time 
    #ampl_exc=20. # amplitude of excitation
    inp=ampl_exc*(np.exp(-(t-t1_exc)**2/(2.*tau1_exc**2))*heaviside(-(t-t1_exc))+\
    np.exp(-(t-t1_exc)**2/(2.*tau2_exc**2))*heaviside(t-t1_exc))
    return inp


def rounded_square(t,ampl,t1,t2,pow1,pow2):
    inp=ampl*(smooth_heaviside(t-t1)**pow1-smooth_heaviside(t-t2)**pow2)
    return inp

def sinusoid(t,ampl,freq):
    return ampl*math.sin(2*math.pi*freq*t)


def mu_V_dN_dp(x0,params,ind,t): 
    
    ''' this mu_V function takes x0, because vsec_vec are too convoluted to play with, the reason being that 
    thir modification stems from the calculation of TF and nothing else !'''
    
    p0=5e-2
    N0e=8000
    N0i=2000
    N=params['size_subpop']
    numb_var=params['numb_pop']
    numb_adapt=params['numb_adapt']
    ext_drive=params['ext_inp']
    input_rate=params['inp_funct']
    
    if (input_rate != 0): input_func=input_rate(t, *params['inp_par'])
    else : input_func=0
    
    numb_sub_exc=params['numb_sub_exc']
    numb_sub_inh=params['numb_sub_inh']

    Ntot_exc=0
    Ntot_inh=0
    
    for i in range(numb_sub_exc): Ntot_exc+=N[i]
    for i in range(numb_sub_inh): Ntot_inh+=N[i+numb_sub_exc]

    p=params['p_conn']
    p_pois=params['p_pois']

    
    El=El=params['El'][ind]
    tau_e=params['tau_e']
    tau_i=params['tau_i']
    Qe=params['qe']
    Qi=params['qi']
    Ee=params['Ee']
    Ei=params['Ei']   
    gl=params['Gl'][ind]
    
    
    
    mu_Ge=0#ve*Ke*tau_e*Qe   
    for i in range(numb_sub_exc): # here we recompose the values ! Keep in mind that we consider the bombardment !
        mu_Ge+=x0[i]*p[ind][i]*N[i]*tau_e*Qe
        
    mu_Ge+=(ext_drive[ind]+input_func)*p_pois[ind]*Ntot_exc*tau_e*Qe
   # sigma_Ge=math.sqrt(ve*Ke*tau_e/2)*Qe
   
   
    mu_Gi=0
    for i in range(numb_sub_inh):
        mu_Gi+=x0[i+numb_sub_exc]*p[ind][i+numb_sub_exc]*N[i+numb_sub_exc]*tau_i*Qi
    #sigma_Gi=math.sqrt(vi*Ki*tau_i/2)*Qi
    
    
    mu_G=mu_Ge+mu_Gi+gl
    muV=(mu_Ge*Ee+mu_Gi*Ei+gl*El-x0[numb_var**2+numb_var+ind])/mu_G
    
    
    return muV

def mu_V_dN_dp_FO(x0,params,ind,t): 
    
    ''' this mu_V function takes x0, because vsec_vec are too convoluted to play with, the reason being that 
    thir modification stems from the calculation of TF and nothing else !'''
    
    p0=5e-2
    N0e=8000
    N0i=2000
    N=params['size_subpop']
    numb_var=params['numb_pop']
    numb_adapt=params['numb_adapt']
    ext_drive=params['ext_inp']
    input_rate=params['inp_funct']
    
    if (input_rate != 0): input_func=input_rate(t, *params['inp_par'])
    else : input_func=0
 
    numb_sub_exc=params['numb_sub_exc']
    numb_sub_inh=params['numb_sub_inh']
  
    Ntot_exc=0
    Ntot_inh=0
    
    for i in range(numb_sub_exc): Ntot_exc+=N[i]
    for i in range(numb_sub_inh): Ntot_inh+=N[i+numb_sub_exc]
    p=params['p_conn']
    p_pois=params['p_pois']

    
    El=params['El'][ind]
    tau_e=params['tau_e']
    tau_i=params['tau_i']
    Qe=params['qe']
    Qi=params['qi']
    Ee=params['Ee']
    Ei=params['Ei']   
    gl=params['Gl'][ind]
    

    
    mu_Ge=0#ve*Ke*tau_e*Qe   
    for i in range(numb_sub_exc): # here we recompose the values !
        mu_Ge+=x0[i]*p[ind][i]*N[i]*tau_e*Qe
        
    mu_Ge+=(ext_drive[ind]+input_func)*p_pois[ind]*Ntot_exc*tau_e*Qe
   # sigma_Ge=math.sqrt(ve*Ke*tau_e/2)*Qe
   
   
    mu_Gi=0
    for i in range(numb_sub_inh):
        mu_Gi+=x0[i+numb_sub_exc]*p[ind][i+numb_sub_exc]*N[i+numb_sub_exc]*tau_i*Qi
    #sigma_Gi=math.sqrt(vi*Ki*tau_i/2)*Qi
    
    
    mu_G=mu_Ge+mu_Gi+gl
    muV=(mu_Ge*Ee+mu_Gi*Ei+gl*El-x0[numb_var+ind])/mu_G
    

    return muV


def mu_V_Cad(x0,params,ind,t): 
    
    ''' this mu_V function takes x0, because vsec_vec are too convoluted to play with, the reason being that 
    thir modification stems from the calculation of TF and nothing else !'''
    
    
    N=params['size_subpop']
    numb_var=params['numb_pop']
    numb_adapt=params['numb_adapt']
    tau_a=params['tau_a']
    Ea=params['Ea']
    ga_bar=params['ga_bar']
    dga=params['dga']
    Delta_a=params['delta_a']
    Va=params['Va']
   
    ext_drive=params['ext_inp']
    input_rate=params['inp_funct']
    
    if (input_rate != 0): input_func=input_rate(t, *params['inp_par'])
    else : input_func=0
 
  
    p=params['p_conn']
    p_pois=params['p_pois']

    
    El=params['El'][ind]
    tau_e=params['tau_e']
    tau_i=params['tau_i']
    Qe=params['qe']
    Qi=params['qi']
    Ee=params['Ee']
    Ei=params['Ei']   
    gl=params['Gl'][ind]
    
    numb_sub_exc=params['numb_sub_exc']
    numb_sub_inh=params['numb_sub_inh']
    
    mu_Ge=0#ve*Ke*tau_e*Qe   
    for i in range(numb_sub_exc): # here we recompose the values !
        mu_Ge+=x0[i]*p[ind][i]*N[i]*tau_e*Qe
        
    mu_Ge+=(ext_drive+input_func)*p_pois[ind]*N[0]*tau_e*Qe
   # sigma_Ge=math.sqrt(ve*Ke*tau_e/2)*Qe
   
   
    mu_Gi=0
    for i in range(numb_sub_inh):
        mu_Gi+=x0[i+numb_sub_exc]*p[ind][i+numb_sub_exc]*N[i+numb_sub_exc]*tau_i*Qi
    #sigma_Gi=math.sqrt(vi*Ki*tau_i/2)*Qi
    
    
    mu_G=mu_Ge+mu_Gi+gl+x0[numb_var**2+numb_var+ind]
    muV=(mu_Ge*Ee+mu_Gi*Ei+gl*El+x0[numb_var**2+numb_var+ind]*Ea[ind])/mu_G
    
    
    return muV

def mu_V_Cad_FO(x0,params,ind,t): 
    
    ''' this mu_V function takes x0, because vsec_vec are too convoluted to play with, the reason being that 
    thir modification stems from the calculation of TF and nothing else !'''
    
    N=params['size_subpop']
    numb_var=params['numb_pop']
    numb_adapt=params['numb_adapt']
    tau_a=params['tau_a']
    Ea=params['Ea']
    ga_bar=params['ga_bar']
    dga=params['dga']
    Delta_a=params['delta_a']
    Va=params['Va']
   
    ext_drive=params['ext_inp']
    input_rate=params['inp_funct']
    
    if (input_rate != 0): input_func=input_rate(t, *params['inp_par'])
    else : input_func=0
 
  
    p=params['p_conn']
    p_pois=params['p_pois']

    
    El=params['El'][ind]
    tau_e=params['tau_e']
    tau_i=params['tau_i']
    Qe=params['qe']
    Qi=params['qi']
    Ee=params['Ee']
    Ei=params['Ei']   
    gl=params['Gl'][ind]
    
    numb_sub_exc=params['numb_sub_exc']
    numb_sub_inh=params['numb_sub_inh']
    

    mu_Ge=0#ve*Ke*tau_e*Qe   
    for i in range(numb_sub_exc): # here we recompose the values !
        mu_Ge+=x0[i]*p[ind][i]*N[i]*tau_e*Qe
        
    mu_Ge+=(ext_drive+input_func)*p_pois[ind]*N[0]*tau_e*Qe
   # sigma_Ge=math.sqrt(ve*Ke*tau_e/2)*Qe
   
   
    mu_Gi=0
    for i in range(numb_sub_inh):
        mu_Gi+=x0[i+numb_sub_exc]*p[ind][i+numb_sub_exc]*N[i+numb_sub_exc]*tau_i*Qi
    #sigma_Gi=math.sqrt(vi*Ki*tau_i/2)*Qi
    
    
    mu_G=mu_Ge+mu_Gi+gl+x0[numb_var+ind]
    muV=(mu_Ge*Ee+mu_Gi*Ei+gl*El+x0[numb_var+ind]*Ea[ind])/mu_G
    
    return muV


def mu_V_discrete_dp(x0,params,ind,t): 
    
    ''' this mu_V function takes x0, because vsec_vec are too convoluted to play with, the reason being that 
    thir modification stems from the calculation of TF and nothing else !'''
    
   
    N=params[6]
    numb_var=params[7]
    numb_adapt=params[8]
    ext_drive=params[21]
    input_rate=params[22]
    if (params[22] != 0): input_func=params[22](t, *params[23])
    else : input_func=0
 
  
    p=params[26]
    
    dp_e=params[27][0]
    dp_i=params[27][1]
    
    p_pois=params[28]
    
    El=params[5][ind]
    tau_e=params[12]
    tau_i=params[13]
    Qe=params[14]
    Qi=params[15]
    Ee=params[16]
    Ei=params[17]   
    gl=params[19]
    
    numb_sub_exc=params[24]
    numb_sub_inh=params[25]
    
    mu_Ge=0#ve*Ke*tau_e*Qe   
    for i in range(numb_sub_exc): # here we recompose the values !
        mu_Ge+=x0[i]*(p[ind][i]+dp_e[ind])*N[i]*tau_e*Qe
        
    mu_Ge+=(ext_drive+input_func)*p_pois[ind]*N[0]
   # sigma_Ge=math.sqrt(ve*Ke*tau_e/2)*Qe
   
   
    mu_Gi=0
    for i in range(numb_sub_inh):
        mu_Gi+=x0[i+numb_sub_exc]*(p[ind][i+numb_sub_exc]+dp_i[i])*N[i+numb_sub_exc]*tau_i*Qi
        #mu_Gi+=x0[i+numb_sub_exc]*(p[ind][i+numb_sub_exc])*N[i+numb_sub_exc]*tau_i*Qi
    #sigma_Gi=math.sqrt(vi*Ki*tau_i/2)*Qi
    
    
    mu_G=mu_Ge+mu_Gi+gl
    muV=(mu_Ge*Ee+mu_Gi*Ei+gl*El-x0[numb_var**2+numb_var+ind])/mu_G
    
    
    return muV


def mu_V_discrete_dp_FO(x0,params,ind,t): 
    
    ''' this mu_V function takes x0, because vsec_vec are too convoluted to play with, the reason being that 
    thir modification stems from the calculation of TF and nothing else !'''
    
   
    N=params[6]
    numb_var=params[7]
    numb_adapt=params[8]
    ext_drive=params[21]
    input_rate=params[22]
    if (params[22] != 0): input_func=params[22](t, *params[23])
    else : input_func=0
 
  
    p=params[26]
    
    dp_e=params[27][0]
    dp_i=params[27][1]
    p_pois=params[28]
    
    El=params[5][ind]
    tau_e=params[12]
    tau_i=params[13]
    Qe=params[14]
    Qi=params[15]
    Ee=params[16]
    Ei=params[17]   
    gl=params[19]
    
    numb_sub_exc=params[24]
    numb_sub_inh=params[25]
    
    mu_Ge=0#ve*Ke*tau_e*Qe   
    for i in range(numb_sub_exc): # here we recompose the values !
        mu_Ge+=x0[i]*(p[ind][i]+dp_e[ind])*N[i]*tau_e*Qe
        
    mu_Ge+=(ext_drive+input_func)*p_pois[ind]*N[0]
   # sigma_Ge=math.sqrt(ve*Ke*tau_e/2)*Qe
   
   
    mu_Gi=0
    for i in range(numb_sub_inh):
        mu_Gi+=x0[i+numb_sub_exc]*(p[ind][i+numb_sub_exc]+dp_i[i])*N[i+numb_sub_exc]*tau_i*Qi
    
    
    mu_G=mu_Ge+mu_Gi+gl
    muV=(mu_Ge*Ee+mu_Gi*Ei+gl*El-x0[numb_var+ind])/mu_G
    
    
    return muV

def deriv_w_mu(funct,v,params,i,t): ### derivative of mu wrt w

    h=params['h']
    
    first_deriv=[]
    #vsec_plus=np.ones(numb_var)
    #vsec_minus=np.ones(numb_var)
    
    #for i in range(numb_var+numb_adapt): # initialization of perturbation vector
    vsec_plus=v.copy()
    vsec_minus=v.copy()
    
    #for i in range(numb_var):
    vsec_plus[2]+=0.5*h # perturb one component
    vsec_minus[2]-=0.5*h
    first_deriv=(funct(vsec_plus,params,i,t)-funct(vsec_minus,params,i,t))/h
    #vsec_plus[i]=v[i] # reset component
    #vsec_minus[i]=v[i]
        
    return first_deriv
