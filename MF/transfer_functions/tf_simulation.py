import numpy as np
#import numba # with NUMBA to make it faster !!!
#from brian2 import *
import argparse
import matplotlib.pyplot as plt

### ================================================
### ========== Reformat parameters  ================
### ======= for single cell simulation =============
### ================================================


def reformat_syn_parameters(params, M):
    """
    valid only of no synaptic differences between excitation and inhibition
    """

    params['Qe'], params['Te'], params['Ee'] = M[0,0]['Q'], M[0,0]['Tsyn'], M[0,0]['Erev']
    params['Qi'], params['Ti'], params['Ei'] = M[1,1]['Q'], M[1,1]['Tsyn'], M[1,1]['Erev']
    params['pconnec'] = M[0,0]['p_conn']
    params['Ntot'], params['gei'] = M[0,0]['Ntot'], M[0,0]['gei']
    

### ================================================
### ======== Conductance Time Trace ================
### ====== Poisson + Exponential synapses ==========
### ================================================

def generate_conductance_shotnoise(freq, t, N, Q, Tsyn, g0=0, seed=0):
    """
    generates a shotnoise convoluted with a waveform
    frequency of the shotnoise is freq,
    K is the number of synapses that multiplies freq
    g0 is the starting value of the shotnoise
    """
    if freq==0:
        # print "problem, 0 frequency !!! ---> freq=1e-9 !!"
        freq=1e-9
    upper_number_of_events = max([int(3*freq*t[-1]*N),1]) # at least 1 event
    np.random.seed(seed=seed)
    spike_events = np.cumsum(np.random.exponential(1./(N*freq),\
                             upper_number_of_events))
    g = np.ones(t.size)*g0 # init to first value
    dt, t = t[1]-t[0], t-t[0] # we need to have t starting at 0
    # stupid implementation of a shotnoise
    event = 0 # index for the spiking events
    for i in range(1,t.size):
        g[i] = g[i-1]*np.exp(-dt/Tsyn)
        while spike_events[event]<=t[i]:
            g[i]+=Q
            event+=1
    return g

### ================================================
### ======== AdExp model (with IaF) ================
### ================================================

def pseq_adexp(cell_params):
    """ function to extract all parameters to put in the simulation
    (just because you can't pass a dict() in Numba )"""

    # those parameters have to be set
    El, Gl = cell_params['El'], cell_params['Gl']
    Ee, Ei = cell_params['Ee'], cell_params['Ei']
    Cm = cell_params['Cm']
    a, b, tauw = cell_params['a'],\
                     cell_params['b'], cell_params['tauw']
    trefrac, delta_v = cell_params['Trefrac'], cell_params['delta_v']
    
    vthresh, vreset =cell_params['Vthre'], cell_params['Vreset']
   
    # then those can be optional
    if 'vspike' not in cell_params.keys():
        vspike = vthresh+5*delta_v # as in the Brian simulator !
    else: vspike=cell_params['vspike']

    return El, Gl, Cm, Ee, Ei, vthresh, vreset, vspike,\
                     trefrac, delta_v, a, b, tauw


# @numba.jit('u1[:](f8[:], f8[:], f8[:], f8[:], f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8)')
def adexp_sim(t, I, Ge, Gi,
              El, Gl, Cm, Ee, Ei, vthresh, vreset, vspike, trefrac, delta_v, a, b, tauw):
    """ functions that solve the membrane equations for the
    adexp model for 2 time varying excitatory and inhibitory
    conductances as well as a current input
    returns : v, spikes
    """
    
    print("RRRR",vspike)
    if delta_v==0: # i.e. Integrate and Fire
        one_over_delta_v = 0
    else:
        one_over_delta_v = 1./delta_v
        
    #vspike=vthresh+5.*delta_v # practival threshold detection
            
    last_spike = -np.inf # time of the last spike, for the refractory period
    V, spikes = El*np.ones(len(t), dtype=np.float), []
    wtime=V
    dt = t[1]-t[0]
    Vc = []
    w, i_exp,wpi = 0., 0.,0. # w and i_exp are the exponential and adaptation currents
    waver,wcounting = 0.,0.


    print("zzz",Cm, Gl,Gl/Cm)

    for i in range(len(t)-1):
        w = w + dt/tauw*(a*(V[i]-El)-w) # adaptation current
        #wtime[i]=w
        i_exp = Gl*delta_v*np.exp((V[i]-vthresh)*one_over_delta_v) 
        #print(i,'th iteration ; t = ', t[i], ' ; lastspike = ', last_spike)
        
        #print("ggggggggggggggggggggggg",w,1./tauw*(a*(V[i]-El)),a)

        if i>len(t)/2: # only after a transient
            waver+=w
            wcounting+=1.0
            wpi+=(a*(V[i]-El))


        if i>len(t)/2:
            if V[i]<vthresh:
            	Vc.append(V[i])

        
        if (t[i]-last_spike)>trefrac: # only when non refractory
            ## Vm dynamics calculus
            V[i+1] = V[i] + dt/Cm*(I[i] + i_exp - w +\
                 Gl*(El-V[i]) + Ge[i]*(Ee-V[i]) + Gi[i]*(Ei-V[i]) )
            #print('\n','\n','INTEGRATE V => V[i+1] = ', V[i+1], '\n','\n')

        if V[i+1] > vspike:

            #print('\n','\n',' => V[i+1] = ', V[i+1], ' ; vspike = ', vspike, '  vreset = ', vreset, '\n','\n')
            V[i+1] = vreset # non estethic version
            w = w + b # then we increase the adaptation current

            last_spike = t[i+1]
            if last_spike>t.max()/2:
            #if last_spike>10.:
                spikes.append(t[i+1])

    
    #Vnew=V[t>t.max()/2]
    print("eeeee",vreset,V[V!=vreset].mean())
    plt.plot(t[V!=vreset],V[V!=vreset])
    #plt.show()
    #print("tttttttttttttttttt",waver/wcounting,wpi/wcounting)
    return V, np.array(spikes),waver/wcounting,np.mean(Vc)






def adexp_sim_A(t, I, Ge, Gi,
              El, Gl, Cm, Ee, Ei, vthresh, vreset, vspike, trefrac, delta_v, a, b, tauw):
    """ functions that solve the membrane equations for the
        adexp model for 2 time varying excitatory and inhibitory
        conductances as well as a current input
        returns : v, spikes
        """
    
    print("RRRR",vspike)
    if delta_v==0: # i.e. Integrate and Fire
        one_over_delta_v = 0
    else:
        one_over_delta_v = 1./delta_v
    
    vspike=vthresh+55.*delta_v # practival threshold detection
    
    last_spike = -np.inf # time of the last spike, for the refractory period
    V, spikes = El*np.ones(len(t), dtype=np.float), []
    wtime=V
    dt = t[1]-t[0]
    Vc = []
    w, i_exp,wpi = 0., 0.,0. # w and i_exp are the exponential and adaptation currents
    waver,wcounting = 0.,0.
    
    
    print("zzz",Cm, Gl,Gl/Cm)
    
    for i in range(len(t)-1):
        w = w + dt/tauw*(a*(V[i]-El)-w) # adaptation current
        #wtime[i]=w
        i_exp = Gl*delta_v*np.exp((V[i]-vthresh)*one_over_delta_v)
        
        #print("ggggggggggggggggggggggg",w,1./tauw*(a*(V[i]-El)),a)
        
        if i>len(t)/2: # only after a transient
            waver+=w
            wcounting+=1.0
            wpi+=(a*(V[i]-El))
        
        
        if i>len(t)/2:
            if V[i]<vthresh:
                Vc.append(V[i])
        
        
        if (t[i]-last_spike)>trefrac: # only when non refractory
            ## Vm dynamics calculus
            V[i+1] = V[i] + dt/Cm*(I[i] + i_exp - w +\
                                   Gl*(El-V[i]) + Ge[i]*(Ee-V[i]) + Gi[i]*(Ei-V[i]) )
    
        if V[i+1] > vspike:
            
            V[i+1] = vreset # non estethic version
            w = w + b # then we increase the adaptation current
            
            last_spike = t[i+1]
            #if last_spike>t.max()/2:
            if last_spike>10.:
                spikes.append(t[i+1])


    Vnew=V[t>10.]
    print("eeeee",vreset,V[V!=vreset].mean())
    plt.plot(t[V!=vreset],V[V!=vreset])
    #plt.show()
    #print("tttttttttttttttttt",waver/wcounting,wpi/wcounting)
    return V, np.array(spikes),waver/wcounting,np.mean(Vc)




def adexp_sim_3(t, I, Ge, Gi,Ginp,
              El, Gl, Cm, Ee, Ei, vthresh, vreset, vspike, trefrac, delta_v, a, b, tauw):
    """ functions that solve the membrane equations for the
        adexp model for 2 time varying excitatory and inhibitory
        conductances as well as a current input
        returns : v, spikes
        """
    
    print("RRRR",vspike)
    if delta_v==0: # i.e. Integrate and Fire
        one_over_delta_v = 0
    else:
        one_over_delta_v = 1./delta_v
    
    #vspike=vthresh+5.*delta_v # practival threshold detection
    
    last_spike = -np.inf # time of the last spike, for the refractory period
    V, spikes = El*np.ones(len(t), dtype=np.float), []
    wtime=V
    dt = t[1]-t[0]
    Vc = []
    w, i_exp,wpi = 0., 0.,0. # w and i_exp are the exponential and adaptation currents
    waver,wcounting = 0.,0.
    
    
    for i in range(len(t)-1):
        w = w + dt/tauw*(a*(V[i]-El)-w) # adaptation current
        #wtime[i]=w
        i_exp = Gl*delta_v*np.exp((V[i]-vthresh)*one_over_delta_v)
        
        #print("ggggggggggggggggggggggg",w,1./tauw*(a*(V[i]-El)),a)
        
        if i>len(t)/2: # only after a transient
            waver+=w
            wcounting+=1.0
            wpi+=(a*(V[i]-El))
        
        
        if i>len(t)/2:
            if V[i]<vthresh:
                Vc.append(V[i])
        
        
        if (t[i]-last_spike)>trefrac: # only when non refractory
            ## Vm dynamics calculus
            V[i+1] = V[i] + dt/Cm*(I[i] + i_exp - w +\
                                   Gl*(El-V[i]) + Ge[i]*(Ee-V[i])+ Ginp[i]*(Ee-V[i]) + Gi[i]*(Ei-V[i]) )
    
        if V[i+1] > vspike:
            
            V[i+1] = vreset # non estethic version
            w = w + b # then we increase the adaptation current
            
            last_spike = t[i+1]
            
            if last_spike>t.max()/2:
            #if last_spike>10.:
                spikes.append(t[i+1])


    Vnew=V[t>t.max()/2]
    print("eeeee",vreset,V[V!=vreset].mean())
    plt.plot(t[V!=vreset],V[V!=vreset])
    #plt.show()
    #print("tttttttttttttttttt",waver/wcounting,wpi/wcounting)
    return V, np.array(spikes),waver/wcounting,np.mean(Vc)



### ================================================
### ========== Single trace experiment  ============
### ================================================

def single_experiment(t, fe, fi, params, seed=0):
    ## fe and fi total synaptic activities, they include the synaptic number
    ge = generate_conductance_shotnoise(fe, t, 1, params['Qe'], params['Te'], g0=0, seed=seed)
    gi = generate_conductance_shotnoise(fi, t, 1, params['Qi'], params['Ti'], g0=0, seed=seed)
    a, b, tauw = params['a'],\
                     params['b'], params['tauw']
    Qe, Te, Ee = params['Qe'], params['Te'], params['Ee']
    Qi, Ti, Ei = params['Qi'], params['Ti'], params['Ei']
    Gl, Cm , El = params['Gl'], params['Cm'] , params['El']
    

    I = np.zeros(len(t))
    v, spikes,w,Vaver = adexp_sim(t, I, ge, gi, *pseq_adexp(params))

    muGe, muGi = Qe*Te*fe, Qi*Ti*fi
    muG = Gl+muGe+muGi

    muV = (muGe*Ee+muGi*Ei+Gl*El-w)/muG

    froutput=len(spikes)/(t.max()/2)
    #froutput=2*len(spikes)/(t.max())
    wth=(froutput)*tauw*(b)-a*(El-muV)

    
    muVV=((muGe*Ee+muGi*Ei+Gl*El-(froutput)*tauw*(b)+a*El)/muG)/(1+a/muG)
    partth=-a*(El-muVV)
    del v
    del spikes
    print ('AAAAAA fout w', fe/(0.8*0.02*10000.),fi/(0.2*0.02*10000.),froutput,w,wth,Vaver,muVV)
    #clear_cache('cython')
    return froutput,w # finally we get the output frequency





def single_experiment_2(t, fe, fi, params, seed=0):
    ## fe and fi total synaptic activities, they include the synaptic number
    ge = generate_conductance_shotnoise(fe, t, 1, params['Qe'], params['Te'], g0=0, seed=seed)
    gi = generate_conductance_shotnoise(fi, t, 1, params['Qi'], params['Ti'], g0=0, seed=seed)
    a, b, tauw = params['a'],\
        params['b'], params['tauw']
    Qe, Te, Ee = params['Qe'], params['Te'], params['Ee']
    Qi, Ti, Ei = params['Qi'], params['Ti'], params['Ei']
    Gl, Cm , El = params['Gl'], params['Cm'] , params['El']
    vress=params['Vreset'],

    I = np.zeros(len(t))
    v, spikes,waver,vsdv= adexp_sim(t, I, ge, gi, *pseq_adexp(params))
    


    froutput=len(spikes)/(t.max()-10.)

    first=ge*(Ee-v)
    second=gi*(Ei -v)
    third=Gl*(El-v)

    derivative=first[(v<-0.05) & (v!=-0.065)].mean()+second[(v<-0.05) & (v!=-0.065)].mean()+third[(v<-0.05) & (v!=-0.065)].mean()
    derivative1=first.mean()+second.mean()+third.mean()

    '''
    geexp=ge[(v<-0.05) & (v>-0.064)].mean()
    giexp=gi[(v<-0.05) & (v>-0.064)].mean()

    '''
    geexp=ge.mean()
    giexp=gi.mean()

    voltageexp=(geexp*Ee+giexp*Ei+Gl*El)/(geexp+giexp+Gl)


    muGe, muGi = Qe*Te*fe, Qi*Ti*fi
    muG = Gl+muGe+muGi
    

    
    muVV=((muGe*Ee+muGi*Ei+Gl*El-(froutput)*tauw*(b)+a*El)/muG)/(1+a/muG)
    wth=(froutput)*tauw*(b)-a*(El-muVV)
    #print("exapmle DATA",muVV,froutput,muG,muGe)
    print ('AAAAAA fout w',vress,a,b,froutput,1000.*voltageexp,1000*v[(t>10) & (v!=vress)  ].mean(),1000*v[(t>10) &(v!=vress)  ].std(),1000.*muVV)
    
    return froutput,v[(t>10) &(v!=vress) ].mean(),v[(t>10) &(v!=vress) ].std() # finally we get the output frequency



def single_experiment_2A(t, fe, fi, params, seed=0):
    ## fe and fi total synaptic activities, they include the synaptic number
    ge = generate_conductance_shotnoise(fe, t, 1, params['Qe'], params['Te'], g0=0, seed=seed)
    gi = generate_conductance_shotnoise(fi, t, 1, params['Qi'], params['Ti'], g0=0, seed=seed)
    a, b, tauw = params['a'],\
        params['b'], params['tauw']
    Qe, Te, Ee = params['Qe'], params['Te'], params['Ee']
    Qi, Ti, Ei = params['Qi'], params['Ti'], params['Ei']
    Gl, Cm , El = params['Gl'], params['Cm'] , params['El']
    vress=params['Vreset'],

    I = np.zeros(len(t))
    v, spikes,waver,vsdv= adexp_sim_A(t, I, ge, gi, *pseq_adexp(params))
    
    
    
    froutput=len(spikes)/(t.max()-10.)
    
    first=ge*(Ee-v)
    second=gi*(Ei -v)
    third=Gl*(El-v)
    
    derivative=first[(v<-0.05) & (v!=-0.065)].mean()+second[(v<-0.05) & (v!=-0.065)].mean()+third[(v<-0.05) & (v!=-0.065)].mean()
    derivative1=first.mean()+second.mean()+third.mean()
    

    geexp=ge.mean()
    giexp=gi.mean()
    
    voltageexp=(geexp*Ee+giexp*Ei+Gl*El)/(geexp+giexp+Gl)
    
    
    muGe, muGi = Qe*Te*fe, Qi*Ti*fi
    muG = Gl+muGe+muGi
    
    
    
    muVV=((muGe*Ee+muGi*Ei+Gl*El-(froutput)*tauw*(b)+a*El)/muG)/(1+a/muG)
    wth=(froutput)*tauw*(b)-a*(El-muVV)

    
    return t,v

def single_experiment_3(t, fe, fi,finp, params, seed=0):
    ## fe and fi total synaptic activities, they include the synaptic number
    ge = generate_conductance_shotnoise(finp, t, 1, 5*params['Qe'], params['Te'], g0=0, seed=seed)
    ginput = generate_conductance_shotnoise(fe, t, 1, params['Qe'], params['Te'], g0=0, seed=seed)
    gi = generate_conductance_shotnoise(fi, t, 1, params['Qi'], params['Ti'], g0=0, seed=seed)
    a, b, tauw = params['a'],\
        params['b'], params['tauw']
    Qe, Te, Ee = params['Qe'], params['Te'], params['Ee']
    Qi, Ti, Ei = params['Qi'], params['Ti'], params['Ei']
    Gl, Cm , El = params['Gl'], params['Cm'] , params['El']
    vress=params['Vreset'],

    I = np.zeros(len(t))
    v, spikes,waver,vsdv= adexp_sim_3(t, I, ge, gi,ginput, *pseq_adexp(params))
    
    
    
    froutput=len(spikes)/(t.max()-10.)
    
    first=ge*(Ee-v)
    second=gi*(Ei -v)
    third=Gl*(El-v)
    
    derivative=first[(v<-0.05) & (v!=-0.065)].mean()+second[(v<-0.05) & (v!=-0.065)].mean()+third[(v<-0.05) & (v!=-0.065)].mean()
    derivative1=first.mean()+second.mean()+third.mean()
    
    '''
        geexp=ge[(v<-0.05) & (v>-0.064)].mean()
        giexp=gi[(v<-0.05) & (v>-0.064)].mean()
        
        '''
    geexp=ge.mean()
    giexp=gi.mean()
    
    voltageexp=(geexp*Ee+giexp*Ei+Gl*El)/(geexp+giexp+Gl)
    
    
    muGe, muGi = Qe*Te*fe, Qi*Ti*fi
    muG = Gl+muGe+muGi
    
    
    
    muVV=((muGe*Ee+muGi*Ei+Gl*El-(froutput)*tauw*(b)+a*El)/muG)/(1+a/muG)
    wth=(froutput)*tauw*(b)-a*(El-muVV)
    #print("exapmle DATA",muVV,froutput,muG,muGe)
    print ('AAAAAA fout w',vress,a,b,froutput,1000.*voltageexp,1000*v[(t>10) & (v!=vress)  ].mean(),1000*v[(t>10) &(v!=vress)  ].std(),1000.*muVV)
    
    return froutput,v[(t>10) &(v!=vress) ].mean(),v[(t>10) &(v!=vress) ].std() # finally we get the output frequency



### ================================================
### ========== Transfer Functions ==================
### ================================================

### generate a transfer function's data
def generate_transfer_function(params,\
                               MAXfexc=40., MAXfinh=30., MINfinh=2.,\
                               discret_exc=9, discret_inh=8, MAXfout=35.,\
                               SEED=3,\
                               verbose=False,
                               filename='data/example_data.npy',
                               dt=5e-5, tstop=500):
    """ Generate the data for the transfer function  """
    
    t = np.arange(int(tstop/dt))*dt

    # this sets the boundaries (factor 20)
    dFexc = MAXfexc/discret_exc
    fiSim=np.linspace(MINfinh,MAXfinh, discret_inh)
    feSim=np.linspace(0, MAXfexc, discret_exc) # by default
    MEANfreq = np.zeros((fiSim.size,feSim.size))
    SDfreq = np.zeros((fiSim.size,feSim.size))
    Fe_eff = np.zeros((fiSim.size,feSim.size))
    w=np.zeros((fiSim.size,feSim.size))
    JUMP = np.linspace(0,MAXfout,discret_exc) # constrains the fout jumps

    for i in range(fiSim.size):
        Fe_eff[i][:] = feSim # we try it with this scaling
        e=1 # we start at fe=!0
        while (e<JUMP.size):
            vec = np.zeros(SEED)
            wec = np.zeros(SEED)
            vec[0],wec[0]= single_experiment(t,\
                Fe_eff[i][e]*(1-params['gei'])*params['pconnec']*params['Ntot'],
                fiSim[i]*params['gei']*params['pconnec']*params['Ntot'], params, seed=0)

            if (vec[0]>JUMP[e-1]): # if we make a too big jump
                # we redo it until the jump is ok (so by a small rescaling of fe)
                # we divide the step by 2
                Fe_eff[i][e] = (Fe_eff[i][e]-Fe_eff[i][e-1])/2.+Fe_eff[i][e-1]
                if verbose:
                    print ("we rescale the fe vector [...]")
                # now we can re-enter the loop as the same e than entering..
            else: # we can run the rest
                if verbose:
                    print ("== the excitation level :", e+1," over ",feSim.size)
                    print ("== ---- the inhibition level :", i+1," over ",fiSim.size)
                for seed in range(1,SEED):
                    params['seed'] = seed
                    vec[seed],wec[seed] = single_experiment(t,\
                            Fe_eff[i][e]*(1-params['gei'])*params['pconnec']*params['Ntot'],\
                            fiSim[i]*params['gei']*params['pconnec']*params['Ntot'], params, seed=seed)
                    if verbose:
                        print ("== ---- _____________ seed :",seed)
                MEANfreq[i][e] = vec.mean()
                SDfreq[i][e] = vec.std()
                w[i][e]= wec.mean()
                del vec
                del wec
               # print 'RRRRRRRRRRRRRRRRRRRRRRRRRR', vec.mean(),wec.mean()
                if verbose:
                    print ("== ---- ===> Fout :",MEANfreq[i][e])
                if e<feSim.size-1: # we set the next value to the next one...
                    Fe_eff[i][e+1] = Fe_eff[i][e]+dFexc
                e = e+1 # and we progress in the fe loop
                
        # now we really finish the fe loop

    # then we save the results
    np.save(filename, np.array([MEANfreq, SDfreq, Fe_eff, fiSim, params,w]))
    print ('numerical TF data saved in :', filename)
    print ('The value of W is', w)

if __name__=='__main__':

    # First a nice documentation 
    parser=argparse.ArgumentParser(description=
     """ Runs two types of protocols on a given neuronal and network model
        1)  ==> Preliminary transfer function protocol ===
           to find the fixed point (with possibility to add external drive)
        2)  =====> Full transfer function protocol ==== 
           i.e. scanning the (fe,fi) space and getting the output frequency""",
              formatter_class=argparse.RawTextHelpFormatter)
    
    parser.add_argument("Neuron_Model",help="Choose a neuronal model from 'neuronal_models.py'")
    parser.add_argument("Network_Model",help="Choose a network model (synaptic and connectivity properties)"+\
                        "\n      from 'network_models'.py")

    parser.add_argument("--max_Fe",type=float, default=20.,\
                        help="Maximum excitatory frequency (default=30.)")
    parser.add_argument("--discret_Fe",type=int, default=30,\
                        help="Discretization of excitatory frequencies (default=9)")
    parser.add_argument("--lim_Fi", type=float, nargs=2, default=[0.,30.],\
                help="Limits for inhibitory frequency (default=[1.,20.])")
    parser.add_argument("--discret_Fi",type=int, default=15,\
               help="Discretization of inhibitory frequencies (default=8)")
    parser.add_argument("--max_Fout",type=float, default=30.,\
                         help="Minimum inhibitory frequency (default=30.)")
    parser.add_argument("--tstop",type=float, default=50.,\
                         help="tstop in s")
    parser.add_argument("--dt",type=float, default=5e-5,\
                         help="dt in s")
    parser.add_argument("--SEED",type=int, default=1,\
                  help="Seed for random number generation (default=1)")

    parser.add_argument("-s", "--save", help="save with the right name",
                         action="store_true")
    
    parser.add_argument("-v", "--verbose", help="increase output verbosity",
                         action="store_true")

    args = parser.parse_args()

    import sys
    sys.path.append('../')
    from single_cell_models.cell_library import get_neuron_params
    from synapses_and_connectivity.syn_and_connec_library import get_connectivity_and_synapses_matrix
    
    params = get_neuron_params(args.Neuron_Model, SI_units=True)
    M = get_connectivity_and_synapses_matrix(args.Network_Model, SI_units=True)

    reformat_syn_parameters(params, M) # merging those parameters

    if args.save:
        FILE = 'data/'+args.Neuron_Model+'_'+args.Network_Model+'.npy'
    else:
        FILE = 'data/example_data.npy'
        
    generate_transfer_function(params,\
                               verbose=True,
                               MAXfexc=args.max_Fe, 
                               MINfinh=args.lim_Fi[0], MAXfinh=args.lim_Fi[1],\
                               discret_exc=args.discret_Fe,discret_inh=args.discret_Fi,\
                               filename=FILE,
                               dt=args.dt, tstop=args.tstop,
                               MAXfout=args.max_Fout, SEED=args.SEED)

