"""
This file construct the equations for brian2
"""
from __future__ import print_function
import numpy as np
import brian2

import sys
sys.path.append('../')
from synapses_and_connectivity.syn_and_connec_library import get_connectivity_and_synapses_matrix

def get_membrane_equation(neuron_params, synaptic_array,\
                          return_equations=False):

    ## pure membrane equation
    if neuron_params['delta_v']==0:
        # if hard threshold : Integrate and Fire
        eqs = """
        dV/dt = (%(Gl)f*nS*(%(El)f*mV - V) + I - w_adapt)/(%(Cm)f*pF) : volt (unless refractory) """ % neuron_params
    else:
        eqs = """
        dV/dt = (%(Gl)f*nS*(%(El)f*mV*hetEl - V) + %(Gl)f*nS*%(delta_v)f*mV*exp(-(%(Vthre)f*mV-V)/(%(delta_v)f*mV)) + I - w_adapt)/(%(Cm)f*pF) : volt (unless refractory)
            
            hetEl : 1
         
            
            
            """ % neuron_params

    ## Adaptation current
    if neuron_params['tauw']>0: # adaptation current or not ?
        eqs += """
        dw_adapt/dt = ( -%(a)f*nS*( %(El)f*mV - V) - w_adapt )/(%(tauw)f*ms) : amp
            
        
            
         
            """ % neuron_params
            
    else:
        eqs += """
        w_adapt : amp  """

    ## synaptic currents, 1) adding all synaptic currents to the membrane equation via the I variable
    eqs += """
        I = I0 """
    for synapse in synaptic_array:
        # loop over each presynaptic element onto this target
        Gsyn = 'G'+synapse['name']
       
        eqs += '+'+Gsyn+'*(%(Erev)f*mV - V)' % synapse
    eqs += ' : amp'
    
    ## synaptic currents, 2) constructing the temporal dynamics of the synaptic conductances
    ## N.B. VALID ONLY FOR EXPONENTIAL SYNAPSES UNTIL NOW !!!!
    for synapse in synaptic_array:
        # loop over each presynaptic element onto this target
        Gsyn = 'G'+synapse['name']
        eqs += """
        """+'d'+Gsyn+'/dt = -'+Gsyn+'*(1./(%(Tsyn)f*ms)) : siemens' % synapse
    eqs += """
        I0 : amp """
    
    # adexp, pratical detection threshold Vthre+5*delta_v

    neurons = brian2.NeuronGroup(neuron_params['N'], model=eqs,\
                                     refractory=str(neuron_params['Trefrac'])+'*ms',
                                     threshold='V>'+str(neuron_params['Vthre']+5.*neuron_params['delta_v'])+'*mV',
                                     reset='V='+str(neuron_params['Vreset'])+'*mV; w_adapt+='+str(neuron_params['b'])+'*pA')
    

    neurons.hetEl=np.random.normal(1., 0., neuron_params['N'])

    
    print(eqs)
    if return_equations:
        return neurons, eqs
    else:
        return neurons

        
if __name__=='__main__':

    print(__doc__)
    
    # starting from an example

    from brian2 import *
    from cell_library import get_neuron_params
    import sys
    sys.path.append('../code/')
    from my_graph import set_plot
    '''
    for model, c in zip(['RS-cell', 'FS-cell'], ['g', 'r']):
        neurons, eqs =  get_membrane_equation(get_neuron_params(model), [],\
                                              return_equations=True)
        fig, ax = plt.subplots(figsize=(5,3))
        print('------------- NEURON model :', model)
        print(eqs)
        # V value initialization
        neurons.V = -65.*mV
        trace = StateMonitor(neurons, 'V', record=0)
        spikes = SpikeMonitor(neurons)
        run(100 * ms)
        neurons.I0 = 200*pA
        run(400 * ms)
        neurons.I0 = 0*pA
        run(200 * ms)
        # We draw nicer spikes
        V = trace[0].V[:]
        for t in spikes.t:
            plt.plot(t/ms*np.ones(2), [V[int(t/defaultclock.dt)]/mV+2,-10], '--', color=c)
        ax.plot(trace.t / ms, V / mV, color=c)
        
        ax.set_title(model)
        set_plot(ax, [])
    ax.annotate('-65mV', (20,-70))
    ax.plot([50], [-65], 'k>')
    ax.plot([100,150], [-50, -50], 'k-', lw=4)
    ax.plot([100,100], [-50, -40], 'k-', lw=4)
    ax.annotate('10mV', (200,-40))
    ax.annotate('50ms', (200,-50))
    show()

    '''
'''
    NTWK='CONFIG1'
    M = get_connectivity_and_synapses_matrix(NTWK, number=2)
    NRN_exc='FS-cell'
    
    # number of neurons
    Ne, Ni= int(M[0,0]['Ntot']*(1-M[0,0]['gei'])), int(M[0,0]['Ntot']*M[0,0]['gei'])

    exc_neurons, eqs = get_membrane_equation(get_neuron_params(NRN_exc, number=1), M[:,0], return_equations=True)
    
    neuron = exc_neurons
    
    neuron.V = 0.
    neuron.w_adapt = 0.
    mon = StateMonitor(neuron, ['V', 'w_adapt'], record=True)
    neuron.I0 = 2000*pA
    run_time = 5000*ms
    #run(run_time)
    

    
    
    
    ivect=np.arange(30,700,10)
    frvect=[]
    for i in range(len(ivect)):
        
        
        neuron.V = 0.
        neuron.w_adapt = 0.
        mon = StateMonitor(neuron,  ['V', 'w_adapt'], record=True)
        neuron.I0 = ivect[i]*pA
        
        #neuron.I0 = 1000*pA
        run_time = 2000*ms
        run(run_time)
        vv=mon.V[0]/mV
        tt=mon.t/ms
        cc=0
        for j in range(10,len(vv)-1):
            if(vv[j]<-49 and vv[j+1]>-49):
                cc+=1
    
        print("eee",1000*cc/(tt[-1]-tt[0]),ivect[i])

        #plt.plot(mon.t/ms, mon.V[0]/mV)
        #plt.show()
        frvect.append(1000*cc/(tt[-1]-tt[0]))


np.save('adexp',[ivect,frvect])

'''

    

    
