import matplotlib.pylab as plt
import numpy as np
import matplotlib.pyplot as plt


from scipy.optimize import minimize
import matplotlib.pyplot as plt

import numpy as np
import sys
sys.path.append('../')
from single_cell_models.cell_library import get_neuron_params
from synapses_and_connectivity.syn_and_connec_library import get_connectivity_and_synapses_matrix
from theoretical_tools import pseq_params,TF_my_templateup
from theoretical_tools import*
from tf_simulation import reformat_syn_parameters





# NTWK
M = get_connectivity_and_synapses_matrix('CONFIG1', SI_units=True)
    
# NRN1
params = get_neuron_params('RS-cellbis', SI_units=True)
reformat_syn_parameters(params, M)

a, b, tauw = params['a'],params['b'], params['tauw']
Qe, Te, Ee = params['Qe'], params['Te'], params['Ee']
Qi, Ti, Ei = params['Qi'], params['Ti'], params['Ei']
Gl, Cm , El = params['Gl'], params['Cm'] , params['El']
pconnec,Ntot,gei,ext_drive=params['pconnec'], params['Ntot'] , params['gei'],M[0,0]['ext_drive']
P = np.load('../transfer_functions/data/RS-cellbis_CONFIG1_fit.npy')
#P1 = np.load('../transfer_functions/data/'+NRN1+'_'+NTWK+'_fit.npy')

   
muGn = 1.
        
params['P'] = P



P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10=P[0], P[1], P[2], P[3], P[4], P[5], P[6], P[7], P[8], P[9], P[10]



muV = np.linspace(-80e-3, -50e-3, 100)
sV=4e-3
TvN=0.5
Vthre = threshold_func(muV, sV, TvN, muGn,P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10)
    
    
    
Fout_th = erfc_func(muV, sV, TvN, Vthre, Gl, Cm)
plt.plot(muV,erfc_func(muV, sV, TvN, Vthre, Gl, Cm))
plt.show()

    
