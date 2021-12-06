import numpy as np
import sys
sys.path.append('../')
from single_cell_models.cell_library import get_neuron_params
from synapses_and_connectivity.syn_and_connec_library import get_connectivity_and_synapses_matrix
from theoretical_tools import pseq_params,TF_my_templateup,TF_my_templateup_heterogeneity
from tf_simulation import reformat_syn_parameters


P1 = np.load('../transfer_functions/data/RS-cell_CONFIG1_fit.npy')

print("P values",P1)


P1[0]=-0.04983106
#it was -0.05023106
        
filename='../transfer_functions/data/RS-cell_CONFIG1_fit.npy'
np.save(filename, np.array(P1))
