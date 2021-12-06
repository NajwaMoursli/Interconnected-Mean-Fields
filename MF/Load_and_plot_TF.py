#!/usr/bin/env python
# coding: utf-8

# #### In this code we show how to load and plot the TF of inhibitory and excitatory neuron models


''' We start by loading the necessary components. You can go through the load_transfer_functions routine, although we don't recommend you to spend too much time following the thread, as there are lots of nested structures in this architecture.  '''

import numpy as np
import sys
#sys.path.append('mean_field_adapt')

from transfer_functions.load_config import load_transfer_functions

import matplotlib.pyplot as plt

''' Here we define the so-called configuration (architecture of the network), and the neuron models under considerations. These names (CONFIG1 and RS/FS-cell) refer to sets of parameters defined respecitvely in the files :

- synapses_and_connectivity/syn_and_connec_library.py
- single_cell_models/cell_library.py
'''

NTWK='CONFIG1'


NRN1='RS-cell'
NRN2='FS-cell'


TF_temp = load_transfer_functions(NRN1, NRN2, NTWK)


TF=(TF_temp[0],TF_temp[1]) # TF defined as a tuple

''' Now we want to plot them. The TF under consideration both take three arguments :

- Excitatory input frequency
- Inhibitory input frequency
- Adaptation

All these variables are defined at the population level. That is, they must compare with the population firing rates observable in the network simulations ! Note also that, as no adaptation is considered so far for inhibitory neurons, it is set to zero. '''


### First, build the TFs ###

TF1_test=np.zeros((201,201))
TF2_test=np.zeros((201,201))

for ve in np.linspace(0,200,201):
    for vi in np.linspace(0,200,201):
        #for w in np.linspace(0,200,21):
            
            w=ve/10.*60e-12 ## steady state value according to MF equations, you can try to tune it !
            
            index_ve=int(ve)
            index_vi=int(vi)
            
            #print(ve,vi)
            
            TF1_test[index_ve][index_vi]=TF[0](ve/10,vi/10,w)
            TF2_test[index_ve][index_vi]=TF[1](ve/10,vi/10,0) ## Adaptation for inhibitory population is set to 0
            
            #print('\n \n INPUTS = ',ve,vi,'\n','\n')


### Now plot them ###


fig2=plt.figure(figsize=(20,12))
ax21=fig2.add_subplot(221)
ax22=fig2.add_subplot(222)

#v=np.linspace(0,200,201)
#x,y=np.meshgrid(v,v)

im21=ax21.imshow(TF1_test, cmap='jet',interpolation='nearest',extent=[3.5,8.5,5,0])
ax21.set_ylabel("Excitatory Frequencies")
ax21.invert_yaxis()
ax21.set_xlabel("Inhibitory Frequencies")
ax21.set_title("Transfer function RS cell")
fig2.colorbar(im21, ax=ax21)

im22=ax22.imshow(TF2_test, cmap='jet', interpolation='nearest',extent=[3.5,8.5,5,0])
ax22.set_ylabel("Excitatory Frequencies")
ax22.invert_yaxis()
ax22.set_xlabel("Inhibitory Frequencies")
ax22.set_title("Transfer function FS cell")
fig2.colorbar(im22,ax=ax22)

plt.show()




