import numpy as np
import sys
#sys.path.append('../')
from single_cell_models.cell_library import get_neuron_params
from synapses_and_connectivity.syn_and_connec_library import get_connectivity_and_synapses_matrix
from transfer_functions.new_fit import pseq_params,TF_my_templateup,TF_my_templateup2,TF_my_templateup_heterogeneity_effective_p
from transfer_functions.New_tf import reformat_syn_parameters
import os





def load_transfer_functions(NRN1, NRN2, NTWK):
    """
    returns the two transfer functions of the mean field model
    """
    
    myCmd = 'pwd'
    os.system(myCmd)

    dirpath = os.getcwd()
    print("current directory is : " + dirpath)
    
    # NTWK
    M = get_connectivity_and_synapses_matrix(NTWK, SI_units=True)
    print ('NRN1=',NRN1,'  NRN2=',NRN2,'   NTWK=',NTWK)
    # NRN1
    params1 = get_neuron_params(NRN1, SI_units=True)
    reformat_syn_parameters(params1, M)
    try:

        
        P1 = np.load('transfer_functions/data/'+NRN1+'_'+NTWK+'_fit.npy')
       
        
        params1['P'] = P1
        print("paramsAAA",P1)
        def TF1(fe, fi,XX):
            return TF_my_templateup(fe, fi,XX, *pseq_params(params1))
        

        



    
    except IOError:
        print('=======================================================')
        print('=====  fit for NRN1 not available  ====================')
        print('=======================================================')

    # NRN1
    params2 = get_neuron_params(NRN2, SI_units=True)
    reformat_syn_parameters(params2, M)
    try:

        P2 = np.load('transfer_functions/data/'+NRN2+'_'+NTWK+'_fit.npy')
    

        
        
        params2['P'] = P2
        def TF2(fe, fi,XX):
            return TF_my_templateup(fe, fi,XX, *pseq_params(params2))

    except IOError:
        print('=======================================================')
        print('=====  fit for NRN2 not available  ====================')
        print('=======================================================')
        
    return TF1, TF2

def load_transfer_functions2(NRN1, NRN2, NTWK):
    """
    returns the two transfer functions of the mean field model
    """
    
    myCmd = 'pwd'
    os.system(myCmd)

    dirpath = os.getcwd()
    print("current directory is : " + dirpath)
    
    # NTWK
    M = get_connectivity_and_synapses_matrix(NTWK, SI_units=True)
    print ('NRN1=',NRN1,'  NRN2=',NRN2,'   NTWK=',NTWK)
    # NRN1
    params1 = get_neuron_params(NRN1, SI_units=True)
    reformat_syn_parameters(params1, M)
    try:

        
        P1 = np.load('transfer_functions/data/'+NRN1+'_'+NTWK+'_fit.npy')
       
        
        params1['P'] = P1
        print("paramsAAA",P1)
        def TF1b(fe, fi,XX):
            return TF_my_templateup2(fe, fi,XX, *pseq_params(params1))
        

        



    
    except IOError:
        print('=======================================================')
        print('=====  fit for NRN1 not available  ====================')
        print('=======================================================')

    # NRN1
    params2 = get_neuron_params(NRN2, SI_units=True)
    reformat_syn_parameters(params2, M)
    try:

        P2 = np.load('transfer_functions/data/'+NRN2+'_'+NTWK+'_fit.npy')
    

        
        
        params2['P'] = P2
        def TF2b(fe, fi,XX):
            return TF_my_templateup2(fe, fi,XX, *pseq_params(params2))

    except IOError:
        print('=======================================================')
        print('=====  fit for NRN2 not available  ====================')
        print('=======================================================')
        
    return TF1b, TF2b









def load_transfer_functions_withparam(NRN1, NRN2, NTWK,qe,el):
    """
        returns the two transfer functions of the mean field model
        """
    
    # NTWK
    M = get_connectivity_and_synapses_matrix(NTWK, SI_units=True)

    # NRN1
    params1 = get_neuron_params(NRN1, SI_units=True)
    
    
    M[0,0]['Q'],M[0,1]['Q']=qe*1.e-09,qe*1.e-09
    
    
    params1['El']=el*0.001
    
    reformat_syn_parameters(params1, M)
    try:

        
        P1 = np.load('../transfer_functions/data/RS-cell_CONFIG1_fit.npy')
        
        
        params1['P'] = P1
        def TF1(fe, fi,XX):
            return TF_my_templateup(fe, fi,XX, *pseq_params(params1))
        

    
    
    
    
    except IOError:
        print('=======================================================')
        print('=====  fit for NRN1 not available  ====================')
        print('=======================================================')

    # NRN1
    params2 = get_neuron_params(NRN2, SI_units=True)
    reformat_syn_parameters(params2, M)
    M[0,0]['Q'],M[0,1]['Q']=qe*1.e-09,qe*1.e-09
    
    
    params1['El']=el*0.001
    
    try:

        P2 = np.load('../transfer_functions/data/FS-cell_CONFIG1_fit.npy')
        
        
        #P2[0]=-0.05149122 #this ensures to have a better fit#
        
        
        params2['P'] = P2
        def TF2(fe, fi,XX):
            return TF_my_templateup(fe, fi,XX, *pseq_params(params2))

    except IOError:
        print('=======================================================')
        print('=====  fit for NRN2 not available  ====================')
        print('=======================================================')
    
    return TF1, TF2






def load_transfer_functions_withparamElIE(NRN1, NRN2, NTWK,el,elI):
    """
        returns the two transfer functions of the mean field model
        """
    
    # NTWK
    M = get_connectivity_and_synapses_matrix(NTWK, SI_units=True)

    # NRN1
    params1 = get_neuron_params(NRN1, SI_units=True)
    
    

    
    
    params1['El']=el*0.001
    
    reformat_syn_parameters(params1, M)
    try:

        
        P1 = np.load('../transfer_functions/data/RS-cell_CONFIG1_fit.npy')
        
        
        params1['P'] = P1
        def TF1(fe, fi,XX):
            return TF_my_templateup(fe, fi,XX, *pseq_params(params1))
        

    
    
    
    
    except IOError:
        print('=======================================================')
        print('=====  fit for NRN1 not available  ====================')
        print('=======================================================')

    # NRN1
    params2 = get_neuron_params(NRN2, SI_units=True)
    reformat_syn_parameters(params2, M)

    
    
    params2['El']=elI*0.001
    try:

        P2 = np.load('../transfer_functions/data/FS-cell_CONFIG1_fit.npy')
        
        

        
        
        params2['P'] = P2
        def TF2(fe, fi,XX):
            return TF_my_templateup(fe, fi,XX, *pseq_params(params2))

    except IOError:
        print('=======================================================')
        print('=====  fit for NRN2 not available  ====================')
        print('=======================================================')
    
    return TF1, TF2


def load_transfer_functions_with_ALL_params(NRN1, NRN2, NTWK,par):
    """
        returns the two transfer functions of the mean field model
        """

    ''' Here we have two sets of parameters : params and par
    
    The first one refers to the model for which the coefficients of the TF have been evaluated, the second refers to 
    the actual neuron model for which the TF is now computed. If they are different, par['diff']=True
    
    par comes in one dictionary, sometimes separated into two entries corresponding to each neuron model
    
    for network features, only one entry
    
    Pop sizes are not accounted for here because they matter in the mutual interactions, and are thus included in the MF 
    routine.
    
    '''
    
    
    # NTWK
    M = get_connectivity_and_synapses_matrix(NTWK, SI_units=True)


    M[0,0]['Q'],M[0,1]['Q']=par['qe'],par['qe']
    M[1,0]['Q'],M[1,1]['Q']=par['qi'],par['qi']
        
    M[0,0]['Tsyn'],M[0,1]['Tsyn']=par['tau_e'],par['tau_e']
    M[1,0]['Tsyn'],M[1,1]['Tsyn']=par['tau_i'],par['tau_i']
        
    M[0,0]['p_conn'],M[0,1]['p_conn']=par['p_conn'][0,0],par['p_conn'][0,1]
    M[1,0]['p_conn'],M[1,1]['p_conn']=par['p_conn'][1,0],par['p_conn'][1,1]

    # NRN1
    params1 = get_neuron_params(NRN1, SI_units=True)
    
    if (par['diff'][0]==True):
          
        params1['El']=par['El'][0]
        params1['Cm']=par['Cm'][0]
        params1['Gl']=par['Gl'][0]
        params1['Vthre']=par['Vthre'][0]
        params1['b']=par['b'][0]
        params1['a']=par['a'][0]
        params1['tauw']=par['tauw'][0]# here divide by 1000 because tauw is in seconds in MF codes

    
    reformat_syn_parameters(params1, M)
    
    try:

        
        P1 = np.load('transfer_functions/data/'+NRN1+'_'+NTWK+'_fit.npy')
        
        
        params1['P'] = P1
        def TF1(fe, fi,XX):
            return TF_my_templateup(fe, fi,XX, *pseq_params(params1))
        

    
    
    
    
    except IOError:
        print('=======================================================')
        print('=====  fit for NRN1 not available  ====================')
        print('=======================================================')

    # NRN1
    params2 = get_neuron_params(NRN2, SI_units=True)
    
    if (par['diff'][1]==True):
        
        params2['El']=par['El'][1]
        params2['Cm']=par['Cm'][1]
        params2['Gl']=par['Gl'][1]
        params2['Vthre']=par['Vthre'][1]
        params2['b']=par['b'][1]
        params2['a']=par['a'][1]
        params2['tauw']=par['tauw'][1]
    
    
    
    reformat_syn_parameters(params2, M)
   
    
    try:

        P2 = np.load('transfer_functions/data/'+NRN2+'_'+NTWK+'_fit.npy')
        
        
        #P2[0]=-0.05149122 #this ensures to have a better fit#
        
        
        params2['P'] = P2
        def TF2(fe, fi,XX):
            return TF_my_templateup(fe, fi,XX, *pseq_params(params2))

    except IOError:
        print('=======================================================')
        print('=====  fit for NRN2 not available  ====================')
        print('=======================================================')
    
    return TF1, TF2



def load_transfer_functions_with_ALL_params_het_exc(NRN1, NRN2, NTWK,par):
    """
        returns the two transfer functions of the mean field model
        """

    ''' Here we have two sets of parameters : params and par
    
    The first one refers to the model for which the coefficients of the TF have been evaluated, the second refers to 
    the actual neuron model for which the TF is now computed. If they are different, par['diff']=True
    
    par comes in one dictionary, sometimes separated into two entries corresponding to each neuron model
    
    for network features, only one entry
    
    Pop sizes are not accounted for here because they matter in the mutual interactions, and are thus included in the MF 

    routine.
    
    '''
    
    
    # NTWK
    M = get_connectivity_and_synapses_matrix(NTWK, SI_units=True)


    M[0,0]['Q'],M[0,1]['Q']=par['qe'],par['qe']
    M[1,0]['Q'],M[1,1]['Q']=par['qi'],par['qi']
        
    M[0,0]['Tsyn'],M[0,1]['Tsyn']=par['tau_e'],par['tau_e']
    M[1,0]['Tsyn'],M[1,1]['Tsyn']=par['tau_i'],par['tau_i']
        
    M[0,0]['p_conn'],M[0,1]['p_conn']=par['p_conn'][0,0],par['p_conn'][0,1]
    M[1,0]['p_conn'],M[1,1]['p_conn']=par['p_conn'][1,0],par['p_conn'][1,1]

    # NRN1
    params1 = get_neuron_params(NRN1, SI_units=True)
    
    if (par['diff'][0]==True):
          
        params1['El']=par['El'][0]
        params1['Cm']=par['Cm'][0]
        params1['Gl']=par['Gl'][0]
        params1['Vthre']=par['Vthre'][0]
        params1['b']=par['b'][0]
        params1['a']=par['a'][0]
        params1['tauw']=par['tauw'][0]# here divide by 1000 because tauw is in seconds in MF codes

    
    reformat_syn_parameters(params1, M)
    
    try:

        
        P1 = np.load('transfer_functions/data/'+NRN1+'_'+NTWK+'_fit.npy')
        
        
        params1['P'] = P1
        def TF1(fe, fi,XX):
            return TF_my_templateup_heterogeneity_effective_p(fe, fi,XX, *pseq_params(params1))
        

    
    
    
    
    except IOError:
        print('=======================================================')
        print('=====  fit for NRN1 not available  ====================')
        print('=======================================================')

    # NRN1
    params2 = get_neuron_params(NRN2, SI_units=True)
    
    if (par['diff'][1]==True):
        
        params2['El']=par['El'][1]
        params2['Cm']=par['Cm'][1]
        params2['Gl']=par['Gl'][1]
        params2['Vthre']=par['Vthre'][1]
        params2['b']=par['b'][1]
        params2['a']=par['a'][1]
        params2['tauw']=par['tauw'][1]
    
    
    
    reformat_syn_parameters(params2, M)
   
    
    try:

        P2 = np.load('transfer_functions/data/'+NRN2+'_'+NTWK+'_fit.npy')
        
        
        #P2[0]=-0.05149122 #this ensures to have a better fit#
        
        
        params2['P'] = P2
        def TF2(fe, fi,XX):
            return TF_my_templateup(fe, fi,XX, *pseq_params(params2))

    except IOError:
        print('=======================================================')
        print('=====  fit for NRN2 not available  ====================')
        print('=======================================================')
    
    return TF1, TF2



def load_transfer_functions_with_ALL_params_het_inh(NRN1, NRN2, NTWK,par):
    """
        returns the two transfer functions of the mean field model
        """

    ''' Here we have two sets of parameters : params and par
    
    The first one refers to the model for which the coefficients of the TF have been evaluated, the second refers to 
    the actual neuron model for which the TF is now computed. If they are different, par['diff']=True
    
    par comes in one dictionary, sometimes separated into two entries corresponding to each neuron model
    
    for network features, only one entry
    
    Pop sizes are not accounted for here because they matter in the mutual interactions, and are thus included in the MF 
    routine.
    
    '''
    
    
    # NTWK
    M = get_connectivity_and_synapses_matrix(NTWK, SI_units=True)


    M[0,0]['Q'],M[0,1]['Q']=par['qe'],par['qe']
    M[1,0]['Q'],M[1,1]['Q']=par['qi'],par['qi']
        
    M[0,0]['Tsyn'],M[0,1]['Tsyn']=par['tau_e'],par['tau_e']
    M[1,0]['Tsyn'],M[1,1]['Tsyn']=par['tau_i'],par['tau_i']
        
    M[0,0]['p_conn'],M[0,1]['p_conn']=par['p_conn'][0,0],par['p_conn'][0,1]
    M[1,0]['p_conn'],M[1,1]['p_conn']=par['p_conn'][1,0],par['p_conn'][1,1]

    # NRN1
    params1 = get_neuron_params(NRN1, SI_units=True)
    
    if (par['diff'][0]==True):
          
        params1['El']=par['El'][0]
        params1['Cm']=par['Cm'][0]
        params1['Gl']=par['Gl'][0]
        params1['Vthre']=par['Vthre'][0]
        params1['b']=par['b'][0]
        params1['a']=par['a'][0]
        params1['tauw']=par['tauw'][0]# here divide by 1000 because tauw is in seconds in MF codes

    
    reformat_syn_parameters(params1, M)
    
    try:

        
        P1 = np.load('transfer_functions/data/'+NRN1+'_'+NTWK+'_fit.npy')
        
        
        params1['P'] = P1
        def TF1(fe, fi,XX):
            return TF_my_templateup(fe, fi,XX, *pseq_params(params1))
        

    
    
    
    
    except IOError:
        print('=======================================================')
        print('=====  fit for NRN1 not available  ====================')
        print('=======================================================')

    # NRN1
    params2 = get_neuron_params(NRN2, SI_units=True)
    
    if (par['diff'][1]==True):
        
        params2['El']=par['El'][1]
        params2['Cm']=par['Cm'][1]
        params2['Gl']=par['Gl'][1]
        params2['Vthre']=par['Vthre'][1]
        params2['b']=par['b'][1]
        params2['a']=par['a'][1]
        params2['tauw']=par['tauw'][1]
    
    
    
    reformat_syn_parameters(params2, M)
   
    
    try:

        P2 = np.load('transfer_functions/data/'+NRN2+'_'+NTWK+'_fit.npy')
        
        
        #P2[0]=-0.05149122 #this ensures to have a better fit#
        
        
        params2['P'] = P2
        def TF2(fe, fi,XX):
            return TF_my_templateup_heterogeneity_effective_p(fe, fi,XX, *pseq_params(params2))

    except IOError:
        print('=======================================================')
        print('=====  fit for NRN2 not available  ====================')
        print('=======================================================')
    
    return TF1, TF2


def load_transfer_functions_with_ALL_params_het_exc_inh(NRN1, NRN2, NTWK,par):
    """
        returns the two transfer functions of the mean field model
        """

    ''' Here we have two sets of parameters : params and par
    
    The first one refers to the model for which the coefficients of the TF have been evaluated, the second refers to 
    the actual neuron model for which the TF is now computed. If they are different, par['diff']=True
    
    par comes in one dictionary, sometimes separated into two entries corresponding to each neuron model
    
    for network features, only one entry
    
    Pop sizes are not accounted for here because they matter in the mutual interactions, and are thus included in the MF 
    routine.
    
    '''
    
    
    # NTWK
    M = get_connectivity_and_synapses_matrix(NTWK, SI_units=True)


    M[0,0]['Q'],M[0,1]['Q']=par['qe'],par['qe']
    M[1,0]['Q'],M[1,1]['Q']=par['qi'],par['qi']
        
    M[0,0]['Tsyn'],M[0,1]['Tsyn']=par['tau_e'],par['tau_e']
    M[1,0]['Tsyn'],M[1,1]['Tsyn']=par['tau_i'],par['tau_i']
        
    M[0,0]['p_conn'],M[0,1]['p_conn']=par['p_conn'][0,0],par['p_conn'][0,1]
    M[1,0]['p_conn'],M[1,1]['p_conn']=par['p_conn'][1,0],par['p_conn'][1,1]

    # NRN1
    params1 = get_neuron_params(NRN1, SI_units=True)
    
    if (par['diff'][0]==True):
          
        params1['El']=par['El'][0]
        params1['Cm']=par['Cm'][0]
        params1['Gl']=par['Gl'][0]
        params1['Vthre']=par['Vthre'][0]
        params1['b']=par['b'][0]
        params1['a']=par['a'][0]
        params1['tauw']=par['tauw'][0]# here divide by 1000 because tauw is in seconds in MF codes

    
    reformat_syn_parameters(params1, M)
    
    try:

        
        P1 = np.load('transfer_functions/data/'+NRN1+'_'+NTWK+'_fit.npy')
        
        
        params1['P'] = P1
        def TF1(fe, fi,XX):
            return TF_my_templateup_heterogeneity_effective_p(fe, fi,XX, *pseq_params(params1))
        

    
    
    
    
    except IOError:
        print('=======================================================')
        print('=====  fit for NRN1 not available  ====================')
        print('=======================================================')

    # NRN1
    params2 = get_neuron_params(NRN2, SI_units=True)
    
    if (par['diff'][1]==True):
        
        params2['El']=par['El'][1]
        params2['Cm']=par['Cm'][1]
        params2['Gl']=par['Gl'][1]
        params2['Vthre']=par['Vthre'][1]
        params2['b']=par['b'][1]
        params2['a']=par['a'][1]
        params2['tauw']=par['tauw'][1]
    
    
    
    reformat_syn_parameters(params2, M)
   
    
    try:

        P2 = np.load('transfer_functions/data/'+NRN2+'_'+NTWK+'_fit.npy')
        
        
        #P2[0]=-0.05149122 #this ensures to have a better fit#
        
        
        params2['P'] = P2
        def TF2(fe, fi,XX):
            return TF_my_templateup_heterogeneity_effective_p(fe, fi,XX, *pseq_params(params2))

    except IOError:
        print('=======================================================')
        print('=====  fit for NRN2 not available  ====================')
        print('=======================================================')
    
    return TF1, TF2

def load_transfer_functions_with_ALL_params_ij(NRN1, NRN2, NTWK,par,i,j):
    """
        ij refers to the indices of the population for multi MF
   
    """
   
   
    # NTWK
    M = get_connectivity_and_synapses_matrix(NTWK, SI_units=True)


    M[0,0]['Q'],M[0,1]['Q']=par['qe'],par['qe']
    M[1,0]['Q'],M[1,1]['Q']=par['qi'],par['qi']
       
    M[0,0]['Tsyn'],M[0,1]['Tsyn']=par['tau_e'],par['tau_e']
    M[1,0]['Tsyn'],M[1,1]['Tsyn']=par['tau_i'],par['tau_i']
       
    #M[0,0]['p_conn'],M[0,1]['p_conn']=par['p_conn'][0,0],par['p_conn'][0,1]
    #M[1,0]['p_conn'],M[1,1]['p_conn']=par['p_conn'][1,0],par['p_conn'][1,1]


    ''' For now we remove the connectivity parameters, are they are supposed to be taken into account outside, here it creates a mess !'''

    # NRN1
    params1 = get_neuron_params(NRN1, SI_units=True)
   
    if (par['diff'][i]==True):
         
        params1['El']=par['El'][i]
        params1['Cm']=par['Cm'][i]
        params1['Gl']=par['Gl'][i]
        params1['Vthre']=par['Vthre'][i]
        params1['b']=par['b'][i]
        params1['a']=par['a'][i]
        params1['tauw']=par['tauw'][i]# here divide by 1000 because tauw is in seconds in MF codes

   
    reformat_syn_parameters(params1, M)
   
    try:

       
        P1 = np.load('transfer_functions/data/'+NRN1+'_'+NTWK+'_fit.npy')
       
       
        params1['P'] = P1
        def TF1(fe, fi,XX):
            return TF_my_templateup(fe, fi,XX, *pseq_params(params1))
       

   
   
   
   
    except IOError:
        print('=======================================================')
        print('=====  fit for NRN1 not available  ====================')
        print('=======================================================')

    # NRN1
    params2 = get_neuron_params(NRN2, SI_units=True)
   
    if (par['diff'][j]==True):
       
        params2['El']=par['El'][j]
        params2['Cm']=par['Cm'][j]
        params2['Gl']=par['Gl'][j]
        params2['Vthre']=par['Vthre'][j]
        params2['b']=par['b'][j]
        params2['a']=par['a'][j]
        params2['tauw']=par['tauw'][j]
   
   
   
    reformat_syn_parameters(params2, M)
   
   
    try:

        P2 = np.load('transfer_functions/data/'+NRN2+'_'+NTWK+'_fit.npy')
       
       
        #P2[0]=-0.05149122 #this ensures to have a better fit#
       
       
        params2['P'] = P2
        def TF2(fe, fi,XX):
            return TF_my_templateup(fe, fi,XX, *pseq_params(params2))

    except IOError:
        print('=======================================================')
        print('=====  fit for NRN2 not available  ====================')
        print('=======================================================')
   
    return TF1, TF2
