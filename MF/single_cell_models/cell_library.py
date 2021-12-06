"""
Some configuration of neuronal properties so that we pick up
within this file
"""
from __future__ import print_function

def get_neuron_params(NAME, name='', number=1, SI_units=False):

    if NAME=='LIF':
        params = {'name':name, 'N':number,\
                  'Gl':10., 'Cm':150.,'Trefrac':5.,\
                  'El':-60., 'Vthre':-50., 'Vreset':-60., 'delta_v':0.,\
                  'a':0., 'b': 0., 'tauw':1e9}
    elif NAME=='EIF':
        params = {'name':name, 'N':number,\
                  'Gl':10., 'Cm':150.,'Trefrac':5.,\
                  'El':-70., 'Vthre':-50., 'Vreset':-65., 'delta_v':2.,\
                  'a':0., 'b':0., 'tauw':1e9}
    elif NAME=='AdExp':
        params = {'name':name, 'N':number,\
                  'Gl':10., 'Cm':200.,'Trefrac':5.,\
                  'El':-70., 'Vthre':-50., 'Vreset':-65., 'delta_v':1.,\
                  'a':0., 'b':0., 'tauw':500.}

    elif NAME=='FS-cell':
        params = {'name':name, 'N':number,\
                  'Gl':10., 'Cm':200.,'Trefrac':5,\
                  'El':-65., 'Vthre':-50., 'Vreset':-65., 'delta_v':0.5,'ampnoise':0.,\
                  'a':0., 'b': 0., 'tauw':1}

    elif NAME=='FS-cell_seizure':
        params = {'name':name, 'N':number,\
                  'Gl':10., 'Cm':200.,'Trefrac':5,\
                  'El':-65., 'Vthre':-48., 'Vreset':-65., 'delta_v':0.5,'ampnoise':0.,\
                  'a':0., 'b': 0., 'tauw':1, 'vspike':-47.5}
     

    elif NAME=='FS-cell_seizure_diffsamp':
        params = {'name':name, 'N':number,\
                  'Gl':10., 'Cm':200.,'Trefrac':5,\
                  'El':-65., 'Vthre':-48., 'Vreset':-65., 'delta_v':0.5,'ampnoise':0.,\
                  'a':0., 'b': 0., 'tauw':1, 'vspike':-47.5}
   
    
    elif NAME=='FS_Eduarda':
        params = {'name':name, 'N':number,\
            'Gl':10., 'Cm':150.,'Trefrac':5,\
            'El':-65, 'Vthre':-50., 'Vreset':-65, 'delta_v':0.5,'ampnoise':0.,\
                'a':0., 'b':0., 'tauw':500.}
        
    elif NAME=='FS_Self_Sustain':
        params = {'name':name, 'N':number,\
            'Gl':10., 'Cm':200.,'Trefrac':5,\
            'El':-65, 'Vthre':-48., 'Vreset':-60, 'delta_v':0.5,'ampnoise':0.,\
                'a':0., 'b':0., 'tauw':10.}
        
    elif NAME=='FS_Jen':
        params = {'name':name, 'N':number,\
                  'Gl':10., 'Cm':200.,'Trefrac':5,\
                  'El':-65., 'Vthre':-50., 'Vreset':-65., 'delta_v':0.5,'ampnoise':0.,\
                  'a':0., 'b': 0., 'tauw':1}
    
    
    
        
    #####################################################################################
    
    ################        RS CELLS    #################################################
    
    #####################################################################################
        
    elif NAME=='RS-cell_seizure':
        params = {'name':name, 'N':number,\
                  'Gl':10., 'Cm':200.,'Trefrac':5,\
                  'El':-65., 'Vthre':-50., 'Vreset':-65., 'delta_v':2.,'ampnoise':0.,\
                  'a':0., 'b':100., 'tauw':1000.}
        
    elif NAME=='RS-cell_seizure_diffsamp':
        params = {'name':name, 'N':number,\
                  'Gl':10., 'Cm':200.,'Trefrac':5,\
                  'El':-65., 'Vthre':-50., 'Vreset':-65., 'delta_v':2.,'ampnoise':0.,\
                  'a':0., 'b':100., 'tauw':1000.}
        
    elif NAME=='RS-cell':
        params = {'name':name, 'N':number,\
                  'Gl':10., 'Cm':200.,'Trefrac':5,\
                  'El':-70., 'Vthre':-50., 'Vreset':-65., 'delta_v':2.,'ampnoise':0.,\
                  'a':0., 'b':60., 'tauw':1000.}
    
    elif NAME=='RS-cell_trial':
        params = {'name':name, 'N':number,\
                  'Gl':10., 'Cm':200.,'Trefrac':5,\
                  'El':-70., 'Vthre':-50., 'Vreset':-65., 'delta_v':2.,'ampnoise':0.,\
                  'a':0., 'b':60., 'tauw':1000.}
        
    elif NAME=='RS-cell0':
        params = {'name':name, 'N':number,\
                  'Gl':10., 'Cm':200.,'Trefrac':5.,\
                  'El':-65., 'Vthre':-50., 'Vreset':-65., 'delta_v':2.,\
                  'a':0., 'b':0., 'tauw':500.}
        
    elif NAME=='RS-cell2':
        params = {'name':name, 'N':number,\
                  'Gl':10., 'Cm':200.,'Trefrac':5.,\
                  'El':-65., 'Vthre':-50., 'Vreset':-65., 'delta_v':2.,\
                  'a':4., 'b':20., 'tauw':500.}

    elif NAME=='RS-cellbis':
        params = {'name':name, 'N':number,\
                'Gl':10., 'Cm':200.,'Trefrac':5,\
                'El':-65., 'Vthre':-50., 'Vreset':-65., 'delta_v':2.,'ampnoise':0.,\
                'a':0., 'b':100., 'tauw':500.}


    elif NAME=='RS-cell_Burst1_El60':
        params = {'name':name, 'N':number,\
            'Gl':10., 'Cm':200.,'Trefrac':5,\
                'El':-60., 'Vthre':-50., 'Vreset':-46., 'delta_v':2,'ampnoise':0.,\
                    'a':2., 'b':100., 'tauw':120.}
    
    elif NAME=='RS-cell_Burst1_El60_test':
        params = {'name':name, 'N':number,\
            'Gl':10., 'Cm':200.,'Trefrac':5,\
                'El':-60., 'Vthre':-50., 'Vreset':-46., 'delta_v':2,'ampnoise':0.,\
                    'a':2., 'b':100., 'tauw':120.}
        
    elif NAME=='RS-cell_Burst1_El58':
        params = {'name':name, 'N':number,\
            'Gl':10., 'Cm':200.,'Trefrac':5,\
                'El':-58., 'Vthre':-50., 'Vreset':-46., 'delta_v':2,'ampnoise':0.,\
                    'a':2., 'b':100., 'tauw':120.}
 
    elif NAME=='RS-cell_Burst2_b100':
        params = {'name':name, 'N':number,\
            'Gl':10., 'Cm':200.,'Trefrac':5,\
                'El':-60., 'Vthre':-54., 'Vreset':-46., 'delta_v':2,'ampnoise':0.,\
                    'a':2., 'b':100., 'tauw':5000., 'vspike':-40} 
        
    elif NAME=='RS-cell_Burst2_b50':
        params = {'name':name, 'N':number,\
            'Gl':10., 'Cm':200.,'Trefrac':5,\
                'El':-60., 'Vthre':-54., 'Vreset':-46., 'delta_v':2,'ampnoise':0.,\
                    'a':2., 'b':50., 'tauw':5000., 'vspike':-40}
        
    elif NAME=='RS-cell_BurstNet':
        params = {'name':name, 'N':number,\
            'Gl':10., 'Cm':200.,'Trefrac':5,\
                'El':-60., 'Vthre':-50., 'Vreset':-46., 'delta_v':2,'ampnoise':0.,\
                    'a':5., 'b':10., 'tauw':1000., 'vspike':-40}  
      
        
    elif NAME=='RS-cell_UD':
        params = {'name':name, 'N':number,\
            'Gl':10., 'Cm':200.,'Trefrac':5,\
                'El':-63., 'Vthre':-50., 'Vreset':-65., 'delta_v':2,'ampnoise':0.,\
                    'a':0., 'b':0., 'tauw':500.}

    
    elif NAME=='RS_Eduarda':
        params = {'name':name, 'N':number,\
            'Gl':10., 'Cm':150.,'Trefrac':5,\
            'El':-65., 'Vthre':-50., 'Vreset':-65., 'delta_v':2.,'ampnoise':0.,\
                'a':4., 'b':20., 'tauw':500.}
    
    elif NAME=='RS_Ed_NoAd':
        params = {'name':name, 'N':number,\
            'Gl':10., 'Cm':150.,'Trefrac':5,\
            'El':-65., 'Vthre':-50., 'Vreset':-65., 'delta_v':2.,'ampnoise':0.,\
                'a':0., 'b':0., 'tauw':500.}
        
        
    elif NAME=='RS-cell_Try':
        params = {'name':name, 'N':number,\
            'Gl':10., 'Cm':200.,'Trefrac':5,\
            'El':-58.73, 'Vthre':-50., 'Vreset':-58.73, 'delta_v':2,'ampnoise':0.,\
                'a':0., 'b':0., 'tauw':500.}

    elif NAME=='RS_Self_Sustain':
        params = {'name':name, 'N':number,\
            'Gl':10., 'Cm':200.,'Trefrac':5,\
            'El':-60, 'Vthre':-50., 'Vreset':-60, 'delta_v':2,'ampnoise':0.,\
                'a':0., 'b':100., 'tauw':150.}
        
    elif NAME=='RS_Jen_Original':
        params = {'name':name, 'N':number,\
                  'Gl':10., 'Cm':200.,'Trefrac':5.,\
                  'El':-65., 'Vthre':-50., 'Vreset':-65., 'delta_v':2.,\
                  'a':0., 'b':0., 'tauw':500.}
        
    elif NAME=='RS_Jen':
        params = {'name':name, 'N':number,\
            'Gl':10., 'Cm':200.,'Trefrac':5,\
            'El':-58.0, 'Vthre':-50., 'Vreset':-60, 'delta_v':2,'ampnoise':0.,\
                'a':0., 'b':100., 'tauw':500.}

    else:
        print('====================================================')
        print('------------ CELL NOT RECOGNIZED !! ---------------')
        print('====================================================')






    if SI_units:
        #print('cell parameters in SI units')
        # mV to V
        params['El'], params['Vthre'], params['Vreset'], params['delta_v'] =\
            1e-3*params['El'], 1e-3*params['Vthre'], 1e-3*params['Vreset'], 1e-3*params['delta_v']
        
        if 'vspike' in params : params['vspike'] = 1e-3*params['vspike']
        # ms to s
        params['Trefrac'], params['tauw'] = 1e-3*params['Trefrac'], 1e-3*params['tauw']
        # nS to S
        params['a'], params['Gl'] = 1e-9*params['a'], 1e-9*params['Gl']
        # pF to F and pA to A
        params['Cm'], params['b'] = 1e-12*params['Cm'], 1e-12*params['b']
    else:
        print('cell parameters --NOT-- in SI units')
        
    return params.copy()

if __name__=='__main__':

    print(__doc__)
