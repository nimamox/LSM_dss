import numpy as np
import pickle 
import subprocess

for w_scale in np.linspace(7, 9, 200):
    params = {
        'delay': (1, 20),
        'W_inp': (4, 8),
        'delay_inp': (1, 20),
    
        'liquid_geometry': (5, 5, 8),
        'exc_inh_ratio': .8,
        'inp_size': 1,
        'liq_inp_size': 10,
        'readout_size': 5,
        'lambda_': 3.5,
        'W_Scale': w_scale,
    
        'dt': .1,
        'seed': 1337
    }
    
    with open('params.conf', 'wb') as fo:
        pickle.dump(params, fo)
        subprocess.call("python3 main.py params.conf", shell=True)
        
        