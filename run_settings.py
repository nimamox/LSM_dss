import numpy as np
import pickle 
import subprocess

for w_scale in np.linspace(7, 7, 1):
    params = {
        'inp_size': 1,
        'W_inp': 7,
        'delay_inp': 1,

        'lambda_': 1,
        'W_Scale': w_scale,
        'delay': 1,

        'liquid_geometry': (5, 5, 8),
        'exc_inh_ratio': .8,
        
        'liq_inp_size': 20,
        'readout_size': 5,

        'dt': .1,
        'seed': 1337
    }
    print('*** W_scale:', w_scale)
    with open('params.conf', 'wb') as fo:
        pickle.dump(params, fo)
    subprocess.call("python3 main.py params.conf", shell=True)
        
