import numpy as np
import ANNarchy as AN
from pops import create_populations
from projs import create_projections
import random



####################################### PARAMETERS

params = {
    'liquid_geometry': (5, 5, 8),
    'exc_inh_ratio': .8,
    'inp_size': 1,
    'readout_size': 5,
    'lambda_': 3,
    'W_Scale': 14.2,
    'W_inp_liq': 5.0,
    'dt': .1,
    'seed': 1337
}


np.random.seed(params['seed'])
random.seed(params['seed'])
AN.setup(dt=params['dt'], seed=params['seed'])


####################################### LIQUID POPULATION

network = create_populations(params)


print("#Nodes in RC: {},  Exc-Inh ratio: {},  #Exc: {},  #Inp: {}".format(network['liquid_pop'].size, 
                                        params['exc_inh_ratio'], network['liquid_exc'].size, params['inp_size']))


######################################## NETWORK PARAMETERS

proj_params = {
    'delay': 1,
    
    'U_ee': 0.5, #STP params
    'U_ei': 0.05,
    'U_ie': 0.25,
    'U_ii': 0.32,
    
    'tau_rec_ee': 1.1,
    'tau_rec_ei': 0.125,
    'tau_rec_ie': 0.7,
    'tau_rec_ii': 0.144,
    
    'tau_facil_ee': 0.05,
    'tau_facil_ei': 1.2,
    'tau_facil_ie': 0.02,
    'tau_facil_ii': 0.06,
    
    'W_ee': .3 * params['W_Scale'],
    'W_ei': .6 * params['W_Scale'],
    'W_ie': .19 * params['W_Scale'],
    'W_ii': .19 * params['W_Scale'],
    
    'C_ee': .3,
    'C_ei': .2,
    'C_ie': .4,
    'C_ii': .1,
}



print("Weights EE:{:.2f}, EI:{:.2f}, IE:{:.2f}, II:{:.2f}".format(proj_params['W_ee'], proj_params['W_ei'], proj_params['W_ie'], proj_params['W_ii']))

projs = create_projections(network, params, proj_params)

######################################## MONITORS
m_liquid = AN.Monitor(network['liquid_pop'], 'spike')
m_readout = AN.Monitor(network['readout_pop'], 'spike')

AN.compile(compiler_flags="-march=native -O0")

######################################## SIMULATION
network['input_pop'].spike_times = [[0]]
AN.simulate(1000, measure_time=False)

readout_spikes = m_readout.get('spike')
liquid_spikes = m_liquid.get('spike')

print('Readout Spikes')

print({kk: len(vv) for kk, vv in readout_spikes.items() if len(vv)!=0})
print('Liquid Spikes')
#print({kk: len(vv) for kk, vv in liquid_spikes.items() if len(vv)!=0})

print()