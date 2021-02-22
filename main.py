import numpy as np
import ANNarchy as AN
AN.clear()
from pops import create_populations
from projs import create_projections
import random
import pickle
import sys



def gen_filename(params):
    out = ('lambda[{lambda_}]_Wscale[{W_Scale}]_D[{delay}]_liqinpS[{liq_inp_size}]' +
           '_Winp[{W_inp}]_Dinp[{delay_inp}]_EIr[{exc_inh_ratio}]_Geom[{liquid_geometry}]' + 
         '_inpS[{inp_size}]_readoutS[{readout_size}]').format(**params)
    out = out.replace('(','').replace(')','').replace(' ', '')
    return out


####################################### PARAMETERS

external_input = False


with open('firing_times.isi', 'rb') as fi:
    firing_times = pickle.load(fi)['firing_times']    
    external_input = True

if len(sys.argv) > 1:
    print('Reading params from file', end='')
    params_file_path = sys.argv[1]
    print(params_file_path)
    with open(params_file_path, 'rb') as fi:
        params = pickle.load(fi)
    
    if len(sys.argv) > 2:
        with open('firing_times.isi', 'rb') as fi:
            firing_times = pickle.load(fi)['firing_times']    
            external_input = True
else:
    params = {
        'delay': (1, 20),
        'W_inp': (6, 8),
        'delay_inp': -20,
        
        'W_out': (.5, 2.5),
    
        'liquid_geometry': (5, 5, 8),
        'exc_inh_ratio': .8,
        'inp_size': 1,
        'liq_inp_size': 10,
        'readout_size': 5,
        'lambda_': 3.5,
        'W_Scale': 7,
    
        'dt': .1,
        'seed': 1337
    }

output_filepath = 'results/' + gen_filename(params) + '.out'


np.random.seed(params['seed'])
random.seed(params['seed'])
AN.setup(dt=params['dt'], seed=params['seed'])


####################################### LIQUID POPULATION

network = create_populations(params)

#print("#Nodes in RC: {},  Exc-Inh ratio: {},  #Exc: {},  #Inp: {}".format(network['liquid_pop'].size, 
                                        #params['exc_inh_ratio'], network['liquid_exc'].size, params['inp_size']))


######################################## NETWORK PARAMETERS

proj_params = {
    'W_ee': .3 * params['W_Scale'], 'W_ei': .6 * params['W_Scale'], 
    'W_ie': .19 * params['W_Scale'], 'W_ii': .19 * params['W_Scale'],

    'U_ee': 0.5, 'U_ei': 0.05, 'U_ie': 0.25, 'U_ii': 0.32,
    'tau_rec_ee': 1.1, 'tau_rec_ei': 0.125, 'tau_rec_ie': 0.7, 'tau_rec_ii': 0.144,
    'tau_facil_ee': 0.05, 'tau_facil_ei': 1.2, 'tau_facil_ie': 0.02, 'tau_facil_ii': 0.06,
    'C_ee': .3, 'C_ei': .2, 'C_ie': .4, 'C_ii': .1,
}

#print("Weights EE:{:.2f}, EI:{:.2f}, IE:{:.2f}, II:{:.2f}".format(proj_params['W_ee'], proj_params['W_ei'], proj_params['W_ie'], proj_params['W_ii']))

projs = create_projections(network, params, proj_params)

######################################## MONITORS
m_liquid = AN.Monitor(network['liquid_pop'], 'spike')
m_readout = AN.Monitor(network['readout_pop'], 'spike')

AN.compile(compiler_flags="-march=native -O0")

######################################## SIMULATION
if not external_input:
    current_time = int(AN.get_current_step() * params['dt'])
    network['input_pop'].spike_times = [current_time + 10]

liq_spikes = []
print('----'*5)
for i in range(20):
    AN.simulate(50, measure_time=False)
    readout_spikes = m_readout.get('spike')
    liquid_spikes = m_liquid.get('spike')
    total_liquid_spikes = sum([len(_) for _ in list(liquid_spikes.values())])
    liq_spikes.append(total_liquid_spikes)
    if total_liquid_spikes:
        print(i+1, total_liquid_spikes)
print('----'*5)



with open(output_filepath, 'wb') as fo:
    pickle.dump(liq_spikes, fo)

##print({kk: len(vv) for kk, vv in liquid_spikes.items() if len(vv)!=0})

#print()