import numpy as np
import ANNarchy as AN
import random

def create_populations(params):
    np.random.seed(params['seed'])
    random.seed(params['seed'])
    
    # Input pop
    input_spiketimings = [[0] for i in range(params['inp_size'])]
    input_pop = AN.SpikeSourceArray(input_spiketimings)
    
    # Readout 
    readout_pop = AN.Population(geometry=5, neuron=AN.IF_curr_exp, name="READOUT")    
    
    # Liquid pop
    
    liquid_pop = AN.Population(geometry=params['liquid_geometry'], neuron=AN.IF_curr_exp, name="LIQUID")
    
    total_liquid_exc = int(liquid_pop.size * params['exc_inh_ratio'])
    total_liquid_inh = liquid_pop.size - total_liquid_exc
    
    
    # Assign EXC and INH neurons
    indices = list(liquid_pop.ranks)
    np.random.shuffle(indices)
    liquid_exc = liquid_pop[indices[:total_liquid_exc]]
    liquid_inh = liquid_pop[indices[:total_liquid_inh]]
    
    
    #Assign INP neurons
    indices = list(liquid_exc.ranks)
    np.random.shuffle(indices)
    liquid_inp = liquid_pop[indices[:params['inp_size']]]
    liquid_noninp = liquid_pop[indices[params['inp_size']:]]    
    
    
    return {'input_pop': input_pop, 'liquid_inp': liquid_inp, 'liquid_noninp': liquid_noninp,
            'liquid_pop':liquid_pop, 'liquid_exc': liquid_exc, 'liquid_inh': liquid_inh,
            'readout_pop': readout_pop
            }