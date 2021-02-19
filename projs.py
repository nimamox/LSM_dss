import ANNarchy as AN
from helpers import liquid_connector

def create_projections(network, params, proj_params):
    projs = {}
    # Liquid internal connections
    for syn_type in ['ee', 'ei', 'ie', 'ii']:
        if syn_type[0] == 'e':
            pre = network['liquid_exc']
            target = 'exc'
        else:
            pre = network['liquid_inh']
            target = 'inh'
        if syn_type[1] == 'e':
            post = network['liquid_exc']
        else:
            post = network['liquid_inh']
            
        W = proj_params['W_'+syn_type]
        U = proj_params['U_'+syn_type]
        C = proj_params['C_'+syn_type]
        tau_rec = proj_params['tau_rec_'+syn_type]
        tau_facil = proj_params['tau_facil_'+syn_type]
            
        projs[syn_type] = AN.Projection(pre, post, target, synapse=AN.models.STP).connect_with_func(
            method=liquid_connector, 
            weights=AN.Normal(W, (W/2.0), min=0.2*W, max=2.0*W), 
            delay=proj_params['delay'], 
            C=C, 
            lambda_=params['lambda_']
        )
        projs[syn_type].U = AN.Normal(U, U/2.0, min=0.1, max=0.9)
        projs[syn_type].tau_rec = AN.Normal(tau_rec, tau_rec/2.0)
        projs[syn_type].tau_facil = AN.Normal(tau_facil, tau_facil/2.0)
    
    # Input to liquid connections
    projs['inp_liq'] = AN.Projection(network['input_pop'], network['liquid_inp'], 'exc'
                                ).connect_one_to_one(weights=params['W_inp_liq'], delays=1)
    
    # Liquid to readout connections
    projs['liq_readout'] = AN.Projection(network['liquid_pop'], network['readout_pop'], 'exc'
                                ).connect_all_to_all(weights=AN.Uniform(0.1, 2.0), delays=1)
    
    
        