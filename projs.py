import ANNarchy as AN
import numpy as np

def liquid_connector(pre, post, weights, delay, C, lambda_):
    synapses = AN.CSR()
    for post_rank in post.ranks:
        post_coord = post.population.coordinates_from_rank(post_rank)
        ranks = []
        for pre_rank in pre.ranks:
            if pre_rank == post_rank:
                continue
            pre_coord = pre.population.coordinates_from_rank(pre_rank)
            dist = np.linalg.norm(np.array(post_coord) - np.array(pre_coord))
            prob = C * np.exp(-dist/lambda_)
            if np.random.rand() < prob:
                ranks.append(pre_rank)
        values = weights.get_list_values(len(ranks))
        delays = [1 for i in range(len(ranks)) ]
        synapses.add(post_rank, ranks, values, delays)
    return synapses

def create_projections(network, params, proj_params):
    projs = {}
    if not hasattr(params['delay'], "__getitem__"):
        print('Const delay in liquid')
        dliq = params['delay']
    else:
        print('Random delay in liquid')
        dliq = AN.Uniform(params['delay'][0], params['delay'][1])
        
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
            delay=dliq, 
            C=C, 
            lambda_=params['lambda_']
        )
        projs[syn_type].U = AN.Normal(U, U/2.0, min=0.1, max=0.9)
        projs[syn_type].tau_rec = AN.Normal(tau_rec, tau_rec/2.0)
        projs[syn_type].tau_facil = AN.Normal(tau_facil, tau_facil/2.0)
    
    # Input to liquid connections
    if not hasattr(params['W_inp'], "__getitem__"):
        print('Const weight from inp to liquid')
        wil = params['W_inp']
    else:
        print('Random weight from inp to liquid')
        wil = AN.Uniform(params['W_inp'][0], params['W_inp'][1])
    
    if not hasattr(params['delay_inp'], "__getitem__"):
        print('Const delay from inp to liquid')
        dil = params['delay_inp']
    else:
        print('Random delay from inp to liquid')
        dil = AN.Uniform(params['delay_inp'][0], params['delay_inp'][1])
    
    inp_liq_proj = AN.Projection(network['input_pop'], network['liquid_inp'], 'exc')
    if network['input_pop'].size == network['liquid_inp'].size:
        print('one2one inp to liquid')
        projs['inp_liq'] = inp_liq_proj.connect_one_to_one(weights=wil, delays=dil)
    else:
        print('all2all inp to liquid')
        projs['inp_liq'] = inp_liq_proj.connect_all_to_all(weights=wil, delays=dil)
    
    # Liquid to readout connections
    projs['liq_readout'] = AN.Projection(network['liquid_pop'], network['readout_pop'], 'exc'
                                ).connect_all_to_all(weights=AN.Uniform(0.1, 2.0), delays=1)
    
    return projs
    
    
        