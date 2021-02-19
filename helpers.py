import numpy as np
import ANNarchy as AN

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
