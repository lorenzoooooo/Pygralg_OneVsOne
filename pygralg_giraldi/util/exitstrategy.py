"""
This is a simple pythion strategy in order to make sure that the random sampler always returns 1 path.
This is specifically important if the random sampler 'gets stuck' in peripheral nodes.
"""

# some important modules
import networkx
from random import randrange
from random import sample

# a simple graph (...just for testing)
H = networkx.Graph()
H.add_edges_from([(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)])

# subgraph order
order = 3

# extract all simple paths
bucket = []
for source in range(len(H.nodes())):
    for target in range(len(H.nodes())):
        # bucket = bucket + [list(path) for path in map(nx.utils.pairwise, networkx.all_simple_paths(H,source,target,order))]
        bucket = bucket + [list(path) for path in networkx.all_simple_paths(H, source, target, order)]

# save all paths with length==order
shortbucket = [x for x in bucket if len(x) == order]

if len(shortbucket) == 0:
    # there are no paths with that length
    raise ValueError
else:
    # random shuffle in order to remove root node ID ordering
    shortbucket = sample(shortbucket, len(shortbucket))
    # pick one at random and return
    this = randrange(len(shortbucket))
    winner = shortbucket[this]
