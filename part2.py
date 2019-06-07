from pomegranate import *
import pygraphviz
import matplotlib.pyplot as plt
import numpy as np
import pdb

# documentation:
# https://pomegranate.readthedocs.io/en/latest/BayesianNetwork.html

# root distributions
p_phase = .5
phase_dist = DiscreteDistribution({1 : p_phase, 0 : 1-p_phase})

# conditional distributions
# each tuple is: given1, given2, result, probability

seed_dist = ConditionalProbabilityTable([
    [0, 1, .01],
    [0, 0, 1-.01],
    [1, 1, .5],
    [1, 0, 1-.5]], [phase_dist])

shg_dist = ConditionalProbabilityTable([
    [0, 1, .1],
    [0, 0, 1-.1],
    [1, 1, .8],
    [1, 0, 1-.8]], [phase_dist])

flexible_dist = ConditionalProbabilityTable([
    [0, 0, 1, 0],
    [0, 0, 0, 1-0],
    [0, 1, 1, .2],
    [0, 1, 0, 1-.2],
    [1, 0, 1, .85],
    [1, 0, 0, 1-.85],
    [1, 1, 1, .5],
    [1, 1, 0, 1-.5]],
    [seed_dist, shg_dist])

violence_dist = ConditionalProbabilityTable([
    [0, 0, 1, .6],
    [0, 0, 0, 1-.6],
    [0, 1, 1, .4],
    [0, 1, 0, 1-.4],
    [1, 0, 1, .85],
    [1, 0, 0, 1-.85],
    [1, 1, 1, .5],
    [1, 1, 0, 1-.5]],
    [shg_dist, flexible_dist])

land_dist = ConditionalProbabilityTable([
    [0, 0, 1, .4],
    [0, 0, 0, 1-.4],
    [0, 1, 1, .45],
    [0, 1, 0, 1-.45],
    [1, 0, 1, .46],
    [1, 0, 0, 1-.46],
    [1, 1, 1, .45],
    [1, 1, 0, 1-.45]],
    [seed_dist, flexible_dist])

knowledge_dist = ConditionalProbabilityTable([
    [0, 1, .1],
    [0, 0, 1-.1],
    [1, 1, .6],
    [1, 0, 1-.6]], [shg_dist])

# define nodes from distribution
seed_node = Node(seed_dist, name = 'seed')
shg_node = Node(shg_dist, name = 'shg')
phase_node = Node(phase_dist, name = 'phase')
flexible_node = Node(flexible_dist, name = 'flexible')

land_node = Node(land_dist, name = 'land')
knowledge_node = Node(knowledge_dist, name = 'knowledge')
violence_node = Node(violence_dist, name = 'violence')

# define network
model = BayesianNetwork('Problem 3')
model.add_states(seed_node, # causes
                 shg_node,
                 phase_node,
                 flexible_node,
                 land_node, # effect nodes
                 knowledge_node,
                 violence_node)

model.add_edge(phase_node, seed_node)
model.add_edge(phase_node, shg_node)

model.add_edge(seed_node, land_node)
model.add_edge(seed_node, flexible_node)

model.add_edge(shg_node, flexible_node)
model.add_edge(shg_node, violence_node)
model.add_edge(shg_node, knowledge_node)

model.add_edge(flexible_node, land_node)
model.add_edge(flexible_node, violence_node)
model.bake()

# joint distribution
# print out probability of each given outcome
# set none if doesn't matter
seed_val = None
shg_val = 1
phase_val = 1
flexible_val = 1
land_val = 1
knowledge_val = 1
violence_val = 1
prob = model.probability([[seed_val,
                    shg_val,
                    phase_val,
                    flexible_val,
                    land_val,
                    knowledge_val,
                    violence_val]])
print(prob)
pdb.set_trace()

# show module
model.plot()
plt.show()
