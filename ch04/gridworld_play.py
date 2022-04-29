if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from common.gridworld import GridWorld
import numpy as np
from collections import defaultdict
env = GridWorld()
V = defaultdict(lambda: 0)
for state in env.states():
    V[state] = np.random.randn()
env.render_v(V)
