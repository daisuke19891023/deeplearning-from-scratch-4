import matplotlib.pyplot as plt
import numpy as np
from bandit import Bandit, Agent

runs = 200
steps = 1000
epsilon = 0.1
all_retes = np.zeros((runs, steps))

for run in range(runs):
    bandit = Bandit()
    agent = Agent(epsilon)
    total_reward = 0
    rates = []

    for step in range(steps):
        action = agent.get_action()
        reward = bandit.play(action)
        agent.update(action, reward)
        total_reward += reward
        
        rates.append(total_reward / (step + 1))
    all_retes[run] = rates

avg_rates = np.average(all_retes, axis=0)

# drow graph
plt.ylabel("Rates")
plt.xlabel("Steps")
plt.plot(avg_rates)
plt.show()