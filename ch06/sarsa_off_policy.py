if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from collections import defaultdict, deque
import numpy as np

from common.gridworld import GridWorld
from common.utils import greedy_probs


class SarsaOfPolicyAgent:
    def __init__(self) -> None:
        self.gamma = 0.9
        self.epsilon = 0.1
        self.alpha = 0.8
        self.action_size = 4

        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.pi = defaultdict(lambda: random_actions)
        self.b = defaultdict(lambda: random_actions)
        self.Q = defaultdict(lambda: 0)
        self.memory: deque[tuple] = deque(maxlen=2)

    def get_action(self, state: tuple[float, float]):
        action_probs = self.b[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)

    def reset(self):
        self.memory.clear()

    def update(self, state, action, reward, done):
        self.memory.append((state, action, reward, done))
        if len(self.memory) < 2:
            return

        state, action, reward, done = self.memory[0]
        next_state, next_action, _, _ = self.memory[1]

        if done:
            next_q = 0
            rho = 1
        else:
            next_q = self.Q[next_state, next_action]
            # 重み
            rho = self.pi[next_state][next_action] / self.b[next_state][next_action]

        target = rho * (reward + self.gamma * next_q)
        self.Q[state, action] += (target - self.Q[state, action]) * self.alpha

        self.pi[state] = greedy_probs(self.Q, state, 0)
        self.b[state] = greedy_probs(self.Q, state, self.epsilon)


if __name__ == "__main__":
    env = GridWorld()
    agent = SarsaOfPolicyAgent()
    episodes = 10000
    for episode in range(episodes):
        state = env.reset()
        agent.reset()

        while True:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, done)
            if done:
                agent.update(next_state, None, None, None)
                break

            state = next_state
    env.render_q(agent.Q)