if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from collections import defaultdict
from ch04.policy_eval import policy_eval
from ch04.policy_iter import greedy_policy
from common.gridworld import GridWorld


def value_iter_onestep(V: defaultdict, env: GridWorld, gamma: float):
    for state in env.states():
        if state == env.goal_state:
            V[state] = 0
            continue

        action_values = []
        for action in env.actions():
            next_state = env.next_state(state, action)
            r = env.reward(state, action, next_state)
            value = r + gamma * V[next_state]
            action_values.append(value)
        V[state] = max(action_values)
    return V


def value_iter(
    V: defaultdict, env: GridWorld, gamma: float, threshold=0.001, is_render=False
):
    while True:
        if is_render:
            env.render_v(V)

        old_V = V.copy()
        V = value_iter_onestep(V, env, gamma)

        delta = 0
        for state in V.keys():
            t = abs(V[state] - old_V[state])
            if delta < t:
                delta = t

        if delta < threshold:
            break
    return V


if __name__ == "__main__":
    V = defaultdict(lambda: 0)
    env = GridWorld()
    gamma = 0.9

    V = value_iter(V, env, gamma)
    pi = greedy_policy(V, env, gamma)
    env.render_v(V, pi)
