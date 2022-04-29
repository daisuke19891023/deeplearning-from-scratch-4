import common.gridworld_render as render_helper
import numpy as np

class GridWorld:
    def __init__(self) -> None:
        self.action_space = [0,1,2,3]
        self.action_meaning= {
            0: "UP",
            1:"DOWN",
            2:"LEFT",
            3:"RIGHT"
        }
        
        self.reward_map: np.ndarray = np.array([
            [0,0,0,1],
            [0,None, 0,-1],
            [0,0,0,0]
        ])
        self.goal_state = (0,3)
        self.wall_state = (1,1)
        self.start_state = (2,0)
        self.agent_state = self.start_state
    
    @property
    def height(self):
        return len(self.reward_map)
    @property
    def width(self):
        return len(self.reward_map[0])

    @property
    def shape(self):
        return self.reward_map.shape
    
    def actions(self):
        return self.action_space
    
    def states(self):
        for h in range(self.height):
            for w in range(self.width):
                yield (h, w)
                
    def next_state(self, state: tuple[float, float], action: int):
        action_move_map: list[tuple[float, float]] = \
            [(-1, 0), (1, 0), (0, -1), (0, 1)]
        move = action_move_map[action]
        next_state = (state[0] + move[0], state[1] + move[1])
        ny, nx = next_state
        
        if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
            next_state = state
        elif next_state == self.wall_state:
            next_state = state
        
        return next_state

    def reward(self, state: tuple[float, float], \
                action: int, \
                next_state: tuple[float, float]) -> np.ndarray:
        return self.reward_map[next_state]

    def render_v(self, v=None, policy=None, print_value=True):
        renderer = render_helper.Renderer(self.reward_map, self.goal_state,
                                          self.wall_state)
        renderer.render_v(v, policy, print_value)
        
    def reset(self):
        self.agent_state = self.start_state
        return self.agent_state

    def step(self, action):
        state = self.agent_state
        next_state = self.next_state(state, action)
        reward = self.reward(state, action, next_state)
        done = (next_state == self.goal_state)

        self.agent_state = next_state
        return next_state, reward, done

    def render_q(self, q=None, print_value=True):
        renderer = render_helper.Renderer(self.reward_map, self.goal_state,
                                          self.wall_state)
        renderer.render_q(q, print_value)