import logging
import enum
from asym_rlpo.policies import Policy
import numpy as np
from gym_gridverse.action import Action
from gym_gridverse.grid_object import Hidden, Beacon, Floor, Wall, Exit
logger = logging.getLogger(__name__)

class SubGoals(enum.Enum):
    FIND_BECON_COLOR = enum.auto()
    FIND_COLORED_GATE = enum.auto()
    GO_TO_COLORED_EXIT = enum.auto()

    

class Memory_Four_Room_HardcodedPolicy(Policy):
    def __init__(self, outer_env):
        super().__init__()
        self.outer_env = outer_env
        self.inner_env = self.outer_env.inner_env
        self.width = self.inner_env.state_space.grid_shape.width
        self.height = self.inner_env.state_space.grid_shape.height
        self.action_space = self.outer_env.action_space
        self.becon_color = None
        self.env_model = np.full((self.width, self.height), np.nan)
        self.seen_gate: dict[str, tuple[int, int]] = {}
        self.action = 0
        self.goal = SubGoals.FIND_BECON_COLOR

    def reset(self, observation):
        self.outer_env.reset()
        self.action_space = self.outer_env.action_space.actions
        self.width = self.inner_env.state_space.grid_shape.width
        self.height = self.inner_env.state_space.grid_shape.height
        self.becon_color = None
        self.env_model = np.full((self.width, self.height), np.nan)
        self.seen_gate: dict[str, tuple[int, int]] = {}
        self.action = 0
        self.goal = SubGoals.FIND_BECON_COLOR

    
    def step(self, action, observation):
        # Get observed grid and agent's full environment position
        obs_grid = self.inner_env.observation.grid
        obs_height, obs_width = obs_grid.shape.height, obs_grid.shape.width
        agent_pos = self.inner_env.observation.agent.position
        agent_orientation = self.inner_env.observation.agent.orientation

        # Since the agent is at the bottom middle of the observation:
        start_y = agent_pos.y - (obs_height - 1)
        start_x = agent_pos.x - (obs_width // 2)

        # Map observed area to the full environment model
        for obs_y in range(obs_height):
            for obs_x in range(obs_width):
                global_y, global_x = start_y + obs_y, start_x + obs_x
                
                # Ensure within bounds of the full environment
                if 0 <= global_y < self.height and 0 <= global_x < self.width:
                    obj = obs_grid[obs_y, obs_x]
                    if obj is not None and not isinstance(obj, Hidden):
                        self.env_model[global_y, global_x] = hash(obj.__class__.__name__)

                        # Track beacon color
                        if isinstance(obj, Beacon):
                            self.becon_color = obj.color

                        # Track seen gates
                        if isinstance(obj, Exit) and obj.color is not None:
                            self.seen_gate[obj.color] = (global_y, global_x)

        # Determine next action based on the goal
        if self.goal == SubGoals.FIND_BECON_COLOR:
            if self.becon_color is not None:
                self.goal = (
                    SubGoals.GO_TO_COLORED_EXIT 
                    if self.becon_color in self.seen_gate 
                    else SubGoals.FIND_COLORED_GATE
                )
            else:
                self.action = self.explore_unexplored()

        elif self.goal == SubGoals.FIND_COLORED_GATE:
            if self.becon_color in self.seen_gate:
                self.goal = SubGoals.GO_TO_COLORED_EXIT
            else:
                self.action = self.explore_unexplored()

        elif self.goal == SubGoals.GO_TO_COLORED_EXIT:
            self.action = self.navigate_to_exit()

        else:
            self.action = np.random.choice(self.inner_env.action_space.actions)

        return self.action

    def explore_unexplored(self):
        """Move toward the nearest unexplored area while avoiding exits and gates."""
        unexplored = np.isnan(self.env_model)
        if np.any(unexplored):
            y, x = np.argwhere(unexplored)[0]  # Pick first unexplored cell
            return self.move_towards((y, x))
        return np.random.choice(self.inner_env.action_space.actions)

    def move_towards(self, target_pos):
        """Move in the direction of a target position."""
        agent_pos = self.inner_env.observation.agent.position

        if target_pos[0] < agent_pos.y:
            return Action.MOVE_BACKWARD
        elif target_pos[0] > agent_pos.y:
            return Action.MOVE_FORWARD
        elif target_pos[1] < agent_pos.x:
            return Action.MOVE_LEFT
        elif target_pos[1] > agent_pos.x:
            return Action.MOVE_RIGHT
        return np.random.choice(self.inner_env.action_space.actions)

    def navigate_to_exit(self):
        """Find a simple path to the correct colored exit."""
        if self.becon_color in self.seen_gate:
            return self.move_towards(self.seen_gate[self.becon_color])
        return np.random.choice(self.inner_env.action_space.actions)


    def sample_action(self):
        return self.action
