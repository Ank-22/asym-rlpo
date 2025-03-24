import logging
import enum
import math

from asym_rlpo.policies import Policy

logger = logging.getLogger(__name__)


class SubGoals(enum.Enum):
    FIND_ITEM_LOCATION = enum.auto()
    GO_TO_ITEM_LOCATION = enum.auto()
    BUY_THE_ITEM = enum.auto()


class Actions(enum.Enum):
    QUERY = enum.auto()
    LEFT = enum.auto()
    RIGHT = enum.auto()
    UP = enum.auto()
    DOWN = enum.auto()
    BUY = enum.auto()

    


class Shopping_HardcodedPolicy(Policy):
    def __init__(self, size: int):
        super().__init__()
        self.size = size
        self.subgoal = SubGoals.FIND_ITEM_LOCATION
        self.item_location = None
        self.current_position = None

    def reset(self, observation):
        """ Reset the policy for a new episode """
        self.subgoal = SubGoals.FIND_ITEM_LOCATION
        self.item_location = None
        self.current_position = None

    def step(self, action, observation):
        """ Choose the next action based on the current subgoal """
        observation = observation.item()
        test = self.get_coordinates(observation)
        breakpoint()
        if self.subgoal == SubGoals.FIND_ITEM_LOCATION:
            #TO-DO: Get the iteam location or obersavation and get the co-ordinates Update the representaion 
            breakpoint()
            self.item_location = observation.get("item_location")  # Assuming environment provides this after querying
            if self.item_location:
                self.subgoal = SubGoals.GO_TO_ITEM_LOCATION
            return Actions.QUERY  # Query for the item location

        elif self.subgoal == SubGoals.GO_TO_ITEM_LOCATION:
            if self.current_position == self.item_location:
                self.subgoal = SubGoals.BUY_THE_ITEM
                return Actions.BUY  # Buy the item once at the correct location
            
            # Move towards the item
            return self._move_towards(self.item_location)

        elif self.subgoal == SubGoals.BUY_THE_ITEM:
            return Actions.BUY
    
    def sample_action(self):
        """ Returns a random action (not needed for this hardcoded policy) """
        #TODO: Actions will take place here
        return 0  # Default action for exploration    

    
    def _move_towards(self, target):
        """ Move one step towards the target location """
        x, y = self.current_position
        tx, ty = target

        if x < tx:
            x  += 1
            self.current_position = (x,y)
            return Actions.RIGHT
        elif x > tx:
            x  -= 1
            self.current_position = (x,y)
            return Actions.LEFT
        elif y < ty:
            y += 1
            self.current_position = (x,y)
            return Actions.UP
        elif y > ty:
            y  -= 1
            self.current_position = (x,y)
            return Actions.DOWN
        elif x == tx and y == ty:
            return Actions.BUY

   
    def get_coordinates(self, location):
        """Returns X & Y Co ordinates from the observation"""
        gird_limit =(self.size**2)
        if(location >= gird_limit):
            grid_loc = location - gird_limit 
        y_location = math.floor(grid_loc / self.size)
        x_location = grid_loc % self.size
        return (x_location, y_location)
    
    _action_to_int = {
    Actions.QUERY: 0,
    Actions.LEFT: 1,
    Actions.RIGHT: 2,
    Actions.UP: 3,
    Actions.DOWN: 4,
    Actions.BUY: 5,
}