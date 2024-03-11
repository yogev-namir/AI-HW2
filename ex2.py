from copy import deepcopy
import itertools
import json
import time
import utils
################################
# TODO: two pirate ships can be in the same location?
# TODO: two marine ships can be in the same location?
# TODO: what happens when a pirate ship collects a treasure and a marine ship encounters it?
################################
EMPTY_SHIP = 2

DEPOSIT_REWARD = 4
RESET_REWARD = -2
CONFISCATE_REWARD = -1 
TERMINATE_REWARD = -10

ids = ["318880754", "324079763"]

def find_base(map):
    for row_idx, row in enumerate(map):
        for col_idx, cell in enumerate(row):
            if cell == 'B':
                return row_idx, col_idx
            
def classify_locations_in_map(map):
    i_locations, non_i_locations = [], []
    for row_idx, row in enumerate(map):
        for col_idx, cell in enumerate(row):
            if cell == 'I':
                i_locations.append((row_idx, col_idx))
            else:
                non_i_locations.append((row_idx, col_idx))
    return non_i_locations, i_locations

def create_all_permutations(pirate_ships_states, treasures_states, marine_ships_states, static_state):
    # Convert the state dictionaries to lists of tuples (key, value) to use in itertools.product
    pirate_ships_list = [(k, v) for k, values in pirate_ships_states.items() for v in values]
    treasures_list = [(k, v) for k, values in treasures_states.items() for v in values]
    marine_ships_list = [(k, v) for k, values in marine_ships_states.items() for v in values]
    
    # Generate all permutations
    all_permutations = itertools.product(pirate_ships_list, treasures_list, marine_ships_list)
    
    all_states = []
    for permutation in all_permutations:
        state = {"pirate_ships": {}, "treasures": {}, "marine_ships": {}}
        for item in permutation:
            if item[0].startswith("pirate_ship"):
                state["pirate_ships"][item[0]] = item[1]
            elif item[0].startswith("treasure"):
                state["treasures"][item[0]] = item[1]
            elif item[0].startswith("marine"):
                state["marine_ships"][item[0]] = item[1]
                
        # Merge with static state
        complete_state = {**static_state, **state}
        all_states.append(complete_state)
    
    return all_states

def assemble_states(initial):
    initial_state = initial
    pirate_ships = initial['pirate_ships']
    treasures = initial['treasures']
    marine_ships = initial['marine_ships']
    map = initial['map']
    non_i_locations, i_locations = classify_locations_in_map(map)
    
    pirate_ships_states = {}
    treasures_states = {}
    marine_ships_states = {}

    for pirate_ship,pirate_ship_details in pirate_ships.items():
        pirate_ships_states[pirate_ship] = []
        for location in non_i_locations:
            # min_capacity = max(0,pirate_ship_details['capacity'] - len(initial_state['treasures']))
            # max_capacity = pirate_ship_details['capacity'] + 1
            for capacity in range(0, EMPTY_SHIP+1):
                pirate_ships_states[pirate_ship].append({"location": location,
                                                         "capacity": capacity})
                
    for treasure, treasure_details in treasures.items():
        treasures_states[treasure] = []
        possible_locations = treasure_details['possible_locations']
        for location in possible_locations:
            treasures_states[treasure].append({"location": location,
                                                "possible_locations": treasure_details['possible_locations'],
                                                "prob_change_location": treasure_details['prob_change_location']})
    
    for marine_ship, marine_ship_details in marine_ships.items():
        marine_ships_states[marine_ship] = []
        for index in range(len(marine_ship_details['path'])):
            marine_ships_states[marine_ship].append({"index": index,
                                                    "path": marine_ship_details['path']})
    
 
    static_state = {"optimal": initial_state['optimal'],
                    "infinite": initial_state['infinite'],
                    "map": map}    
    
    if initial_state['infinite']:
        static_state["gamma"] = initial_state['gamma']

    all_possible_states_dict = create_all_permutations(pirate_ships_states, treasures_states, marine_ships_states, static_state)
    all_possible_states_json = [json.dumps(state, sort_keys=True) for state in all_possible_states_dict]
    return all_possible_states_json, all_possible_states_dict


def check_sail(current_location, pirate_ship, map):
    actions = []
    x, y = current_location
    directions = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

    for (dx, dy) in directions.values():
        new_x, new_y = x + dx, y + dy
        # Check if new position is within map bounds and not 'I'
        if 0 <= new_x < len(map) and 0 <= new_y < len(map[0]) and map[new_x][new_y] != 'I':
            actions.append(("sail", pirate_ship, (new_x, new_y)))
    return actions

def check_collect_treasure(current_location, treasures, pirate_ship, capacity):
    actions = []
    if capacity == 0: return actions # Ship is full, no place for more treasures
    for treasure, treasure_details in treasures.items():
        if utils.distance(treasure_details['location'], current_location) == 1:
            actions.append(("collect_treasure", pirate_ship, treasure))
    return actions                       

def check_deposit_treasure(current_location, pirate_ship, capacity, base):
    actions = []
    if capacity < EMPTY_SHIP and utils.distance(base, current_location) == 0:
        actions.append(("deposit_treasure", pirate_ship))
    return actions


def get_next_actions(current_state, base):
    """
    returns:
    - all_actions_combinations: a dictionary mapping each state to possible actions.
    """
    map = current_state['map']
    treasures = current_state['treasures']
    actions = {}
    for pirate_ship, pirate_ship_details in current_state['pirate_ships'].items():
        current_location = pirate_ship_details['location']
        capacity = pirate_ship_details['capacity']
        sail_actions = check_sail(current_location, pirate_ship, map)
        collect_treasure_actions = check_collect_treasure(current_location,
                                                           treasures, pirate_ship, capacity)
        deposit_treasure_actions = check_deposit_treasure(current_location, pirate_ship, capacity, base)
        actions[pirate_ship] = [('wait', pirate_ship)] + sail_actions + \
                                collect_treasure_actions + deposit_treasure_actions
    all_actions_combinations = list(itertools.product(*actions.values()))
    all_actions_combinations = all_actions_combinations + ['reset','terminate']
    return all_actions_combinations

'''
def deterministic_result_state(current_state, action_tuple):
    result_state = deepcopy(current_state)
    marine_ships_locations = get_marine_ship_locations_from_state(current_state)  
    reward = 0

    if action_tuple == 'reset': 
        reward = RESET_REWARD  

    # change pirate ships states
    for pirate_ship_action in action_tuple:
        action_type, pirate_ship = pirate_ship_action[0], pirate_ship_action[1]
        # No need to change anything if action is 'wait'
        if action_type == 'sail':
            result_state['pirate_ships'][pirate_ship]['location'] = pirate_ship_action[2]

        elif action_type == 'collect_treasure':
            result_state['pirate_ships'][pirate_ship]['capacity'] -= 1

        else: # action_type == 'deposit_treasure'
            result_state['pirate_ships'][pirate_ship]['capacity'] = EMPTY_SHIP

    # result_state = {"pirate_ships": result_state['pirate_ships']}
    return result_state, reward
'''
def deterministic_result_and_reward(current_state, action_tuple):
    result_state = deepcopy(current_state)
    marine_ships_locations = get_marine_ship_locations_from_state(current_state)  
    reward = 0

    if action_tuple == 'reset': 
        reward = RESET_REWARD  

    # change pirate ships states
    for action in action_tuple:
        pirate_ship = action
        if action[0] == 'sail' or action[0] == 'wait':
                location = action[2] if action[0] == 'sail' else current_state['pirate_ships'][pirate_ship]['location']
                confiscate = int(location in marine_ships_locations)
                if confiscate:
                    current_state['pirate_ships'][pirate_ship]['capacity'] = EMPTY_SHIP
                reward += confiscate * CONFISCATE_REWARD 

        elif action[0] == 'collect_treasure':
            result_state['pirate_ships'][pirate_ship]['capacity'] -= 1

        elif action[0] == 'deposit_treasure':
            result_state['pirate_ships'][pirate_ship]['capacity'] = EMPTY_SHIP
            reward += (EMPTY_SHIP - current_state['pirate_ships'][pirate_ship]['capacity']) * DEPOSIT_REWARD 

    # result_state = {"pirate_ships": result_state['pirate_ships']}
    return result_state, reward


def stochastic_combinations(treasures_states, marine_ships_states):
    combined_states = []

    # Generate all combinations of treasure and marine ship states
    for ((treasure, t_states), (marine_ship, m_states)) in itertools.product(treasures_states.items(), marine_ships_states.items()):
        for (t_state, t_prob), (m_state, m_prob) in itertools.product(t_states, m_states):
            combined_state = {
                "treasures": {treasure: t_state},
                "marine_ships": {marine_ship: m_state}
            }
            combined_prob = t_prob * m_prob
            combined_states.append((combined_state, combined_prob))

    return combined_states


def stochastic_result_states(state):
    marine_ships = state['marine_ships']
    treasures = state['treasures']
    
    treasures_states = {}
    marine_ships_states = {}
    
    for treasure, treasure_details in treasures.items():
        treasures_states[treasure] = []
        current_location = treasure_details['location']
        locations = treasure_details['possible_locations']
        prob = treasure_details['prob_change_location']
        uniform_prob = prob/len(locations)
        for location in locations:
            probability = uniform_prob
            if location == current_location:
                probability = 1 - prob + uniform_prob
            treasures_states[treasure].append(({"location": location,
                                                "possible_locations": locations,
                                                "prob_change_location": prob}, probability))
                
    for marine_ship, marine_ship_details in marine_ships.items():
        marine_ships_states[marine_ship] = []
        index = marine_ship_details['index']
        path = marine_ship_details['path']
        if len(path) <= 2:
            probability = 1 / len(path)
            for new_index in range(len(path)):
                marine_ships_states[marine_ship].append(({"index": new_index, "path": path}, probability))
        elif len(path) > 2: 
            if 0 < index < len(path):
                for dx in [-1,0,1]:
                    marine_ships_states[marine_ship].append(({"index": index+dx, "path": path}, 1/3))
            elif index == 0:
                for new_index in [0,1]:
                    marine_ships_states[marine_ship].append(({"index": new_index, "path": path}, 1/2))
            elif index == len(path)-1:
                for new_index in [len(path)-1, len(path)-2]:
                    marine_ships_states[marine_ship].append(({"index": new_index, "path": path}, 1/2))
                    
    all_stochastic_combinations = stochastic_combinations(treasures_states, marine_ships_states)
    return all_stochastic_combinations


def get_marine_ship_locations_from_state(state):
    marine_ships = state['marine_ships']  # Extract marine ships info from the state
    locations = []
    for ship_details in marine_ships.values():
        index = ship_details['index']
        path = ship_details['path']
        current_location = path[index]
        locations.append(current_location)
    return locations


def calculate_reward_and_apply_action(state, actions_tuple):
    reward = 0
    # state = deepcopy(state)
    marine_ships_locations = get_marine_ship_locations_from_state(state)  
    if actions_tuple == 'reset': 
        reward = RESET_REWARD  
    else:
        for action in actions_tuple:
            pirate_ship = action[1]
            if action[0] == 'deposit_treasure':  
                reward += (EMPTY_SHIP - state['pirate_ships'][pirate_ship]['capacity']) * DEPOSIT_REWARD
                state['pirate_ships'][pirate_ship]['capacity'] = EMPTY_SHIP  

            elif action[0] == 'sail' or action[0] == 'wait':
                location = action[2] if action[0] == 'sail' else state['pirate_ships'][pirate_ship]['location']
                confiscate = int(location in marine_ships_locations)
                if confiscate:
                    state['pirate_ships'][pirate_ship]['capacity'] = EMPTY_SHIP
                reward += confiscate * CONFISCATE_REWARD  
    pirate_ships_part = {'pirate_ships': state['pirate_ships']}
    return pirate_ships_part, reward


def get_next_stochastic_states(current_state_json, current_state, action, initial_state):
    """
    returns: 
    - next_stochastic_states: a list of (probability, next_state, reward) tuples for an input of (state, action) pair
    """
    next_stochastic_states = []
    deterministic_state_part = {"optimal": current_state['optimal'],
                                  "infinite": current_state['infinite'],
                                  "map": current_state['map'],
                                  }
    if current_state['infinite']:
        deterministic_state_part["gamma"] = initial_state['gamma']

    if action == 'reset':
        next_stochastic_states.append((current_state_json, 1))
        reward = RESET_REWARD
    elif action == 'terminate':
        next_stochastic_states.append((current_state_json, 1))
        reward = TERMINATE_REWARD
    
    else:
        # change pirate ships states
        pirate_ships_part, reward = calculate_reward_and_apply_action(current_state, action)
        deterministic_state_part.update(pirate_ships_part)

        # change marine ships and treasures states
        all_stochastic_states = stochastic_result_states(current_state)
        for (stochastic_state, probability) in all_stochastic_states:
            deterministic_state_part.update(stochastic_state)
            # reward = calculate_reward_and_alter_state(deterministic_state_part, action) # also handles encounters with Marine Ships
            next_stochastic_states.append((json.dumps(deterministic_state_part, sort_keys=True), probability))

    return (reward, next_stochastic_states)
        
    

def possible_next_states(states_json, states_dict, initial_state):
    next_actions_dict = {}
    next_states_dict = {}
    base = find_base(initial_state['map'])
    for state_json, state_dict in zip(states_json, states_dict):
        next_actions_dict[state_json] = get_next_actions(state_dict, base) # tuples of possible actions for each pirate ship 
        for action in next_actions_dict[state_json]:
            next_states_dict[state_json, action] = get_next_stochastic_states(state_json, state_dict, action,initial_state)

    return next_actions_dict, next_states_dict


def value_iterations(possible_states, next_actions_dict, next_states_dict, turns_to_go):
    """
    Perform value iteration to find the optimal policy.
    Returns:
    - policy: a dictionary mapping states to the optimal action to take from that state.
    - V: a dictionary of state values.
    """
    V = {(state,0): 0 for state in possible_states}  # Initialize value function
    policy = {(state,0): None for state in possible_states}  # Initialize policy
    
    for turn in range(1, turns_to_go+1):
        for state in possible_states:
            # Compute the value for all possible actions and choose the one with the max value
            action_values = []
            for action in next_actions_dict[state]:
                reward, next_states = next_states_dict[(state, action)]
                action_value = reward + sum(prob * V[(next_state,turn-1)] for (next_state, prob) in next_states)
                action_values.append((action_value, action))
            best_value, best_action = max(action_values)
            
            V[(state,turn)] = best_value  # Update the value function
            policy[(state,turn)] = best_action  # Update the policy
    
    return policy, V


class OptimalPirateAgent:
    def __init__(self, initial):
        self.initial = deepcopy(initial)
        self.turns_to_go = self.initial.pop('turns to go')
        self.possible_states_json, self.possible_states_dict = assemble_states(self.initial)
        self.next_actions_dict, self.next_states_dict = possible_next_states(self.possible_states_json,
                                                                             self.possible_states_dict, self.initial)
        
        self.policy, self.v_star = value_iterations(self.possible_states_json, self.next_actions_dict,
                                                    self.next_states_dict, self.turns_to_go)

    def act(self, state):
        turn = state.pop('turns to go')
        state_json = json.dumps(state, sort_keys=True)
        return self.policy[state_json, turn]


class PirateAgent:
    def __init__(self, initial):
        self.initial = initial
        


    def act(self, state):
        raise NotImplemented


class InfinitePirateAgent:
    def __init__(self, initial, gamma):
        self.initial = initial
        self.gamma = gamma

    def act(self, state):
        raise NotImplemented

    def value(self, state):
        raise NotImplemented
