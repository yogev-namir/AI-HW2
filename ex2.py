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


def permutate_dicts(**kwargs):
    """
    Merge and cross product dicts - Build permutations.
    """
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


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

    for pirate_ship, pirate_ship_details in pirate_ships.items():
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
        length = len(marine_ship_details['path'])
        for index in range(length):
            marine_ships_states[marine_ship].append({"index": index,
                                                     "path": marine_ship_details['path']})
    
    pirate_ships_states = list(permutate_dicts(**pirate_ships_states))
    treasures_states = list(permutate_dicts(**treasures_states))
    marine_ships_states = list(permutate_dicts(**marine_ships_states))

    
    all_states_raw = {"optimal": [initial_state['optimal']],
                    "infinite": [initial_state['infinite']],
                    "map": [map],
                    "pirate_ships": pirate_ships_states,
                    "treasures": treasures_states,
                    "marine_ships": marine_ships_states}    
    
    if initial_state['infinite']:
        all_states_raw["gamma"] = [initial_state['gamma']]

    all_possible_states_dict = list(permutate_dicts(**all_states_raw))
    all_possible_states_json = [json.dumps(state) for state in all_possible_states_dict]
    return all_possible_states_json, all_possible_states_dict


def check_sail(current_location, pirate_ship, map):
    actions = []
    x, y = current_location
    num_rows, num_cols = len(map), len(map[0])
    directions = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
    
    for (dx, dy) in directions.values():
        new_x, new_y = x + dx, y + dy
        # Check if new position is within map bounds and not 'I'
        if 0 <= new_x < num_rows and 0 <= new_y < num_cols and map[new_x][new_y] != 'I':
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

import itertools

def generate_permutations(state_entities):
    """
    Generate all permutations for a given set of state entities (treasures or marine ships).
    """
    permutations = []
    for entity, states in state_entities.items():
        entity_permutations = [({entity: state}, prob) for state, prob in states]
        permutations.append(entity_permutations)
    return list(itertools.product(*permutations))

def combine_entity_permutations(treasure_permutations, marine_ship_permutations):
    """
    Combine permutations of treasures and marine ships into single states.
    """
    combined_states = []
    for treasure_perm in treasure_permutations:
        for marine_perm in marine_ship_permutations:
            combined_state = {"treasures": {}, "marine_ships": {}}
            combined_prob = 1

            # Unpack treasure permutations
            for t in treasure_perm:
                combined_state["treasures"].update(t[0])
                combined_prob *= t[1]

            # Unpack marine ship permutations
            for m in marine_perm:
                combined_state["marine_ships"].update(m[0])
                combined_prob *= m[1]

            combined_states.append((combined_state, combined_prob))

    return combined_states

def stochastic_combinations(treasures_states, marine_ships_states):
    # Generate permutations for treasures and marine ships
    treasure_permutations = generate_permutations(treasures_states)
    marine_ship_permutations = generate_permutations(marine_ships_states)
    
    # Combine permutations between treasures and marine ships
    all_stochastic_combinations = combine_entity_permutations(treasure_permutations, marine_ship_permutations)
    return all_stochastic_combinations


def stochastic_result_states(state):
    marine_ships = state['marine_ships']
    treasures = state['treasures']
    
    treasures_states = {}
    marine_ships_states = {}
    
    for treasure, treasure_details in state['treasures'].items():
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
                
    for marine_ship, marine_ship_details in state['marine_ships'].items():
        marine_ships_states[marine_ship] = []
        index = marine_ship_details['index']
        path = marine_ship_details['path']
        length = len(path)
        if length <= 2:
            probability = 1 / length
            for new_index in range(length):
                marine_ships_states[marine_ship].append(({"index": new_index, "path": path}, probability))
        elif length > 2: 
            if index == 0:
                for new_index in [index, index+1]:
                    marine_ships_states[marine_ship].append(({"index": new_index, "path": path}, 1/2))
            elif index == length-1:
                for new_index in [index, index-1]:
                    marine_ships_states[marine_ship].append(({"index": new_index, "path": path}, 1/2))
            else:
                for dx in [-1,0,1]:
                    marine_ships_states[marine_ship].append(({"index": index+dx, "path": path}, 1/3))
                    
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
    stochastic_state = {"optimal": current_state['optimal'],
                                  "infinite": current_state['infinite'],
                                  "map": current_state['map'],
                                  }
    if current_state['infinite']:
        stochastic_state["gamma"] = initial_state['gamma']

    if action == 'reset':
        next_stochastic_states.append((current_state_json, 1))
        reward = RESET_REWARD
    elif action == 'terminate':
        next_stochastic_states.append((current_state_json, 1))
        reward = TERMINATE_REWARD
    
    else:
        # change pirate ships states
        pirate_ships_part, reward = calculate_reward_and_apply_action(current_state, action)
        stochastic_state.update(pirate_ships_part)

        # change marine ships and treasures states
        all_stochastic_states = stochastic_result_states(current_state)
        for (treasure_marine_permutaion, probability) in all_stochastic_states:
            stochastic_state.update(treasure_marine_permutaion)
            # reward = calculate_reward_and_alter_state(deterministic_state_part, action) # also handles encounters with Marine Ships
            next_stochastic_states.append((json.dumps(stochastic_state), probability))

    return (reward, next_stochastic_states)
        
    
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
        state_json = json.dumps(state)
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
