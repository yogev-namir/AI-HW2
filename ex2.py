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
CAPACITY = 2
THRESHOLD = 1000000
DEPOSIT_REWARD = 4
WAIT_PENALTY = 0.0001
CONFISCATE_REWARD = -1
RESET_REWARD = -2
TERMINATE_REWARD = -10
LOST_TREASURES = -100

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
            for capacity in range(0, CAPACITY + 1):
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
    if capacity == 0:
        return actions  # Ship is full, no place for more treasures
    for treasure, treasure_details in treasures.items():
        if utils.distance(treasure_details['location'], current_location) == 1:
            actions.append(("collect", pirate_ship, treasure))
    return actions


def check_deposit_treasure(current_location, pirate_ship, capacity, base):
    actions = []
    if capacity < CAPACITY and utils.distance(base, current_location) == 0:
        actions.append(("deposit", pirate_ship))
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
            result_state['pirate_ships'][pirate_ship]['capacity'] = CAPACITY

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
                current_state['pirate_ships'][pirate_ship]['capacity'] = CAPACITY
            reward += confiscate * CONFISCATE_REWARD

        elif action[0] == 'collect':
            result_state['pirate_ships'][pirate_ship]['capacity'] -= 1

        elif action[0] == 'deposit':
            result_state['pirate_ships'][pirate_ship]['capacity'] = CAPACITY
            reward += (CAPACITY - current_state['pirate_ships'][pirate_ship]['capacity']) * DEPOSIT_REWARD

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
    treasures_states = {}
    marine_ships_states = {}

    for treasure, treasure_details in state['treasures'].items():
        treasures_states[treasure] = []
        current_location = treasure_details['location']
        locations = treasure_details['possible_locations']
        prob = treasure_details['prob_change_location']
        uniform_prob = prob / len(locations)
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
                for new_index in [index, index + 1]:
                    marine_ships_states[marine_ship].append(({"index": new_index, "path": path}, 1 / 2))
            elif index == length - 1:
                for new_index in [index, index - 1]:
                    marine_ships_states[marine_ship].append(({"index": new_index, "path": path}, 1 / 2))
            else:
                for dx in [-1, 0, 1]:
                    marine_ships_states[marine_ship].append(({"index": index + dx, "path": path}, 1 / 3))

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
    new_state = deepcopy(state)
    reward = 0
    # state = deepcopy(state)
    marine_ships_locations = get_marine_ship_locations_from_state(new_state)
    if actions_tuple == 'reset':
        reward = RESET_REWARD
    else:
        for action in actions_tuple:
            pirate_ship = action[1]
            pirate_ship_capacity = new_state['pirate_ships'][pirate_ship]['capacity']
            if action[0] == 'deposit':
                reward += (CAPACITY - pirate_ship_capacity) * DEPOSIT_REWARD
                new_state['pirate_ships'][pirate_ship]['capacity'] = CAPACITY

            elif action[0] == 'sail' or action[0] == 'wait':
                if action[0] == 'wait':
                    reward += WAIT_PENALTY
                current_location = new_state['pirate_ships'][pirate_ship]['location']
                new_location = action[2] if action[0] == 'sail' else current_location
                new_state['pirate_ships'][pirate_ship]['location'] = new_location
                confiscate = int(new_location in marine_ships_locations)
                if confiscate:
                    new_state['pirate_ships'][pirate_ship]['capacity'] = CAPACITY
                    reward += CONFISCATE_REWARD
                    if pirate_ship_capacity < CAPACITY:
                        reward += LOST_TREASURES

            elif action[0] == 'collect':
                new_state['pirate_ships'][pirate_ship]['capacity'] -= 1

    pirate_ships_part = {'pirate_ships': new_state['pirate_ships']}
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
        for (treasure_marine_permutation, probability) in all_stochastic_states:
            stochastic_state.update(treasure_marine_permutation)
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
    all_actions_combinations = all_actions_combinations + ['reset', 'terminate']
    return all_actions_combinations


def possible_next_states(states_json, states_dict, initial_state):
    next_actions_dict = {}
    next_states_dict = {}
    base = find_base(initial_state['map'])
    for state_json, state_dict in zip(states_json, states_dict):
        next_actions_dict[state_json] = get_next_actions(state_dict,
                                                         base)  # tuples of possible actions for each pirate ship
        for action in next_actions_dict[state_json]:
            next_states_dict[state_json, action] = get_next_stochastic_states(state_json, state_dict, action,
                                                                              initial_state)

    return next_actions_dict, next_states_dict


'''
def value_iterations(possible_states, next_actions_dict, next_states_dict, turns_to_go):
    """
    Perform value iteration to find the optimal policy.
    Returns:
    - policy: a dictionary mapping states to the optimal action to take from that state.
    - V: a dictionary of state values.
    """
    V = {(state, 0): 0 for state in possible_states}  # Initialize value function
    policy = {(state, 0): None for state in possible_states}  # Initialize policy

    for turn in range(1, turns_to_go + 1):
        for state in possible_states:
            # Compute the value for all possible actions and choose the one with the max value
            next_actions = next_actions_dict[state]
            action_values = []

            for action in next_actions:
                reward, next_states = next_states_dict[(state, action)]
                action_value = reward + sum(prob * V[(next_state, turn - 1)] for (next_state, prob) in next_states)
                action_values.append((action_value, action))
            best_value, best_action = max(action_values, key=lambda x: x[0])

            V[(state, turn)] = best_value  # Update the value function
            policy[(state, turn)] = best_action  # Update the policy
    for v,p in zip(V.values(), policy.values()):
        print(v)
        print(p)
    return policy, V
'''


def value_iterations(possible_states, next_actions_dict, next_states_dict, turns_to_go):
    """
    Perform value iteration to find the optimal policy.
    Returns:
    - policy: a dictionary mapping states to the optimal action to take from that state.
    - V: a dictionary of state values.
    """
    V = {(state, 0): 0 for state in possible_states}  # Initialize value function
    policy = {(state, 0): None for state in possible_states}  # Initialize policy

    for turn in range(1, turns_to_go + 1):
        for state in possible_states:
            dict_state = json.loads(state)
            # Compute the value for all possible actions and choose the one with the max value
            next_actions = next_actions_dict[state]
            action_values = []

            for action in next_actions:
                reward, next_states = next_states_dict[(state, action)]
                for (next_state, prob) in next_states:
                    reward += prob * V[(next_state, turn - 1)]
                action_value = reward
                action_values.append((action_value, action))
            best_value, best_action = max(action_values, key=lambda x: x[0])

            V[(state, turn)] = best_value  # Update the value function
            policy[(state, turn)] = best_action  # Update the policy

    return policy, V


class OptimalPirateAgent:
    def __init__(self, initial):
        calc_total_states(initial)
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
        best_action = self.policy[(state_json, turn)]
        return best_action


def calc_total_states(initial):
    total_states = 1
    non_i_locations, i_locations = classify_locations_in_map(initial['map'])

    # pirate ships states
    total_states *= len(initial['pirate_ships'].items()) * CAPACITY * len(non_i_locations)

    # treasures states
    for _, treasures_details in initial['treasures'].items():
        total_states *= len(treasures_details['possible_locations'])

    # treasures states
    for _, marine_ships_details in initial['marine_ships'].items():
        total_states *= len(marine_ships_details['path'])

    total_states *= initial['turns to go']
    print('Total states:', total_states)
    return total_states


def calculate_expected_distance(ship_location, treasure_details):
    """
    Calculate the expected distance from a pirate ship to a treasure,
    considering the treasure's possible locations and probability to change location.
    """
    current_location = treasure_details['location']
    possible_locations = treasure_details['possible_locations']
    prob_change = treasure_details['prob_change_location']
    location_probability = prob_change / len(possible_locations)
    # Calculate the distance to the current location
    distance_to_current = utils.distance(ship_location, current_location)
    expected_distance = distance_to_current * (1 - prob_change + location_probability)

    # Calculate expected distances to possible locations
    for loc in possible_locations:
        if loc != current_location:  # Exclude the current location, already considered
            distance = utils.distance(ship_location, loc)
            expected_distance += distance * location_probability

    return expected_distance


def merge_clusters(clusters):
    merged_treasures = {}

    for cluster in clusters:
        combined_locations = set()
        total_prob_change = 0

        for treasure, _ in cluster:
            combined_locations.update(treasure['possible_locations'])
            total_prob_change += treasure['prob_change_location']

        # Create the merged treasure, the first one is the representative
        cluster_id = cluster[0]
        merged_treasures[cluster_id[1]] = {
            "location": cluster_id[0]['location'],
            "possible_locations": tuple(combined_locations),
            "prob_change_location": total_prob_change / len(cluster)
        }

    return merged_treasures


def is_overlap_significant(treasure1, treasure2):
    locations1 = set(treasure1['possible_locations'])
    locations2 = set(treasure2['possible_locations'])
    intersection = locations1.intersection(locations2)
    union = locations1.union(locations2)
    return len(intersection) >= 0.5 * len(union)


def cluster_treasures(treasures):
    # Initialize clusters and a set to keep track of clustered treasure IDs
    clusters = []
    clustered = set()

    # Function to find or create a cluster for a new treasure
    def find_or_create_cluster(new_treasure, new_treasur_id):
        for cluster in clusters:
            for _, treasure_id in cluster:
                if is_overlap_significant(treasures[new_treasur_id], treasures[treasure_id]):
                    cluster.append((new_treasure, new_treasur_id))
                    return True
        return False

    for treasure, treasure_details in treasures.items():
        if treasure not in clustered:
            # Attempt to add the treasure to an existing cluster
            if not find_or_create_cluster(treasure_details, treasure):
                # If it doesn't fit in existing clusters, create a new one
                clusters.append([(treasure_details, treasure)])
            clustered.add(treasure)

    return clusters


def reducted_initial_state(initial):
    initial_copy = deepcopy(initial)
    unpicked_pirate_ships = None
    representative_pirate_ship = next(iter(initial_copy['pirate_ships'].keys()), None)
    best_treasure = next(iter(initial_copy['treasures'].keys()), None)

    if len(initial_copy['pirate_ships']) > 1:
        representative_pirate_ship = next(iter(initial_copy['pirate_ships'].keys()), None)
        unpicked_pirate_ships = {k: v for k, v in initial_copy['pirate_ships'].items() if
                                 k != representative_pirate_ship}
        initial_copy['pirate_ships'] = {
            representative_pirate_ship: initial_copy['pirate_ships'][representative_pirate_ship]}

    if len(initial_copy['treasures']) > 1:
        '''
        # option 1
        reducted_treasures = merge_clusters(cluster_treasures(initial['treasures']))
        initial['treasures'] = reducted_treasures
        '''
        # option 2
        pirate_ship_location = initial_copy['pirate_ships'][representative_pirate_ship]['location']
        best_treasure = find_best_treasure(pirate_ship_location=pirate_ship_location,
                                           treasures=initial_copy['treasures'])
        initial_copy['treasures'] = {best_treasure: initial_copy['treasures'][best_treasure]}

    return initial_copy, unpicked_pirate_ships, representative_pirate_ship, best_treasure


def find_best_treasure(pirate_ship_location, treasures):
    # Calculate inertia for each treasure: sum of squared distances between current location and possible locations
    inertia_values = {}

    for treasure, properties in treasures.items():
        current_location = properties["location"]
        possible_locations = properties["possible_locations"]
        squared_distances = [utils.distance(current_location, loc) ** 2 for loc in possible_locations]
        inertia = sum(squared_distances)
        inertia_values[treasure] = inertia

    # Calculate the stability score for each treasure: lower score indicates higher stability
    stability_scores = {}

    for treasure, properties in treasures.items():
        prob_change_location = properties["prob_change_location"]
        inertia = inertia_values[treasure]
        # Stability score based on probability of changing location and inertia
        stability_score = prob_change_location * inertia
        stability_scores[treasure] = stability_score

    # Calculate the centroid (mean location) of possible locations for each treasure
    centroid_locations = {}

    for treasure, properties in treasures.items():
        possible_locations = properties["possible_locations"]
        x_coords = [loc[0] for loc in possible_locations]
        y_coords = [loc[1] for loc in possible_locations]
        centroid_x = sum(x_coords) / len(x_coords)
        centroid_y = sum(y_coords) / len(y_coords)
        centroid_locations[treasure] = (centroid_x, centroid_y)

    # Calculate the distance from the pirate ship to each treasure's centroid location
    distances_from_ship_to_centroid = {}

    for treasure, centroid_location in centroid_locations.items():
        distance_from_ship_to_centroid = utils.distance(pirate_ship_location, centroid_location)
        distances_from_ship_to_centroid[treasure] = distance_from_ship_to_centroid

    # Assuming equal importance for both factors (alpha = beta = 1)
    alpha = 1
    beta = 1

    # Calculate total score for each treasure
    total_scores = {}

    for treasure in treasures.keys():
        total_score = alpha * stability_scores[treasure] + beta * distances_from_ship_to_centroid[treasure]
        total_scores[treasure] = total_score

    # Find the treasure with the lowest total score
    best_treasure = min(total_scores, key=total_scores.get)
    return best_treasure


def expand_action_to_all_ships(state, action, unpicked_pirate_ships):
    """
    Expands the action of the closest ship to all pirate ships in the game.
    Returns:
    - A tuple of actions expanded to all pirate ships in the game.
    """
    if action == 'reset' or action == 'terminate':
        return action
    if unpicked_pirate_ships is None:
        return action
    # Extract the action and the ship identifier of the closest ship
    atomic_action = action[0]

    expanded_actions = [atomic_action]

    for pirate_ship in unpicked_pirate_ships.keys():
        if len(atomic_action) == 2:
            expanded_actions.append((atomic_action[0], pirate_ship))
        else:
            expanded_actions.append((atomic_action[0], pirate_ship, atomic_action[2]))
    # Convert the list of expanded actions into a tuple and return
    return tuple(expanded_actions)


class PirateAgent:
    def __init__(self, initial):
        self.initial = initial
        self.strategy = self.choose_strategy()
        if self.strategy == 'relaxed':
            self.reducted_initial, self.unpicked_pirate_ships, \
                self.representative_pirate_ship, self.best_treasure = reducted_initial_state(self.initial)
            self.policy = OptimalPirateAgent(self.reducted_initial).policy
        else:
            self.policy = OptimalPirateAgent(self.initial).policy

    def choose_strategy(self):
        if calc_total_states(self.initial) > THRESHOLD:
            return 'relaxed'
        else:
            return 'normal'

    def act(self, state):
        turn = state.pop('turns to go')
        if self.strategy == 'relaxed':
            state = self.minimize_state(state)
        state_json = json.dumps(state)
        best_action = self.policy[(state_json, turn)]
        if self.strategy == 'relaxed':
            best_action = expand_action_to_all_ships(state, best_action, self.unpicked_pirate_ships)
        return best_action

    def minimize_state(self, state):
        state = deepcopy(state)
        state['pirate_ships'] = {self.representative_pirate_ship:
                                     state['pirate_ships'][self.representative_pirate_ship]}
        state['treasures'] = {self.best_treasure:
                                  state['treasures'][self.best_treasure]}
        return state


def value_iteration_infinite(possible_states, next_actions_dict, next_states_dict, gamma, threshold=0.01):
    """
    Perform value iteration for an infinite number of turns.
    Args:
    - possible_states: a list of possible states
    - next_actions_dict: a dictionary mapping states to possible actions
    - next_states_dict: a dictionary mapping (state, action) pairs to lists of (next_state, probability, reward) tuples
    - gamma: the discount factor for future rewards
    - threshold: the convergence threshold

    Returns:
    - policy: a dictionary mapping states to the optimal action to take from that state.
    - V: a dictionary of state values.
    """
    V = {state: 0 for state in possible_states}  # Initialize value function
    policy = {state: None for state in possible_states}  # Initialize policy
    while True:
        delta = 0  # Initialize the maximum change in value function to zero
        for state in possible_states:
            # Compute the value for all possible actions and choose the one with the max value
            next_actions = next_actions_dict[state]
            action_values = []

            for action in next_actions:
                total_reward = 0
                next_states = next_states_dict[(state, action)]
                for (next_state, prob, reward) in next_states:
                    total_reward += prob * (reward + gamma * V[next_state])
                action_values.append((total_reward, action))

            best_value, best_action = max(action_values, key=lambda x: x[0])

            # Track the maximum change in the value function
            delta = max(delta, abs(best_value - V[state]))
            V[state] = best_value  # Update the value function
            policy[state] = best_action  # Update the policy

        # Check for convergence
        if delta < threshold:
            break

    return policy, V


class InfinitePirateAgent:
    def __init__(self, initial, gamma):
        self.initial = initial
        self.gamma = gamma

    def act(self, state):
        raise NotImplemented

    def value(self, state):
        raise NotImplemented
