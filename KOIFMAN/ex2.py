import itertools
import json
from copy import deepcopy
import math
import numpy as np

ids = ["212984801", "316111442"]


def calc_distance(x, y, n, distances_matrix):
    return distances_matrix[n * x[0] + x[1]][n * y[0] + y[1]]


def floyd_warshall(grid_map):
    size = len(grid_map[0]) * len(grid_map)  # Get number of vertices in grid.
    matrix = [[math.inf for _ in range(size)] for _ in range(size)]  # initialize distances matrix.
    gas_locations = []

    # Build first distances according to neighbouring vertices.
    for i in range(len(grid_map)):
        for j in range(len(grid_map[0])):
            if grid_map[i][j] == "G":
                gas_locations.append((i, j))
            if grid_map[i][j] == "I":
                continue
            else:
                matrix[i * len(grid_map[0]) + j][i * len(grid_map[0]) + j] = 0
                if 0 < i < len(grid_map) - 1 and 0 < j < len(grid_map[0]) - 1:
                    if grid_map[i + 1][j] != "I":
                        matrix[i * len(grid_map[0]) + j][(i + 1) * len(grid_map[0]) + j] = 1
                        matrix[(i + 1) * len(grid_map[0]) + j][i * len(grid_map[0]) + j] = 1
                    if grid_map[i - 1][j] != "I":
                        matrix[i * len(grid_map[0]) + j][(i - 1) * len(grid_map[0]) + j] = 1
                        matrix[(i - 1) * len(grid_map[0]) + j][i * len(grid_map[0]) + j] = 1
                    if grid_map[i][j + 1] != "I":
                        matrix[i * len(grid_map[0]) + j + 1][i * len(grid_map[0]) + j] = 1
                        matrix[i * len(grid_map[0]) + j][i * len(grid_map[0]) + j + 1] = 1
                    if grid_map[i][j - 1] != "I":
                        matrix[i * len(grid_map[0]) + j - 1][i * len(grid_map[0]) + j] = 1
                        matrix[i * len(grid_map[0]) + j][i * len(grid_map[0]) + j - 1] = 1

                elif i == 0 and j == 0:
                    if grid_map[i + 1][j] != "I":
                        matrix[i * len(grid_map[0]) + j][(i + 1) * len(grid_map[0]) + j] = 1
                        matrix[(i + 1) * len(grid_map[0]) + j][i * len(grid_map[0]) + j] = 1
                    if grid_map[i][j + 1] != "I":
                        matrix[i * len(grid_map[0]) + j + 1][i * len(grid_map[0]) + j] = 1
                        matrix[i * len(grid_map[0]) + j][i * len(grid_map[0]) + j + 1] = 1

                elif i == 0 and j < len(grid_map[0]) - 1:
                    if grid_map[i + 1][j] != "I":
                        matrix[i * len(grid_map[0]) + j][(i + 1) * len(grid_map[0]) + j] = 1
                        matrix[(i + 1) * len(grid_map[0]) + j][i * len(grid_map[0]) + j] = 1
                    if grid_map[i][j + 1] != "I":
                        matrix[i * len(grid_map[0]) + j + 1][i * len(grid_map[0]) + j] = 1
                        matrix[i * len(grid_map[0]) + j][i * len(grid_map[0]) + j + 1] = 1
                    if grid_map[i][j - 1] != "I":
                        matrix[i * len(grid_map[0]) + j - 1][i * len(grid_map[0]) + j] = 1
                        matrix[i * len(grid_map[0]) + j][i * len(grid_map[0]) + j - 1] = 1

                elif i == 0 and j == len(grid_map[0]) - 1:
                    if grid_map[i + 1][j] != "I":
                        matrix[i * len(grid_map[0]) + j][(i + 1) * len(grid_map[0]) + j] = 1
                        matrix[(i + 1) * len(grid_map[0]) + j][i * len(grid_map[0]) + j] = 1
                    if grid_map[i][j - 1] != "I":
                        matrix[i * len(grid_map[0]) + j - 1][i * len(grid_map[0]) + j] = 1
                        matrix[i * len(grid_map[0]) + j][i * len(grid_map[0]) + j - 1] = 1

                elif i < len(grid_map) - 1 and j == 0:
                    if grid_map[i + 1][j] != "I":
                        matrix[i * len(grid_map[0]) + j][(i + 1) * len(grid_map[0]) + j] = 1
                        matrix[(i + 1) * len(grid_map[0]) + j][i * len(grid_map[0]) + j] = 1
                    if grid_map[i - 1][j] != "I":
                        matrix[i * len(grid_map[0]) + j][(i - 1) * len(grid_map[0]) + j] = 1
                        matrix[(i - 1) * len(grid_map[0]) + j][i * len(grid_map[0]) + j] = 1
                    if grid_map[i][j + 1] != "I":
                        matrix[i * len(grid_map[0]) + j + 1][i * len(grid_map[0]) + j] = 1
                        matrix[i * len(grid_map[0]) + j][i * len(grid_map[0]) + j + 1] = 1

                elif i == len(grid_map) - 1 and j == 0:
                    if grid_map[i - 1][j] != "I":
                        matrix[i * len(grid_map[0]) + j][(i - 1) * len(grid_map[0]) + j] = 1
                        matrix[(i - 1) * len(grid_map[0]) + j][i * len(grid_map[0]) + j] = 1
                    if grid_map[i][j + 1] != "I":
                        matrix[i * len(grid_map[0]) + j + 1][i * len(grid_map[0]) + j] = 1
                        matrix[i * len(grid_map[0]) + j][i * len(grid_map[0]) + j + 1] = 1

                elif i == len(grid_map) - 1 and j == len(grid_map[0]) - 1:
                    if grid_map[i - 1][j] != "I":
                        matrix[i * len(grid_map[0]) + j][(i - 1) * len(grid_map[0]) + j] = 1
                        matrix[(i - 1) * len(grid_map[0]) + j][i * len(grid_map[0]) + j] = 1
                    if grid_map[i][j - 1] != "I":
                        matrix[i * len(grid_map[0]) + j - 1][i * len(grid_map[0]) + j] = 1
                        matrix[i * len(grid_map[0]) + j][i * len(grid_map[0]) + j - 1] = 1

                elif i < len(grid_map) - 1 and j == len(grid_map[0]) - 1:
                    if grid_map[i + 1][j] != "I":
                        matrix[i * len(grid_map[0]) + j][(i + 1) * len(grid_map[0]) + j] = 1
                        matrix[(i + 1) * len(grid_map[0]) + j][i * len(grid_map[0]) + j] = 1
                    if grid_map[i - 1][j] != "I":
                        matrix[i * len(grid_map[0]) + j][(i - 1) * len(grid_map[0]) + j] = 1
                        matrix[(i - 1) * len(grid_map[0]) + j][i * len(grid_map[0]) + j] = 1
                    if grid_map[i][j - 1] != "I":
                        matrix[i * len(grid_map[0]) + j - 1][i * len(grid_map[0]) + j] = 1
                        matrix[i * len(grid_map[0]) + j][i * len(grid_map[0]) + j - 1] = 1

                elif i == len(grid_map) - 1 and j < len(grid_map[0]) - 1:
                    if grid_map[i - 1][j] != "I":
                        matrix[i * len(grid_map[0]) + j][(i - 1) * len(grid_map[0]) + j] = 1
                        matrix[(i - 1) * len(grid_map[0]) + j][i * len(grid_map[0]) + j] = 1
                    if grid_map[i][j + 1] != "I":
                        matrix[i * len(grid_map[0]) + j + 1][i * len(grid_map[0]) + j] = 1
                        matrix[i * len(grid_map[0]) + j][i * len(grid_map[0]) + j + 1] = 1
                    if grid_map[i][j - 1] != "I":
                        matrix[i * len(grid_map[0]) + j - 1][i * len(grid_map[0]) + j] = 1
                        matrix[i * len(grid_map[0]) + j][i * len(grid_map[0]) + j - 1] = 1

    # Finding optimal distances via floyd warshall algorithm.
    # Taking all vertices one by one and setting them as intermediate vertices
    for k in range(size):
        # Pick all vertices as source one by one.
        for s in range(size):
            # Pick all vertices as the destination for the above chosen source vertex.
            for r in range(size):
                # Update the value of matrix[i][j] if k provides the shortest path from i to j
                matrix[s][r] = min(matrix[s][r], matrix[s][k] + matrix[k][r])

    for i in range(size):
        for j in range(size):
            if matrix[i][j] == math.inf:
                matrix[i][j] = -1

    return matrix, gas_locations


def check_possible_grid_moves(location, taxi_name, grid_map):
    """
    Check possible moves in the grid.
    """
    possible_moves = []
    x = location[0]
    y = location[1]
    if x > 0:
        possible_moves.append(("move", taxi_name, (x - 1, y))) if grid_map[x - 1][y] != 'I' else None
    if x < len(grid_map) - 1:
        possible_moves.append(("move", taxi_name, (x + 1, y))) if grid_map[x + 1][y] != 'I' else None
    if y > 0:
        possible_moves.append(("move", taxi_name, (x, y - 1))) if grid_map[x][y - 1] != 'I' else None
    if y < len(grid_map[0]) - 1:
        possible_moves.append(("move", taxi_name, (x, y + 1))) if grid_map[x][y + 1] != 'I' else None

    return possible_moves


def check_if_contains_passenger(taxi_location, passengers, taxi_name, taxi_passengers):
    """
    Check if the location contains a passenger and if the taxi hasn't already picked him,
    nor did the passenger reach his destination.
    """
    passenger_actions = []
    for passenger in passengers.keys():
        if passengers[passenger]["location"] == taxi_location and passenger not in taxi_passengers and \
                passengers[passenger]["location"] != passengers[passenger]["destination"]:
            passenger_actions.append(("pick up", taxi_name, passenger))
    return passenger_actions


def check_drop_off_passenger(location, passengers_of_taxi, taxi_name, all_passengers):
    """
    Check if a passenger reached his destination.
    """
    passenger_actions = []
    for passenger in passengers_of_taxi:
        if all_passengers[passenger]["destination"] == location:
            passenger_actions.append(("drop off", taxi_name, passenger))
    return passenger_actions


def extract_locations(action, state):
    """
    Extract the locations from the actions.
    """
    locations = []
    for taxi_action in action:
        if taxi_action[0] == "move":
            locations.append(tuple(taxi_action[2]))
        else:
            locations.append(tuple(state["taxis"][taxi_action[1]]["location"]))

    return locations


def eliminate_not_valid_actions(all_actions, state):
    """
    Check all actions and eliminate the ones that are not valid (2 taxis in the same location)
    """
    new_all_actions = []
    if len(state["taxis"]) > 1:
        for action in all_actions:
            locations = extract_locations(action, state)
            if len(locations) == len(set(locations)):
                new_all_actions.append(action)
        return new_all_actions
    return all_actions


def get_pickedup_passengers(state):
    """
    Build a dictionary that contains the name of passengers that are on each taxi.
    """
    taxis_passengers = {}
    passengers = state["passengers"]
    for taxi in state["taxis"].keys():
        taxis_passengers[taxi] = []

    for passenger_name, passenger_dict in passengers.items():
        if isinstance(passenger_dict["location"], str):
            taxis_passengers[passenger_dict["location"]].append(passenger_name)

    return taxis_passengers


def actions(state):
    """
    Build possible actions for each taxi given a specific state.
    """
    possible_actions = {}
    taxis_passengers = get_pickedup_passengers(state)
    grid_map = state["map"]

    for taxi in state["taxis"]:
        possible_actions[taxi] = []
        # Checking if the taxi can move in the grid.
        if state["taxis"][taxi]["fuel"] > 0:
            possible_actions[taxi] = possible_actions[taxi] + check_possible_grid_moves(
                state["taxis"][taxi]["location"], taxi, grid_map)

        # Checking if the taxi can pick up a passenger.
        if state["taxis"][taxi]["capacity"] > 0:
            possible_actions[taxi] = possible_actions[taxi] + check_if_contains_passenger(
                state["taxis"][taxi]["location"], state["passengers"], taxi, taxis_passengers[taxi])

        # Checking if the taxi can drop off a passenger.
        possible_actions[taxi] = possible_actions[taxi] + check_drop_off_passenger(
            state["taxis"][taxi]["location"], taxis_passengers[taxi], taxi, state["passengers"])

        # Checking if the taxi can refuel.
        if grid_map[state["taxis"][taxi]["location"][0]][state["taxis"][taxi]["location"][1]] == 'G':
            possible_actions[taxi] = possible_actions[taxi] + [("refuel", taxi)]

        possible_actions[taxi] = possible_actions[taxi] + [("wait", taxi)]

    all_actions = list(itertools.product(*list(possible_actions.values())))
    legal_actions = eliminate_not_valid_actions(all_actions, state)
    legal_actions.append("reset")

    return legal_actions


def merge_product_dicts(**kwargs):
    """
    Merge and cross product dicts - Build permutations.
    """
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


def delete_illegal_states(all_states, initial):
    """
    Keep and return legal states only from all states' permutations.
    """
    legal_states = []

    for state in all_states:
        is_legal = True
        picked_up_dict = get_pickedup_passengers(state)
        taxis = state["taxis"]
        taxis_location = []

        for taxi in state["taxis"]:
            taxis_location.append(tuple(taxis[taxi]["location"]))
            if initial["taxis"][taxi]["capacity"] - len(picked_up_dict[taxi]) != state["taxis"][taxi]["capacity"]:
                is_legal = False
                break

        if is_legal and len(set(tuple(taxis_location))) == len(taxis_location):
            legal_states.append(json.dumps(state, sort_keys=True))

    return legal_states


def build_states(initial):
    """
    Build all states permutations.
    """
    # Initialize variables.
    taxis = initial["taxis"]
    grid_map = initial["map"]
    rows_num = len(grid_map)
    cols_num = len(grid_map[0])
    locations = [(x, y) for x in range(rows_num) for y in range(cols_num) if grid_map[x][y] != 'I']
    states_per_taxi = {}
    states_per_passenger = {}

    for taxi in taxis:
        states_per_taxi[taxi] = []
        for current_fuel in range(initial["taxis"][taxi]["fuel"] + 1):
            for capacity in range(initial["taxis"][taxi]["capacity"],
                                  max(0, initial["taxis"][taxi]["capacity"] - len(initial["passengers"].keys())) - 1, -1):
                for location in locations:
                    states_per_taxi[taxi].append({"location": location, "fuel": current_fuel, "capacity": capacity})

    for passenger, passenger_dict in initial["passengers"].items():
        states_per_passenger[passenger] = []
        passenger_destinations = set(passenger_dict["possible_goals"]).union({passenger_dict["destination"]})
        passenger_locations = {passenger_dict["location"]}.union(taxis.keys(), passenger_destinations)

        for loc in passenger_locations:
            for dest in passenger_destinations:
                states_per_passenger[passenger].append({"location": loc, "destination": dest,
                                                        "possible_goals": passenger_dict["possible_goals"],
                                                        "prob_change_goal": passenger_dict["prob_change_goal"]})

    taxis_states = list(merge_product_dicts(**states_per_taxi))
    passengers_states = list(merge_product_dicts(**states_per_passenger))
    states_to_permute = {"optimal": [initial["optimal"]], "map": [grid_map], "taxis": taxis_states,
                         "passengers": passengers_states}

    all_states = list(merge_product_dicts(**states_to_permute))
    str_legal_states = delete_illegal_states(all_states, initial)

    return str_legal_states


def calc_total_reward(action):
    total_reward = 0

    if action == "reset":
        return -50

    for atomic_action in action:
        if atomic_action[0] == "drop off":
            total_reward += 100
        elif atomic_action[0] == "refuel":
            total_reward -= 10

    return total_reward


def apply_action(state, initial, action):
    """
    Alter state and provide next state according to given action.
    """

    if action == "reset":
        next_state = deepcopy(initial)

    else:
        next_state = deepcopy(state)

        for atomic_action in action:
            taxi_name = atomic_action[1]
            if atomic_action[0] == 'move':
                next_state['taxis'][taxi_name]['location'] = atomic_action[2]
                next_state['taxis'][taxi_name]['fuel'] -= 1

            elif atomic_action[0] == 'pick up':
                passenger_name = atomic_action[2]
                next_state['taxis'][taxi_name]['capacity'] -= 1
                next_state['passengers'][passenger_name]['location'] = taxi_name

            elif atomic_action[0] == 'drop off':
                passenger_name = atomic_action[2]
                next_state['passengers'][passenger_name]['location'] = next_state['taxis'][taxi_name]['location']
                next_state['taxis'][taxi_name]['capacity'] += 1

            elif atomic_action[0] == 'refuel':
                next_state['taxis'][taxi_name]['fuel'] = initial['taxis'][taxi_name]['fuel']

    return next_state


def get_states_and_probs(state, initial, action):
    """
    Build next states' and their probability dictionary.
    """
    states_per_passenger = {}
    next_states = []
    next_state = apply_action(state, initial, action)

    if action == "reset":
        probability = 1
        next_states.append((json.dumps(next_state, sort_keys=True), probability))

    else:
        for passenger, passenger_dict in next_state["passengers"].items():
            states_per_passenger[passenger] = []
            if passenger_dict["destination"] in passenger_dict["possible_goals"]:
                passenger_destinations = passenger_dict["possible_goals"]
            else:
                passenger_destinations = passenger_dict["possible_goals"] + [passenger_dict["destination"]]

            for dest in passenger_destinations:
                states_per_passenger[passenger].append({"location": passenger_dict["location"], "destination": dest,
                                                        "possible_goals": passenger_dict["possible_goals"],
                                                        "prob_change_goal": passenger_dict["prob_change_goal"]})

        passengers_states = list(merge_product_dicts(**states_per_passenger))
        states_to_permute = {"optimal": [next_state["optimal"]], "map": [next_state["map"]], "taxis": [next_state["taxis"]],
                             "passengers": passengers_states}

        all_next_states = list(merge_product_dicts(**states_to_permute))
        legal_next_states = delete_illegal_states(all_next_states, initial)

        for nxt_state in legal_next_states:
            nxt_state_dict = json.loads(nxt_state)
            probability = 1

            for passenger in nxt_state_dict["passengers"]:
                n = len(state["passengers"][passenger]["possible_goals"])
                p = state["passengers"][passenger]["prob_change_goal"]
                if nxt_state_dict["passengers"][passenger]["destination"] != state["passengers"][passenger]["destination"]:
                    probability *= p/n
                else:
                    if state["passengers"][passenger]["destination"] in state["passengers"][passenger]["possible_goals"]:
                        probability *= 1 - p + p/n
                    else:
                        probability *= 1 - p

            next_states.append((nxt_state, probability))

    return next_states


def build_next_states_dict(possible_states, initial):
    """
    A wrap function - build next states dictionary.
    """
    next_states_dict = {}
    actions_dict = {}

    for str_state in possible_states:
        dict_state = json.loads(str_state)
        possible_actions_from_s = actions(dict_state)
        actions_dict[str_state] = possible_actions_from_s
        for action in possible_actions_from_s:
            next_states_dict[str_state, action] = get_states_and_probs(dict_state, initial, action)

    return next_states_dict, actions_dict


def calc_states_total(initial):
    """
    Calculate number of states.
    """
    rows_num = len(initial["map"])
    cols_num = len(initial["map"][0])
    poss_locations = len([(x, y) for x in range(rows_num) for y in range(cols_num) if initial["map"][x][y] != "I"])

    taxis_num = len(initial["taxis"].keys())
    total_states = 1

    for taxi in initial["taxis"]:
        total_states *= (initial["taxis"][taxi]["fuel"] + 1) * poss_locations
    for passenger in initial["passengers"]:
        destinations = set(initial["passengers"][passenger]["possible_goals"]).union({initial["passengers"]
                                                                                      [passenger]["destination"]})
        locations_num = len({initial["passengers"][passenger]["location"]}.union(destinations)) + taxis_num
        total_states *= locations_num * len(destinations)

    total_states *= initial["turns to go"]

    return total_states


def find_closest_station(gas_stations, location, distances_matrix, n):
    """
    Find the closest gas station to a specific location.
    """
    distances = []
    for gas_station in gas_stations:
        distances.append([gas_station, calc_distance(location, gas_station, n, distances_matrix)])
    min_gas_station, min_distance = min(distances, key=lambda x: x[1])
    return min_gas_station, min_distance


class OptimalTaxiAgent:
    def __init__(self, initial):
        self.initial = deepcopy(initial)
        self.turns_to_go = self.initial.pop("turns to go")
        self.possible_states = build_states(self.initial)
        self.next_states_dict, self.actions_dict = build_next_states_dict(self.possible_states, self.initial)
        self.optimal_policy, self.v_star = self.value_iteration()

    def act(self, state):
        current_state = deepcopy(state)
        t = current_state.pop("turns to go")
        return self.optimal_policy[json.dumps(current_state, sort_keys=True), t]

    def value_iteration(self):
        v_star = {}
        policy = {}

        for t in range(self.turns_to_go + 1):
            for str_state in self.possible_states:
                maximum_value = -math.inf
                if t == 0:
                    v_star[str_state, t] = 0
                    continue

                possible_actions_from_s = self.actions_dict[str_state]
                for action in possible_actions_from_s:
                    R = calc_total_reward(action)
                    Q = R + sum([p * v_star[next_state, t - 1] for next_state, p in
                                 self.next_states_dict[str_state, action]])

                    if Q > maximum_value:
                        argmax = action
                        maximum_value = Q

                v_star[str_state, t] = maximum_value
                policy[str_state, t] = argmax

        return policy, v_star


class TaxiAgent:
    def __init__(self, initial):
        self.initial = initial
        self.map = initial["map"]
        self.is_relaxed = False
        self.retry = False

        if calc_states_total(self.initial) > 7000000 and (len(self.initial["taxis"].keys()) > 1 or
                                                          len(self.initial["passengers"].keys()) > 1):
            self.is_relaxed = True
            self.relaxed_initial, self.best_taxi, self.best_passenger, self.secondary_passenger = self.relax_input()
            # If the relaxed input is small enough and there is another optional passenger to add, add it to input.
            if calc_states_total(self.relaxed_initial) < 380000 and self.secondary_passenger is not None:
                self.relaxed_initial["passengers"][self.secondary_passenger] = self.initial["passengers"][self.secondary_passenger]

            self.policy = OptimalTaxiAgent(self.relaxed_initial).optimal_policy

        else:
            self.policy = OptimalTaxiAgent(self.initial).optimal_policy

    def act(self, state):
        current_state = deepcopy(state)
        t = current_state.pop("turns to go")
        action = []
        # If relaxation was made, make state suitable for policy keys.
        if self.is_relaxed:
            current_state["taxis"] = {self.best_taxi: current_state["taxis"][self.best_taxi]}
            current_state["passengers"] = {}
            current_state["map"] = self.relaxed_initial["map"]

            for passenger in self.relaxed_initial["passengers"]:
                current_state["passengers"][passenger] = state["passengers"][passenger]

            act = self.policy[json.dumps(current_state, sort_keys=True), t]
            if act == "reset":
                return act

            action.append(act[0])
            for taxi, taxi_dict in state["taxis"].items():
                if taxi == self.best_taxi:
                    continue
                # Make sure no collisions occur.
                if self.retry:
                    if act[0][0] == "move":
                        if taxi_dict["location"] == act[0][2]:
                            if taxi_dict["fuel"] == 0:
                                return "terminate"
                            else:
                                action.append(("move", taxi, state["taxis"][self.best_taxi]["location"]))
                        else:
                            action.append(("wait", taxi))
                    else:
                        action.append(("wait", taxi))
                else:
                    action.append(("wait", taxi))

            return tuple(action)
        else:
            return self.policy[json.dumps(current_state, sort_keys=True), t]

    def relax_input(self):
        """
        Original input's relaxation process.
        """
        relaxed_initial = deepcopy(self.initial)
        taxi_pass_score = self.get_best_taxi_passenger(relaxed_initial)
        minimum = math.inf
        secondary_passenger = None

        if len(taxi_pass_score) > 0:
            best_taxi, best_passenger, relaxed_map = min(taxi_pass_score, key=lambda x: x[3])[:3]
        else:
            self.retry = True
            taxi_pass_score = self.get_best_taxi_passenger(relaxed_initial)
            if len(taxi_pass_score) > 0:
                best_taxi, best_passenger, relaxed_map = min(taxi_pass_score, key=lambda x: x[3])[:3]

        if len(taxi_pass_score) == 0:
            best_taxi = list(relaxed_initial["taxis"].keys())[0]
            best_passenger = list(relaxed_initial["passengers"].keys())[0]
            relaxed_map = self.initial["map"]
        else:
            # Search for the second most suitable passenger for best taxi.
            if len(relaxed_initial["passengers"].keys()) > 1:
                for key in taxi_pass_score:
                    if key[0] == best_taxi and key[1] != best_passenger:
                        if key[3] < minimum:
                            minimum = key[3]
                            secondary_passenger = key[1]

        relaxed_initial["taxis"] = {best_taxi: relaxed_initial["taxis"][best_taxi]}
        relaxed_initial["passengers"] = {best_passenger: relaxed_initial["passengers"][best_passenger]}
        relaxed_initial["map"] = relaxed_map

        return relaxed_initial, best_taxi, best_passenger, secondary_passenger

    def get_best_taxi_passenger(self, relaxed_initial):
        """
        Get best pair of taxi-passenger according to a heuristic-based score.
        """
        taxi_pass_score = []
        grid = relaxed_initial["map"]
        n = len(grid[0])

        for curr_taxi, curr_taxi_dict in relaxed_initial["taxis"].items():
            curr_taxi_fuel = curr_taxi_dict["fuel"]
            copied_grid = deepcopy(grid)
            if not self.retry:
                for banned_taxi, banned_taxi_dict in relaxed_initial["taxis"].items():
                    if banned_taxi != curr_taxi:
                        copied_grid[banned_taxi_dict["location"][0]][banned_taxi_dict["location"][1]] = "I"

            distances_matrix, gas_locations = floyd_warshall(copied_grid)

            for passenger, passenger_dict in relaxed_initial["passengers"].items():
                p = passenger_dict["prob_change_goal"]
                passenger_location = passenger_dict["location"]
                taxi_passenger_distance = calc_distance(curr_taxi_dict["location"], passenger_location, n,
                                                        distances_matrix)
                if taxi_passenger_distance == -1:
                    continue

                unreachable_num = 0
                unreachable_penalty = 0
                distance = calc_distance(passenger_location, passenger_dict["destination"], n, distances_matrix)
                if distance == -1:
                    unreachable_num += 1
                    unreachable_penalty += 2 * (1-p)
                passenger_destination_distance = abs(distance * (1-p))
                poss_goals = passenger_dict["possible_goals"]
                passenger_goals = set(poss_goals).union({passenger_dict["destination"]})

                for poss_destination in poss_goals:
                    distance = calc_distance(passenger_location, poss_destination, n, distances_matrix)
                    if distance == -1:
                        unreachable_penalty += 2 * (p/len(poss_goals))
                        if poss_destination != passenger_dict["destination"]:
                            unreachable_num += 1
                    passenger_destination_distance += abs(distance * (p/len(poss_goals)))
                # If passenger is unreachable or all his destinations are unreachable, the pair is not feasible.
                if unreachable_num == len(passenger_goals):
                    continue
                # If taxi's fuel is not enough, search for closest gas stations.
                if curr_taxi_fuel < taxi_passenger_distance + passenger_destination_distance:
                    # If there are no gas stations, the pair is not feasible.
                    if len(gas_locations) == 0:
                        continue
                    closest_gas_station, min_dist_to_station = find_closest_station(gas_locations,
                                                                                    curr_taxi_dict["location"],
                                                                                    distances_matrix, n)
                    # If no gas station is reachable, the pair is not feasible,
                    if min_dist_to_station > curr_taxi_fuel:
                        continue
                    # If taxi can't reach passenger directly, calculate distance to passenger through gas station.
                    if curr_taxi_fuel < taxi_passenger_distance:
                        station_passenger_dist = calc_distance(closest_gas_station, passenger_location, n, distances_matrix)
                        taxi_passenger_distance = min_dist_to_station + station_passenger_dist +\
                                                  math.ceil(station_passenger_dist/curr_taxi_fuel)
                    # Add number of refuel needed for the taxi to take the passenger to his destination.
                    if curr_taxi_fuel < passenger_destination_distance:
                        passenger_destination_distance += math.ceil(passenger_destination_distance/curr_taxi_fuel)
                # Calculate distance
                distances_sum = taxi_passenger_distance + passenger_destination_distance
                # Calculate passenger's destinations' variance
                destinations_mean = np.mean(np.array(list(passenger_goals)), axis=0)
                destinations_variance = ((np.array(passenger_dict["destination"]) - destinations_mean) ** 2) * (1 - p)
                destinations_variance += sum([((np.array(dest) - destinations_mean) ** 2) * (p / len(poss_goals))
                                             for dest in poss_goals])
                destination_variance = sum(destinations_variance)
                # Calculate pair's score as the sum of variance and distance.
                taxi_pass_score.append([curr_taxi, passenger, copied_grid, distances_sum + destination_variance + unreachable_penalty])

        return taxi_pass_score
