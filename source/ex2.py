import itertools
import random

ids = ["0", "0"]


class QItem:
    def __init__(self, row, col, dist, parent):
        self.row = row
        self.col = col
        self.dist = dist
        self.parent = parent

    def __repr__(self):
        return f"QItem({self.row}, {self.col}, {self.dist})"


# checking where move is valid or not
def is_valid(x, y, grid, visited):
    if ((x >= 0 and y >= 0) and
            (x < len(grid) and y < len(grid[0])) and
            (grid[x][y] != 'I') and (visited[x][y] == False)):
        return True
    return False


def create_path(node):
    path = [(node.row, node.col)]
    while node.parent is not None:
        path.append((node.parent.row, node.parent.col))
        node = node.parent

    path.reverse()
    return path


def min_distance(grid, source_loc, dest_loc):
    if grid[dest_loc[0]][dest_loc[1]] == 'I':
        return {"distance": float('inf'), "path": [-1, -1]}

    source = QItem(0, 0, 0, None)
    source.row = source_loc[0]
    source.col = source_loc[1]

    # To maintain location visit status
    visited = [[False for _ in range(len(grid[0]))]
               for _ in range(len(grid))]

    # applying BFS on matrix cells starting from source
    queue = [source]
    visited[source.row][source.col] = True
    while len(queue) != 0:
        source = queue.pop(0)

        # Destination found;
        if source.row == dest_loc[0] and source.col == dest_loc[1]:
            return {"distance": source.dist, "path": create_path(source)}

        # moving up
        if is_valid(source.row - 1, source.col, grid, visited):
            queue.append(QItem(source.row - 1, source.col, source.dist + 1, source))
            visited[source.row - 1][source.col] = True

        # moving down
        if is_valid(source.row + 1, source.col, grid, visited):
            queue.append(QItem(source.row + 1, source.col, source.dist + 1, source))
            visited[source.row + 1][source.col] = True

        # moving left
        if is_valid(source.row, source.col - 1, grid, visited):
            queue.append(QItem(source.row, source.col - 1, source.dist + 1, source))
            visited[source.row][source.col - 1] = True

        # moving right
        if is_valid(source.row, source.col + 1, grid, visited):
            queue.append(QItem(source.row, source.col + 1, source.dist + 1, source))
            visited[source.row][source.col + 1] = True

        # moving right and up
        if is_valid(source.row - 1, source.col + 1, grid, visited):
            queue.append(QItem(source.row - 1, source.col + 1, source.dist + 1, source))
            visited[source.row - 1][source.col + 1] = True

        # moving left and up
        if is_valid(source.row - 1, source.col - 1, grid, visited):
            queue.append(QItem(source.row - 1, source.col - 1, source.dist + 1, source))
            visited[source.row - 1][source.col - 1] = True

        # moving down and right
        if is_valid(source.row + 1, source.col + 1, grid, visited):
            queue.append(QItem(source.row + 1, source.col + 1, source.dist + 1, source))
            visited[source.row + 1][source.col + 1] = True

        # moving down and left
        if is_valid(source.row + 1, source.col - 1, grid, visited):
            queue.append(QItem(source.row + 1, source.col - 1, source.dist + 1, source))
            visited[source.row + 1][source.col - 1] = True

    # return float("inf")
    return {"distance": float('inf'), "path": [-1, -1]}


def flatten(t):
    return set([item for sublist in t for item in sublist])


def num_delivery(action):
    if action == "reset" or action == "terminate":
        return 0
    count = 0
    for atomic_action in action:
        if atomic_action[0] == "deliver":
            count += 1
    return count


def valid_action(action):
    picked_packages = [a[2] for a in action if a[0] == "pick up"]
    if len(picked_packages) != len(set(picked_packages)):
        return False
    return True


def match_package_to_client(initial):
    package_to_client = {}
    for client, properties in initial["clients"].items():
        for packages in properties["packages"]:
            package_to_client[packages] = client

    return package_to_client


def client_next_move(properties, state):
    movements = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
    new_coordinates = (properties["location"][0], properties["location"][1])
    for _ in range(1000):
        movement = random.choices(movements, weights=properties["probabilities"])[0]
        new_coordinates = (properties["location"][0] + movement[0], properties["location"][1] + movement[1])
        if new_coordinates[0] < 0 or new_coordinates[1] < 0 or new_coordinates[0] >= len(state["map"]) or \
                new_coordinates[1] >= len(state["map"][0]):
            continue
        break
    return new_coordinates


class DroneAgent:
    def __init__(self, initial):
        self.current_time = 0
        self.reset_times = [0]
        self.grid = initial["map"]
        self.grid_rows = len(self.grid)
        self.grid_cols = len(self.grid[0])
        self.path = {}
        self.package_to_client = match_package_to_client(initial)
        self.package_time_laps = self.initial_package_time_laps(initial)

    def initial_package_time_laps(self, initial):
        package_time_laps = {}
        for client, properties in initial["clients"].items():
            for package in properties["packages"]:
                package_time_laps[package] = []
        return package_time_laps

    def is_in_grid(self, loc):
        """
        :param loc:
        :return:
        """
        return 0 <= loc[0] < self.grid_rows and 0 <= loc[1] < self.grid_cols and self.grid[loc[0]][loc[1]] == 'P'

    def distance(self, source, destination):
        if (source, destination) not in self.path.keys():
            if source == destination:
                self.path[(source, destination)] = {"distance": 0, "path": [source, source]}
            else:
                self.path[(source, destination)] = min_distance(self.grid, source, destination)

        return self.path[(source, destination)]

    def greedy_matching(self, state, remaining_pick_packages, remaining_drones):
        drones_packeges_matcing = {}
        num_drones = len(remaining_drones)
        for i, package in enumerate(remaining_pick_packages):
            if i == num_drones:
                break

            drones_scores = [(drone, self.distance(state["drones"][drone], state["packages"][package])["distance"],
                              self.distance(state["drones"][drone], state["packages"][package])["path"][1]) for drone in
                             remaining_drones]
            drones_scores.sort(key=lambda x: x[1])
            next_step = drones_scores[0][2]
            if next_step != -1:
                drones_packeges_matcing[drones_scores[0][0]] = (
                    "move", drones_scores[0][0], drones_scores[0][2])  # TODO: check for bugs
                remaining_drones.remove(drones_scores[0][0])
        return drones_packeges_matcing

    def full_drone(self, state, drone):
        drone_loc = state["drones"][drone]
        packages_scores = []
        for package, loc in state["packages"].items():
            if loc == drone:
                properties = state["clients"][self.package_to_client[package]]
                next_move = client_next_move(properties, state)
                package_score = (("move", drone, self.distance(drone_loc, next_move)["path"][1]),
                                 self.distance(drone_loc, next_move)["distance"])
                packages_scores.append(package_score)

        packages_scores.sort(key=lambda x: x[1])
        if packages_scores[0][0][2] == -1:
            return "wait", drone
        return packages_scores[0][0]

    def act(self, state):
        self.current_time += 1

        if len(state["packages"].keys()) == 0:
            num_positive_packages = sum([1 for package, times in self.package_time_laps.items() if
                                         sum(times) / len(times) < state["turns to go"]])
            if num_positive_packages * 10 > 15:
                self.reset_times.append(self.current_time)
                return "reset"

        remaining_drones = list(state["drones"].keys())
        remaining_packages = [package for package, loc in state["packages"].items() if
                              type(loc) == tuple and package in flatten([properties["packages"] for properties in
                                                                         state["clients"].values()])]
        final_action = []
        drones_actions = {}
        for drone_name, drone_loc in state["drones"].items():
            drones_actions[drone_name] = self.single_drone_actions(state, drone_name, drone_loc)

        for drone, actions in drones_actions.items():
            for action in actions:
                # delivery actions
                if action[0] == "deliver":
                    final_action.append(action)
                    remaining_drones.remove(drone)
                    self.package_time_laps[action[3]].append(self.current_time - self.reset_times[-1])
                    break

                # delivery actions
                if action[0] == "pick up" and action[2] in remaining_packages:
                    final_action.append(action)
                    remaining_drones.remove(drone)
                    remaining_packages.remove(action[2])
                    break

        drones_num_carries = {}
        for drone in remaining_drones:
            drones_num_carries[drone] = 0

        for package, loc in state["packages"].items():
            if type(loc) == str:
                if loc in remaining_drones:
                    drones_num_carries[loc] += 1

        # drones with 2 packages on them
        final_action += [self.full_drone(state, drone) for drone, num_carries in drones_num_carries.items() if
                         num_carries == 2]

        remaining_drones = [drone for drone, num_carries in drones_num_carries.items() if num_carries < 2]

        for drone in remaining_drones:
            if drones_num_carries[drone] == 1:
                drone_next_step = self.full_drone(state, drone)
                if drone_next_step[0] != "wait":
                    final_action.append(drone_next_step)
                    remaining_drones.remove(drone)

        num_remaining_drones = len(remaining_drones)
        count = 0
        while len(remaining_drones) > 0 and count < num_remaining_drones:
            drones_packages_matching = self.greedy_matching(state, remaining_packages, remaining_drones)
            final_action += list(drones_packages_matching.values())
            count += 1

        for drone in remaining_drones:
            final_action.append(("wait", drone))

        return final_action

    def single_drone_actions(self, state, drone_name, drone_loc):
        """

        :param state:
        :param drone_name:
        :param drone_loc:
        :return:
        """
        drone_num_packages_carried = len([1 for p in state["packages"] if state["packages"][p] == drone_name])
        actions = [("wait", drone_name)]

        # move grid actions
        for step in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            new_loc = (drone_loc[0] + step[0], drone_loc[1] + step[1])
            if self.is_in_grid(new_loc):
                actions.append(("move", drone_name, new_loc))

        for package_name, package_loc in state["packages"].items():
            if drone_num_packages_carried < 2:
                # pick up actions
                if drone_loc == package_loc:
                    # if package is carried by other drone than package loc is null -> False
                    actions.append(("pick up", drone_name, package_name))
            if drone_num_packages_carried > 0:
                # deliver actions
                for client_name, properties in state["clients"].items():
                    if drone_loc == properties["location"] and package_name in properties[
                        "packages"] and package_loc == drone_name:
                        actions.append(("deliver", drone_name, client_name, package_name))
        return actions

    def actions(self, state):
        """
        Returns all the actions that can be executed in the given
        state. The result should be a tuple (or other iterable) of actions
        as defined in the problem description file
        :param state:
        :return:
        """

        drones_actions = []
        for drone_name, drone_loc in state["drones"].items():
            drones_actions.append(self.single_drone_actions(state, drone_name, drone_loc))

        actions = [sub_action for sub_action in itertools.product(*drones_actions) if valid_action(sub_action)]
        return ["reset", "terminate"] + actions
