import math
import scipy.stats as st
import numpy as np
import random
import itertools
from scipy import ndimage

# This is my lazy solution for global variables in multithreading
args = None
width, height = None, None
scale_factor = None

density_water = None
gravity = None

m_min, m_max, m_avg, m_dev, m_static = None, None, None, None, None
hemispheres_enabled, max_hemispheres = None, None
friction_constant_force = None
attraction_radius = None

beta, time_step, t_max = None, None, None
residual_floor, residual_ceil = None, None

active_drops, drop_array, new_drops = None, None, None

height_map, kernel = None, None
floor_value = None


class Droplet:
    def __init__(self, x, y, mass, velocity=0):
        self.x = x
        self.y = y
        self.mass = mass
        self.velocity = velocity
        self.direction = 0
        self.t_i = 0
        self.path = []
        self.hemispheres = [(self.x, self.y)]  # (x,y) tuples (could add z to represent share of mass)
        if mass < m_static and hemispheres_enabled:
            self.generate_hemispheres()

        self.radius = 0
        self.calculate_radius()

    def calculate_radius(self):
        # Only called when mass changes to avoid excessive calculations
        # TODO: Adjust radius calculations somehow
        self.radius = np.cbrt((3 / 2) / math.pi * (self.mass / len(self.hemispheres)) / density_water) / scale_factor * width

    def generate_hemispheres(self):
        # Random walk to decide locations of hemispheres.
        self.hemispheres = [(self.x, self.y)]
        num_hemispheres = random.randint(1, max_hemispheres)
        directions = (1, 2, 3, 4)
        next_dirs = directions
        while len(self.hemispheres) < num_hemispheres:
            new_x, new_y = self.hemispheres[-1]
            direction = random.choice(next_dirs)
            if direction == 1:  # Down
                next_dirs = (1, 2, 4)
                new_y -= 1
            if direction == 2:  # Left
                next_dirs = (1, 2, 3)
                new_x -= 1
            if direction == 3:  # Up
                next_dirs = (2, 3, 4)
                new_y += 1
            if direction == 4:  # Right
                next_dirs = (1, 3, 4)
                new_x += 1

            if (new_x, new_y) in self.hemispheres: # To avoid infinite loops
                break
            else:
                self.hemispheres.append((new_x, new_y))

    def iterate_position(self):
        if self.mass > m_static:
            self.path = [(self.x, self.y)]
            acceleration = (self.mass * gravity - friction_constant_force)/self.mass
            self.velocity = self.velocity + acceleration * time_step
            for i in range(math.ceil(self.velocity * scale_factor * width)):
                self.x += choose_direction(self)
                self.y += 1
                self.t_i += 1
                self.path.append((self.x, self.y))

            self.hemispheres = [(self.x, self.y)]

    def intersects(self, drop):
        if drop.get_lowest_y() < self.get_highest_y() or drop.get_highest_y() > self.get_lowest_y():
            return False

        for self_x, self_y in self.hemispheres:
            for drop_x, drop_y in drop.hemispheres:
                delta_x = self_x - drop_x
                delta_y = self_y - drop_y
                if math.sqrt(delta_x**2 + delta_y**2) < drop.radius + self.radius + attraction_radius:
                    return True
        return False

    def residual_probability(self):
        if self.velocity > 0:
            return min(1, beta * time_step / t_max * min(1, self.t_i / t_max))
        else:
            return 0

    def get_lowest_y(self):
        if self.mass < m_static or len(self.path) == 0:
            return min(self.hemispheres, key=lambda t: t[1])[1] - math.ceil(self.radius)
        else:
            return min(self.path, key=lambda t: t[1])[1] - math.ceil(self.radius)

    def get_highest_y(self):
        if self.mass < m_static or len(self.path) == 0:
            return max(self.hemispheres, key=lambda t: t[1])[1] + math.ceil(self.radius)
        else:
            return max(self.path, key=lambda t: t[1])[1] + math.ceil(self.radius)

    def get_lowest_x(self):
        if self.mass < m_static or len(self.path) == 0:
            return min(self.hemispheres, key=lambda t: t[0])[0] - math.ceil(self.radius)
        else:
            return min(self.path, key=lambda t: t[0])[0] - math.ceil(self.radius)

    def get_highest_x(self):
        if self.mass < m_static or len(self.path) == 0:
            return max(self.hemispheres, key=lambda t: t[0])[0] + math.ceil(self.radius)
        else:
            return max(self.path, key=lambda t: t[0])[0] + math.ceil(self.radius)

    def get_height(self, x, y):
        if len(self.path) != 0:
            summation = 0.0
            count = 0
            max_height = 0
            for path_x, path_y in self.hemispheres:
                delta_x = x - path_x
                delta_y = y - path_y
                rad_sqr = self.radius ** 2
                distance_from_center_sqr = delta_x ** 2 + delta_y ** 2
                if rad_sqr >= distance_from_center_sqr:
                    max_height = max(max_height, np.sqrt(rad_sqr - distance_from_center_sqr))
                    summation += np.sqrt(rad_sqr - distance_from_center_sqr)
                    count += 1
            return max_height
            if count != 0:
                return summation / count
            return summation

        elif self.mass < m_static:
            summation = 0.0
            count = 0
            for hemi_x, hemi_y in self.hemispheres:
                delta_x = x - hemi_x
                delta_y = y - hemi_y
                rad_sqr = self.radius ** 2
                distance_from_center_sqr = delta_x ** 2 + delta_y ** 2
                if rad_sqr >= distance_from_center_sqr:
                    summation += np.sqrt(rad_sqr - distance_from_center_sqr)
                    count += 1
            if count != 0:
                return summation / count
            return summation

        else:
            return np.sqrt(self.radius ** 2 - (y - self.y) ** 2 - (x - self.x) ** 2)

    def delete(self):
        if self in drop_array:
            drop_array.remove(self)
        elif self in active_drops:
            active_drops.remove(self)
        elif self in new_drops:
            new_drops.remove(self)


def choose_direction(droplet):
    # Chooses three regions in front of the drop, calculates total water volume
    # and returns region with highest water volume, or random if equal quantities of
    # water present
    radius = math.floor(droplet.radius)
    start_y = droplet.y + radius
    end_y = start_y + radius * 2
    x_1 = droplet.x - radius * 3
    x_2 = droplet.x - radius
    x_3 = droplet.x + radius
    x_4 = droplet.x + 3 * radius

    sum1 = 0

    for x in range(x_1, x_2+1):
        for y in range(start_y, end_y+1):
            if 0 <= x < width and 0 <= y < height:
                sum1 += height_map[x, y]

    sum2 = 0

    for x in range(x_2, x_3+1):
        for y in range(start_y, end_y+1):
            if 0 <= x < width and 0 <= y < height:
                sum2 += height_map[x, y]

    sum3 = 0

    for x in range(x_3, x_4+1):
        for y in range(start_y, end_y+1):
            if 0 <= x < width and 0 <= y < height:
                sum3 += height_map[x, y]

    if sum1 > sum2 >= sum3:
        return -1
    if sum2 > sum1 >= sum3:
        return 0
    if sum3 > sum1 >= sum2:
        return 1
    else:
        return random.randint(-1, 1)


# Adds avg drops, placed randomly on the height map, with randomly generated
# masses bounded by m_min and m_max
def add_drops(avg):
    for x in range(avg):
        if args.dist == "normal":
            mass = min(m_max, max(m_min, np.random.normal(m_avg, m_dev, 1)))
        elif args.dist == "uniform":
            mass = np.random.uniform(m_min, m_max)

        drop_to_add = Droplet(random.randint(0, width), random.randint(0, height), mass)
        if mass > m_static:
            active_drops.append(drop_to_add)
        else:
            new_drops.append(drop_to_add)


# Iterates over all active drops (which are moving faster than a given speed) to update their position
def iterate_over_drops():
    for drop in active_drops:
        drop.iterate_position()


# Goes over all active drops, and has them leave
def leave_residual_droplets():
    for drop in active_drops:
        if drop.mass > m_static:
            if drop.residual_probability() < np.random.uniform():
                drop.t_i = 0
                a = np.random.uniform(residual_floor, residual_ceil) # Pass in as command line args
                new_drop_mass = min(m_static, a*drop.mass)
                drop.mass -= new_drop_mass
                drop.calculate_radius()

                if drop.mass < m_static:
                    drop.delete()
                    new_drops.append(drop)
                new_drops.append(Droplet(drop.x, drop.y, new_drop_mass, 0))


def compute_height_map():
    smooth_height_map()
    floor_water()


def update_height_map():
    for drop in itertools.chain(active_drops, new_drops):
        for y in range(drop.get_lowest_y(), drop.get_highest_y() + 1):
            for x in range(drop.get_lowest_x(), drop.get_highest_x() + 1):
                if (0 <= y < height) and (0 <= x < width):
                    new_height = drop.get_height(x, y)
                    if height_map[x][y] < new_height:
                        height_map[x][y] = new_height


def smooth_height_map():
    global height_map
    height_map = ndimage.convolve(height_map, kernel, mode='constant', cval=0)


def floor_water():
    height_map[height_map < floor_value] = 0.0


def detect_intersections():
    detections = []
    temp_array = []
    temp_array.extend(active_drops)
    temp_array.extend(new_drops)

    # All intersections of active and new drops with each other
    for a in range(len(temp_array)):
        for b in range(a+1, len(temp_array)):
            if temp_array[a].intersects(temp_array[b]):
                detections.append((drop_array[a], drop_array[b]))

    for drop_a in temp_array:
        for drop_b in drop_array:
            if drop_a.intersects(drop_b):
                detections.append((drop_a, drop_b))

    return detections


def merge_drops():
    intersecting_drops = detect_intersections()

    for a, b in intersecting_drops:
        new_velocity = (a.velocity * a.mass + b.velocity * b.mass) / (a.mass + b.mass)

        if a.y < b.y:
            low_drop = a
            high_drop = b
        else:
            low_drop = b
            high_drop = b

        low_drop.velocity = new_velocity
        low_drop.mass += b.mass
        low_drop.calculate_radius()

        low_drop.delete()
        high_drop.delete()

        if low_drop.mass <= m_static and hemispheres_enabled:
            low_drop.generate_hemispheres()
            new_drops.append(low_drop)
        elif a.mass > m_static:
            active_drops.append(low_drop)


def trim_drops():
    drops_to_remove = []
    for drop in active_drops:
        radius = drop.radius
        x = drop.x
        y = drop.y
        if x + radius < 0 or x - radius > width or y + radius < 0 or y - radius > height:
            drops_to_remove.append(drop)

    for drop in drops_to_remove:
        drop.delete()


def compose_string(start_time, curr_step, drop_array, active_drops, args):
    import time
    verbose = args.verbose
    show_time = "t" in verbose
    show_drop_count = "d" in verbose
    show_average_drop_mass = "a" in verbose

    output_string = "\rStep " + str(curr_step + 1) + " out of " + str(args.steps) + " is complete."

    if show_time:
        elapsed_time = time.time() - start_time
        output_string = output_string + "\nCalculation took " + str(elapsed_time) + " seconds."
    if show_drop_count:
        output_string = output_string + "\nThere are currently " + str(len(drop_array) + len(active_drops)) + \
                        " drops in the height map, of which " + str(len(active_drops)) + " are in motion."
    if show_average_drop_mass:
        masses = sum(drop.mass for drop in itertools.chain(drop_array, active_drops))
        avg_mass = masses / (len(drop_array)+len(active_drops))
        output_string = output_string + "\nThe average mass of the drops is " + str(avg_mass) + " kg."
    return output_string


def initialize_globals(inject_args):
    global args
    args = inject_args

    global width, height
    global scale_factor
    width, height = inject_args.width, inject_args.height
    scale_factor = inject_args.scale

    global density_water, gravity
    density_water = inject_args.density_water
    gravity = inject_args.g

    global m_min, m_max, m_avg, m_dev, m_static
    m_min = inject_args.m_min
    m_max = inject_args.m_max
    m_avg = inject_args.m_avg
    m_dev = inject_args.m_dev

    if args.dist == "normal":
        m_static = m_avg + m_dev * st.norm.ppf(inject_args.m_static)
    elif args.dist == "uniform":
        m_static = m_max * inject_args.m_static

    global friction_constant_force
    friction_constant_force = m_static * gravity

    global hemispheres_enabled, max_hemispheres
    hemispheres_enabled = inject_args.enable_hemispheres
    max_hemispheres = inject_args.max_hemispheres

    global attraction_radius
    attraction_radius = inject_args.attraction

    global time_step, t_max, beta, floor_value, residual_floor, residual_ceil
    time_step = inject_args.time
    t_max = time_step * 4
    beta = args.beta
    floor_value = inject_args.floor_value
    residual_floor, residual_ceil = inject_args.residual_floor, inject_args.residual_ceil

    global kernel
    if inject_args.kernel == "dwn":
        kernel = np.array([[4 / 27, 1 / 9, 2 / 27],
                           [4 / 27, 1 / 9, 2 / 27],
                           [4 / 27, 1 / 9, 2 / 27]])
    else:
        kernel = np.array([[1 / 9, 1 / 9, 1 / 9],
                           [1 / 9, 1 / 9, 1 / 9],
                           [1 / 9, 1 / 9, 1 / 9]])


def main(args, multiprocessing=False, curr_run=1):
    from src import file_ops as fo
    import time

    global active_drops, drop_array, new_drops
    global height_map

    initialize_globals(args)
    print(multiprocessing)
    if multiprocessing:
        runs = 1
    else:
        runs = args.runs

    for i in range(runs):
        drop_array = []
        active_drops = []

        height_map = np.zeros(shape=(width, height))

        for j in range(int(args.steps)):
            new_drops = []

            start_time = time.time()

            add_drops(int(args.drops))  # Very fast
            iterate_over_drops()  # Very fast

            if bool(args.leave_residuals):
                leave_residual_droplets()  # Very fast
            merge_drops()  # Takes around 1/3 of the processing time
            trim_drops()  # Very fast
            update_height_map()
            compute_height_map()
            drop_array.extend(new_drops)
            new_drops = []

            if not multiprocessing:
                print(compose_string(start_time, j, drop_array, active_drops, args))

        new_drops = drop_array
        update_height_map()
        compute_height_map()

        if multiprocessing:
            fo.save(fo.choose_file_name(args, curr_run), height_map, args)
            print("\rRun " + str(curr_run + 1) + " out of " + str(args.runs) + " is complete.")
        else:
            fo.save(fo.choose_file_name(args, i), height_map, args)
            if runs > 1:
                print("\rRun " + str(i + 1) + " out of " + str(i) + " is complete.")
