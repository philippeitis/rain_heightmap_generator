import math
import scipy.stats as st
import random
import itertools
from scipy import ndimage
import numpy as np
import time


class Surface:
    def __init__(self, args, multiprocessing=False, curr_run=1):
        self.args = args

        self.width, self.height = args.width, args.height
        self.scale_factor = args.scale

        self.density_water = args.density_water
        self.gravity = args.g

        self.m_min = args.m_min
        self.m_max = args.m_max
        self.m_avg = args.m_avg
        self.m_dev = args.m_dev

        if args.dist == "normal":
            self.m_static = self.m_avg + self.m_dev * st.norm.ppf(args.m_static)
        elif args.dist == "uniform":
            self.m_static = self.m_max * args.m_static

        self.friction_constant_force = self.m_static * self.gravity

        self.hemispheres_enabled = args.enable_hemispheres
        self.max_hemispheres = args.max_hemispheres

        self.attraction_radius = args.attraction

        self.time_step = args.time
        self.t_max = self.time_step * 4
        self.beta = args.beta
        self.floor_value = args.floor_value
        self.residual_floor, self.residual_ceil = args.residual_floor, args.residual_ceil
        self.floor_value = args.floor_value

        if args.kernel == "dwn":
            self.kernel = np.array([[4 / 27, 1 / 9, 2 / 27],
                               [4 / 27, 1 / 9, 2 / 27],
                               [4 / 27, 1 / 9, 2 / 27]])
        else:
            self.kernel = np.array([[1 / 9, 1 / 9, 1 / 9],
                               [1 / 9, 1 / 9, 1 / 9],
                               [1 / 9, 1 / 9, 1 / 9]])

        self.drop_array = []
        self.active_drops = []
        self.new_drops = []
        self.height_map = np.zeros(shape=(self.width, self.height))
        self.id_map = np.zeros(shape=(self.width, self.height))
        self.start_time = None
        self.multiprocessing = multiprocessing
        self.curr_run = curr_run
        self.avg = args.drops
        self.max_id = 0
        self.steps_so_far = 0

    class Droplet:
        def __init__(self, x, y, mass, drop_id, super, velocity=0):
            self.x = x
            self.y = y
            self.mass = mass
            self.velocity = velocity
            self.direction = 0
            self.t_i = 0
            self.path = []
            self.super = super
            self.hemispheres = [(self.x, self.y)]  # (x,y) tuples (could add z to represent share of mass)
            if mass < super.m_static and super.hemispheres_enabled:
                self.generate_hemispheres()
            self.id = drop_id
            self.radius = 0
            self.calculate_radius()

        def calculate_radius(self):
            self.radius = np.cbrt((3 / 2) / math.pi * (self.mass / len(self.hemispheres)) / self.super.density_water) * self.super.scale_factor * self.super.width

        def generate_hemispheres(self):
            # Random walk to decide locations of hemispheres.
            self.hemispheres = [(self.x, self.y)]
            num_hemispheres = random.randint(1, self.super.max_hemispheres)
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
            if self.mass > self.super.m_static:
                self.path = [(self.x, self.y)]
                acceleration = (self.mass * self.super.gravity - self.super.friction_constant_force)/self.mass
                self.velocity = self.velocity + acceleration * self.super.time_step
                for i in range(math.ceil(self.velocity * self.super.scale_factor * self.super.width)):
                    self.x += self.super.choose_direction(self)
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
                    if math.sqrt(delta_x**2 + delta_y**2) < drop.radius + self.radius + self.super.attraction_radius:
                        return True
            return False

        def residual_probability(self):
            if self.velocity > 0:
                return min(1, self.super.beta * self.super.time_step / self.super.t_max * min(1, self.t_i / self.super.t_max))
            else:
                return 0

        def get_lowest_y(self):
            if self.mass < self.super.m_static or len(self.path) == 0:
                return min(self.hemispheres, key=lambda t: t[1])[1] - math.ceil(self.radius)
            else:
                return min(self.path, key=lambda t: t[1])[1] - math.ceil(self.radius)

        def get_highest_y(self):
            if self.mass < self.super.m_static or len(self.path) == 0:
                return max(self.hemispheres, key=lambda t: t[1])[1] + math.ceil(self.radius)
            else:
                return max(self.path, key=lambda t: t[1])[1] + math.ceil(self.radius)

        def get_lowest_x(self):
            if self.mass < self.super.m_static or len(self.path) == 0:
                return min(self.hemispheres, key=lambda t: t[0])[0] - math.ceil(self.radius)
            else:
                return min(self.path, key=lambda t: t[0])[0] - math.ceil(self.radius)

        def get_highest_x(self):
            if self.mass < self.super.m_static or len(self.path) == 0:
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

            elif self.mass < self.super.m_static:
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

    def delete(self, droplet):
        if droplet in self.drop_array:
            self.drop_array.remove(droplet)
        elif droplet in self.active_drops:
            self.active_drops.remove(droplet)
        elif droplet in self.new_drops:
            self.new_drops.remove(droplet)

    def choose_direction(self, droplet):
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
                if 0 <= x < self.width and 0 <= y < self.height:
                    sum1 += self.height_map[x, y]

        sum2 = 0

        for x in range(x_2, x_3+1):
            for y in range(start_y, end_y+1):
                if 0 <= x < self.width and 0 <= y < self.height:
                    sum2 += self.height_map[x, y]

        sum3 = 0

        for x in range(x_3, x_4+1):
            for y in range(start_y, end_y+1):
                if 0 <= x < self.width and 0 <= y < self.height:
                    sum3 += self.height_map[x, y]

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
    def add_drops(self):
        for x in range(self.avg):
            if self.args.dist == "normal":
                mass = min(self.m_max, max(self.m_min, np.random.normal(self.m_avg, self.m_dev, 1)))
            elif self.args.dist == "uniform":
                mass = np.random.uniform(self.m_min, self.m_max)

            self.max_id += 1
            drop_to_add = self.Droplet(random.randint(0, self.width), random.randint(0, self.height), mass, self.max_id, self)
            if mass > self.m_static:
                self.active_drops.append(drop_to_add)
            else:
                self.new_drops.append(drop_to_add)


    # Iterates over all active drops (which are moving faster than a given speed) to update their position
    def iterate_over_drops(self):
        for drop in self.active_drops:
            drop.iterate_position()


    # Goes over all active drops, and has them leave
    def leave_residual_droplets(self):
        for drop in self.active_drops:
            if drop.mass > self.m_static:
                if drop.residual_probability() < np.random.uniform():
                    drop.t_i = 0
                    a = np.random.uniform(self.residual_floor, self.residual_ceil) # Pass in as command line args
                    new_drop_mass = min(self.m_static, a*drop.mass)
                    drop.mass -= new_drop_mass
                    drop.calculate_radius()

                    if drop.mass < self.m_static:
                        self.delete(drop)
                        self.new_drops.append(drop)
                    self.max_id += 1
                    self.new_drops.append(self.Droplet(drop.x, drop.y, new_drop_mass, self.max_id, self))

    def compute_height_map(self):
        self.smooth_height_map()
        self.floor_water()

    def update_height_map(self):
        for drop in itertools.chain(self.active_drops, self.new_drops):
            for y in range(drop.get_lowest_y(), drop.get_highest_y() + 1):
                for x in range(drop.get_lowest_x(), drop.get_highest_x() + 1):
                    if (0 <= y < self.height) and (0 <= x < self.width):
                        new_height = drop.get_height(x, y)
                        if self.height_map[x][y] < new_height:
                            self.height_map[x][y] = new_height

    def smooth_height_map(self):
        self.height_map = ndimage.convolve(self.height_map, self.kernel, mode='constant', cval=0)

    def floor_water(self):
        self.height_map[self.height_map < self.floor_value] = 0.0
        self.id_map[self.height_map == 0] = 0

    def detect_intersections(self):
        detections = []
        temp_array = []
        temp_array.extend(self.active_drops)
        temp_array.extend(self.new_drops)

        # All intersections of active and new drops with each other
        for a in range(len(temp_array)):
            for b in range(a+1, len(temp_array)):
                if temp_array[a].intersects(temp_array[b]):
                    detections.append((self.drop_array[a], self.drop_array[b]))

        for drop_a in temp_array:
            for drop_b in self.drop_array:
                if drop_a.intersects(drop_b):
                    detections.append((drop_a,drop_b))

        return detections

    def merge_drops(self):
        intersecting_drops = self.detect_intersections()

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

            self.delete(low_drop)
            self.delete(high_drop)

            if low_drop.mass <= self.m_static and self.hemispheres_enabled:
                low_drop.generate_hemispheres()
                self.new_drops.append(low_drop)
            elif a.mass > self.m_static:
                self.active_drops.append(low_drop)

    def trim_drops(self):
        drops_to_remove = []
        for drop in self.active_drops:
            radius = drop.radius
            x = drop.x
            y = drop.y
            if x + radius < 0 or x - radius > self.width or y + radius < 0 or y - radius > self.height:
                drops_to_remove.append(drop)

        for drop in drops_to_remove:
            self.delete(drop)

    def compose_string(self):
        import time
        verbose = self.args.verbose
        show_time = "t" in verbose
        show_drop_count = "d" in verbose
        show_average_drop_mass = "a" in verbose

        output_string = "\rStep " + str(self.steps_so_far + 1) + " out of " + str(self.args.steps) + " is complete."

        if show_time:
            elapsed_time = time.time() - self.start_time
            output_string = output_string + "\nCalculation took " + str(elapsed_time) + " seconds."
        if show_drop_count:
            output_string = output_string + "\nThere are currently " + str(len(self.drop_array) + len(self.active_drops)) + \
                            " drops in the height map, of which " + str(len(self.active_drops)) + " are in motion."
        if show_average_drop_mass:
            masses = 0.0
            for drop in self.drop_array:
                masses += drop.mass
            avg_mass = masses / len(self.drop_array)
            output_string = output_string + "\nThe average mass of the drops is " + str(avg_mass) + " kg."
        return output_string

    def step(self):
        self.new_drops = []

        self.start_time = time.time()

        self.add_drops()  # Very fast
        self.iterate_over_drops()  # Very fast

        if bool(self.args.leave_residuals):
            self.leave_residual_droplets()  # Very fast
        self.merge_drops()  # Takes around 1/3 of the processing time
        self.trim_drops()  # Very fast
        self.update_height_map()
        self.compute_height_map()
        self.drop_array.extend(self.new_drops)
        self.steps_so_far += 1

        return self.compose_string()
        new_drops = []

    def save(self):
        from src import file_ops as fo
        if self.multiprocessing:
            fo.save(fo.choose_file_name(self.args, self.curr_run), self.height_map, None, self.args)
            print("\rRun " + str(self.curr_run + 1) + " out of " + str(self.args.runs) + " is complete.")
        else:
            fo.save(fo.choose_file_name(self.args, self.curr_run), self.height_map, None, self.args)
            if self.args.runs > 1:
                print("\rRun " + str(self.curr_run + 1) + " out of " + str(self.args.runs) + " is complete.")

