import math
import scipy.stats as st
import random
import itertools
from scipy import ndimage
import numpy as np
import time

from math import sqrt


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
            self.m_static = self.m_avg + self.m_dev * st.norm.ppf(args.p_static)
        elif args.dist == "uniform":
            self.m_static = self.m_max * args.p_static

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

        self.passive_drops = []
        self.active_drops = []
        self.residual_drops = []
        self.new_drops = []
        self.height_map = np.zeros(shape=(self.width, self.height))
        self.id_map = np.zeros(shape=(self.width, self.height))
        self.start_time = None
        self.multiprocessing = multiprocessing
        self.curr_run = curr_run
        self.avg = args.drops
        self.max_id = 0
        self.steps_so_far = 0
        self.drop_dict = {}
        self.trail_map = np.zeros(shape=(self.width, self.height), dtype=bool)
        self.color_dict = {0: (255, 255, 255)}
        self.affinity_map = np.reshape(np.random.uniform(size=self.width*self.height), (self.width,self.height))
        self.static_drop_map = np.zeros(shape=(self.width, self.height))


    class Droplet:
        def __init__(self, x, y, mass, drop_id, super, velocity=0, parent_id=None):
            self.x = x
            self.y = y
            self.mass = 0
            self.velocity = velocity
            self.direction = 0
            self.t_i = 0
            self.path = []
            self.super = super
            self.hemispheres = [(self.x, self.y)]  # (x,y) tuples (could add z to represent share of mass)
            self.radius = 0
            self.rad_sqr = 0
            self.rad_sqr_extended = 0
            self.delta = 0
            self.lowest_x = 0
            self.lowest_y = 0
            self.highest_x = 0
            self.highest_y = 0
            self.id = drop_id
            self.super.drop_dict[drop_id] = self
            self.update_mass(mass)
            self.parent_id = parent_id

        def update_mass(self, mass):
            self.super.delete(self)
            self.mass = mass
            if mass > self.super.m_static:
                self.hemispheres = [(self.x, self.y)]
                self.path = []
                self.super.active_drops.append(self)
            else:
                self.super.new_drops.append(self)
                self.generate_hemispheres()
            self.calculate_radius()

        def calculate_radius(self):
            self.radius = np.cbrt((3 / 2) / math.pi * (self.mass / len(self.hemispheres)) / self.super.density_water) * self.super.scale_factor * self.super.width
            self.rad_sqr = self.radius ** 2
            self.rad_sqr_extended = (self.radius + self.super.attraction_radius) ** 2
            self.delta = -2 * self.super.attraction_radius - self.super.attraction_radius ** 2

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
            self.compute_bounding_box()

        def iterate_position(self):
            self.lowest_y = self.y - math.ceil(self.radius)
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
                self.highest_y = self.y + math.ceil(self.radius)

            self.compute_bounding_box(no_y=True)

        def compute_bounding_box(self, no_y=False):
            if self.mass < self.super.m_static or len(self.path) == 0:
                self.lowest_y = min(self.hemispheres, key=lambda t: t[1])[1] - math.ceil(self.radius)
                self.highest_y = max(self.hemispheres, key=lambda t: t[1])[1] + math.ceil(self.radius)
                self.lowest_x = min(self.hemispheres, key=lambda t: t[0])[0] - math.ceil(self.radius)
                self.highest_x = max(self.hemispheres, key=lambda t: t[0])[0] + math.ceil(self.radius)
            else:
                if not no_y:
                    self.lowest_y = min(self.path, key=lambda t: t[1])[1] - math.ceil(self.radius)
                    self.highest_y = max(self.path, key=lambda t: t[1])[1] + math.ceil(self.radius)
                self.lowest_x = min(self.path, key=lambda t: t[0])[0] - math.ceil(self.radius)
                self.highest_x = max(self.path, key=lambda t: t[0])[0] + math.ceil(self.radius)

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

        def get_height_arr(self):
            internal_height_map = np.zeros(shape=(self.highest_x-self.lowest_x, self.highest_y-self.lowest_y))
            internal_height_map_counts = np.zeros(shape=(self.highest_x-self.lowest_x, self.highest_y-self.lowest_y))

            if len(self.path) != 0:
                for x, y in self.path:
                    dists_from_center_sqr = np.array([[self.rad_sqr - (x-new_x)**2 + (y-new_y)**2
                                                      for new_y in range(self.lowest_x,self.highest_x)]
                                                     for new_x in range(self.lowest_y,self.highest_y)])\
                        .reshape((self.highest_x-self.lowest_x, self.highest_y-self.lowest_y))

                    dists_from_center_sqr[dists_from_center_sqr < 0] = 0
                    internal_height_map = np.add(internal_height_map, np.sqrt(dists_from_center_sqr))
                    internal_height_map_counts[dists_from_center_sqr > 0] += 1
                internal_height_map_counts[dists_from_center_sqr == 0] = 1
                return np.divide(internal_height_map,internal_height_map_counts)

            else:
                for x, y in self.hemispheres:
                    dists_from_center_sqr = np.array([[self.rad_sqr - (x-new_x)**2 + (y-new_y)**2
                                                      for new_y in range(self.lowest_x,self.highest_x)]
                                                     for new_x in range(self.lowest_y,self.highest_y)])
                    print(np.shape(dists_from_center_sqr))
                    dists_from_center_sqr[dists_from_center_sqr < 0] = 0
                    np.add(internal_height_map, np.sqrt(dists_from_center_sqr))
                    internal_height_map_counts[dists_from_center_sqr > 0] += 1
                internal_height_map_counts[dists_from_center_sqr == 0] = 1
                return np.divide(internal_height_map,internal_height_map_counts)

        def get_height(self, x, y):
            if len(self.path) != 0:
                distance_from_center_sqr = [self.rad_sqr - (x - path_x) ** 2 + (y - path_y) ** 2 for path_x, path_y in self.path]
                distance_from_center_sqr_pos = [sqrt(x) for x in distance_from_center_sqr if x > 0]
                length = len(distance_from_center_sqr_pos)
                if length > 0:
                    return sum(distance_from_center_sqr_pos) / length
                else:
                    return 0

            else:
                distance_from_center_sqr = [self.rad_sqr - (x - hemi_x) ** 2 + (y - hemi_y) ** 2 for hemi_x, hemi_y in self.hemispheres]
                distance_from_center_sqr_pos = [sqrt(x) for x in distance_from_center_sqr if x > 0]
                length = len(distance_from_center_sqr_pos)
                if length > 0:
                    return sum(distance_from_center_sqr_pos) / length

        def get_height_and_id(self, x, y): # <- This guy sucks down all the processing time.
            if self.lowest_x <= x <= self.highest_x and self.lowest_y <= y <= self.highest_y:
                flag = False
                if len(self.path) != 0:
                    ## EVIL EVIL PERFORMANCE HOG
                    distance_from_center_sqr = [(x-path_x)*(x-path_x) + (y-path_y)*(y-path_y) for path_x, path_y in
                                                self.path if y - self.radius <= path_y <= y + self.radius]
                    #filtered_dist = [dist for dist in distance_from_center_sqr if dist > self.delta]
                    summation = 0
                    count = 0
                    for delta_distance in distance_from_center_sqr:
                        if not flag:
                            if self.rad_sqr_extended >= delta_distance:
                                flag = True
                        if self.rad_sqr >= delta_distance:
                            summation += sqrt(self.rad_sqr - delta_distance)
                            count += 1
                    if count > 0:
                        return summation / count, True
                    return 0, flag

                else: # <- Never gets touched
                    distance_from_center_sqr = [(x-hemi_x)**2 + (y-hemi_y)**2 for hemi_x, hemi_y in self.path]
                    summation = 0
                    count = 0
                    for delta_distance in distance_from_center_sqr:
                        if not flag:
                            if self.rad_sqr_extended >= delta_distance:
                                flag = True
                        if self.rad_sqr >= delta_distance:
                            summation += sqrt(self.rad_sqr - delta_distance)
                            count += 1
                    if count != 0:
                        return summation / count, True
                    return 0, flag
            else:
                return 0, self.get_id_at_pos(x,y)

        def get_id_at_pos(self, x, y):
            if len(self.path) != 0:
                return any((x - path_x) * (x - path_x) + (y - path_y) * (y - path_y) < self.rad_sqr_extended
                           for path_x, path_y in self.path)
            else:
                return any((x - hemi_x) * (x - hemi_x) + (y - hemi_y) * (y - hemi_y) < self.rad_sqr_extended
                           for hemi_x, hemi_y in self.hemispheres)

    def delete(self, droplet):
        if not isinstance(droplet, self.Droplet):
            raise TypeError()
        if droplet in self.passive_drops:
            self.passive_drops.remove(droplet)
        if droplet in self.active_drops:
            self.active_drops.remove(droplet)
        if droplet in self.new_drops:
            self.new_drops.remove(droplet)
        if droplet in self.residual_drops:
            self.residual_drops.remove(droplet)

    def set_ids(self, old_id, new_id, delete=False):
        if delete:
            self.trail_map[self.id_map == old_id] = True
        if old_id == 0:
            raise Exception("old_id should be a non zero value")
        self.id_map[self.id_map == old_id] = new_id

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

        sum1 = np.sum(self.height_map[x_1:x_2+1, start_y:end_y + 1])
        sum2 = np.sum(self.height_map[x_2:x_3+1, start_y:end_y + 1])
        sum3 = np.sum(self.height_map[x_3:x_4 + 1, start_y:end_y + 1])

        if sum1 > sum2 >= sum3:
            return -1
        if sum2 > sum1 >= sum3:
            return 0
        if sum3 > sum1 >= sum2:
            return 1
        else:
            sum1 = np.sum(self.affinity_map[x_1:x_2 + 1, start_y:end_y + 1])
            sum2 = np.sum(self.affinity_map[x_2:x_3 + 1, start_y:end_y + 1])
            sum3 = np.sum(self.affinity_map[x_3:x_4 + 1, start_y:end_y + 1])

            if sum1 > sum2 >= sum3 or sum1 > sum3 >= sum2:
                return -1
            if sum2 > sum1 >= sum3 or sum2 > sum3 >= sum1:
                return 0
            if sum3 > sum1 >= sum2 or sum3 > sum2 >= sum1:
                return 1
            else:
                return 0

    # Adds avg drops, placed randomly on the height map, with randomly generated
    # masses bounded by m_min and m_max
    def add_drops(self):
        for x in range(self.avg):
            if self.args.dist == "normal":
                mass = min(self.m_max, max(self.m_min, np.random.normal(self.m_avg, self.m_dev, 1)))
            elif self.args.dist == "uniform":
                mass = np.random.uniform(self.m_min, self.m_max)

            self.max_id += 1
            self.Droplet(random.randint(0, self.width), random.randint(0, self.height), mass, self.max_id, self)

    # Iterates over all active drops (which are moving faster than a given speed) to update their position
    def iterate_over_drops(self):
        for drop in self.active_drops:
            drop.iterate_position()

    # Goes over all active drops, and has them leave
    def leave_residual_droplets(self):
        for drop in self.active_drops:
            if drop.mass > self.m_static:
                if drop.residual_probability() > np.random.uniform():
                    drop.t_i = 0
                    a = np.random.uniform(self.residual_floor, self.residual_ceil) # Pass in as command line args
                    new_drop_mass = min(self.m_static, a*drop.mass)
                    drop.update_mass(drop.mass - new_drop_mass)
                    self.max_id += 1
                    self.residual_drops.append(self.Droplet(drop.x, drop.y, new_drop_mass, self.max_id, self, parent_id=drop.id))

    def compute_height_map(self):
        self.smooth_height_map()
        self.floor_water()

    def update_maps(self):
        collisions = []
        for drop in itertools.chain(self.active_drops, self.new_drops):
            for y in range(drop.lowest_y - self.args.attraction, drop.highest_y + self.args.attraction):
                for x in range(drop.lowest_x - self.args.attraction, drop.highest_x + self.args.attraction):
                    if (0 <= y < self.height) and (0 <= x < self.width):
                        new_height, flag = drop.get_height_and_id(x, y)
                        if self.height_map[x, y] < new_height:
                            self.height_map[x, y] = new_height

                        if flag:
                            curr_id = self.id_map[x, y]
                            if curr_id != drop.parent_id and curr_id != 0:
                                collisions.append((drop.id, curr_id))
                            self.id_map[x, y] = drop.id
                            self.trail_map[x, y] = True
        return collisions

    def update_height_map(self):
        for drop in itertools.chain(self.active_drops, self.new_drops):
            for y in range(drop.lowest_y, drop.highest_y):
                for x in range(drop.lowest_x, drop.highest_x):
                    if (0 <= y < self.height) and (0 <= x < self.width):
                        new_height = drop.get_height(x, y)
                        if self.height_map[x, y] < new_height:
                            self.height_map[x, y] = new_height
            drop.path = []

    def update_height_map_arrs(self):
        for drop in itertools.chain(self.active_drops, self.new_drops):
            self.height_map[drop.lowest_x:drop.highest_x, drop.lowest_y:drop.highest_y] = np.maximum(
                self.height_map[drop.lowest_x:drop.highest_x, drop.lowest_y:drop.highest_y], drop.get_height_arr())

    def update_id_map(self):
        collisions = []
        for drop in self.active_drops:
            for y in range(drop.lowest_y - self.args.attraction, drop.highest_y + self.args.attraction):
                for x in range(drop.lowest_x - self.args.attraction, drop.highest_x + self.args.attraction):
                    if (0 <= y < self.height) and (0 <= x < self.width):
                        if drop.get_id_at_pos(x, y):
                            curr_id = self.id_map[x, y]
                            if curr_id != drop.parent_id and curr_id != 0:
                                collisions.append((drop.id,curr_id))
                            self.id_map[x, y] = drop.id
                            self.trail_map[x, y] = True

        for drop in self.new_drops:
            for y in range(drop.lowest_y - self.args.attraction, drop.highest_y + self.args.attraction):
                for x in range(drop.lowest_x - self.args.attraction, drop.highest_x + self.args.attraction):
                    if (0 <= y < self.height) and (0 <= x < self.width):
                        if drop.get_id_at_pos(x, y):
                            curr_id = self.id_map[x, y]
                            if curr_id != drop.parent_id and curr_id != 0:
                                collisions.append((drop.id,curr_id))
                            self.id_map[x, y] = drop.id

        return collisions

    def smooth_height_map(self):
        self.height_map[self.trail_map==True] = ndimage.convolve(self.height_map, self.kernel, mode='constant', cval=0)[self.trail_map==True]

    def floor_water(self):
        self.height_map[self.height_map < self.floor_value] = 0.0
        self.id_map[self.height_map == 0] = 0
        self.trail_map[self.id_map == 0] = 0

    # Merges and deletes drops that have merged
    def merge_drops(self):
        #intersecting_drops = self.update_id_map()
        intersecting_drops = self.update_maps()

        for a, b in intersecting_drops:
            if a in self.drop_dict.keys() and b in self.drop_dict.keys():
                a = self.drop_dict[a]
                b = self.drop_dict[b]
                new_velocity = (a.velocity * a.mass + b.velocity * b.mass) / (a.mass + b.mass)

                if a.y < b.y:
                    low_drop = a
                    high_drop = b
                else:
                    low_drop = b
                    high_drop = a

                low_drop.velocity = new_velocity
                low_drop.update_mass(low_drop.mass + high_drop.mass)
                low_drop.calculate_radius()

                self.set_ids(low_drop.id, high_drop.id, delete=True)
                self.delete(high_drop)
                self.drop_dict.pop(high_drop.id)

    # Deletes drops that are out of bounds
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
            self.set_ids(drop.id, 0)
            self.drop_dict.pop(drop.id)

    def compose_string(self):
        import time
        verbose = self.args.verbose
        show_time = "t" in verbose
        show_drop_count = "d" in verbose
        show_average_drop_mass = "a" in verbose

        output_string = "\rStep " + str(self.steps_so_far) + " out of " + str(self.args.steps) + " is complete."

        if show_time:
            elapsed_time = time.time() - self.start_time
            output_string = output_string + "\nCalculation took " + str(elapsed_time) + " seconds."
        if show_drop_count:
            output_string = output_string + "\nThere are currently " + str(len(self.passive_drops) + len(self.active_drops)) + \
                            " drops in the height map, of which " + str(len(self.active_drops)) + " are in motion."
        if show_average_drop_mass:
            masses = 0.0
            for drop in self.passive_drops:
                masses += drop.mass
            avg_mass = masses / len(self.passive_drops)
            output_string = output_string + "\nThe average mass of the drops is " + str(avg_mass) + " kg."
        return output_string

    def clear_passives(self):
        to_pop = []
        for drop in self.passive_drops:
            if drop.id not in self.id_map:
                self.delete(drop)
                to_pop.append(drop.id)

        for drop_id in to_pop:
            self.drop_dict.pop(drop_id)

    def step(self):
        self.new_drops = []
        self.new_drops.extend(self.residual_drops)
        self.start_time = time.time()

        self.add_drops()  # Very fast
        self.iterate_over_drops()  # Very fast

        if bool(self.args.leave_residuals):
            self.leave_residual_droplets()  # Very fast
        self.merge_drops()  # Takes around 1/3 of the processing time
        self.trim_drops()  # Very fast
        #self.update_height_map_arrs()
        self.compute_height_map()
        self.passive_drops.extend(self.new_drops)
        self.steps_so_far += 1

        #self.clear_passives()
        return self.compose_string()

    def save(self):
        from src import file_ops as fo
        if self.multiprocessing:
            fo.save(fo.choose_file_name(self.args, self.curr_run), self.height_map, self.id_map, self.args)
            print("\rRun " + str(self.curr_run +1) + " out of " + str(self.args.runs) + " is complete.")
        else:
            fo.save(fo.choose_file_name(self.args, self.curr_run), self.height_map, self.id_map, self.args)
            if self.args.runs > 1:
                print("\rRun " + str(self.curr_run + 1) + " out of " + str(self.args.runs) + " is complete.")

    def save_temp(self):
        from src import file_ops as fo
        fo.save_temp(self.height_map, self.id_map, self.color_dict, self.args, self.steps_so_far)

    def add_old_drops(self):
        self.new_drops = self.passive_drops
        self.update_height_map()
        self.compute_height_map()

    def blur_masked(self):
        height_map_copy = np.array(self.height_map, copy=True)
        self.smooth_height_map()
        self.height_map[height_map_copy == 0] = 0
        self.smooth_height_map()
        self.height_map[height_map_copy == 0] = 0
        self.smooth_height_map()
        self.height_map[height_map_copy == 0] = 0
