import math
import scipy.stats as st
import numpy as np
import random
import itertools
from scipy import ndimage


class Droplet:
    def __init__(self, x, y, mass, velocity, height_map, active_drops, drop_array, new_drops, args = None):
        self.height_map = height_map
        self.active_drops = active_drops
        self.drop_array = drop_array
        self.new_drops = new_drops
        self.args = args

        self.x = x
        self.y = y
        self.mass = 0
        self.velocity = velocity
        self.direction = 0
        self.t_i = 0
        self.hemispheres = [(self.x, self.y)]  # (x,y) tuples (could add z to represent share of mass)
        self.radius = 0
        self.set_mass(mass)

    def set_mass(self, new_mass):
        old_mass = self.mass
        if old_mass > self.args.m_static >= new_mass:
            self.active_drops.remove(self)
            self.new_drops.append(self)
            self.generate_hemispheres()
        elif old_mass <= self.args.m_static < new_mass:
            self.active_drops.append(self)
            if self in self.new_drops:
                self.new_drops.remove(self)
            elif self in self.drop_array:
                self.drop_array.remove(self)
        else:
            if self not in self.new_drops:
                self.new_drops.append(self)
        self.mass = new_mass
        self.calculate_radius()

    def calculate_radius(self):
        # Only called when mass changes to avoid excessive calculations
        self.radius = np.cbrt((3 / 2) / math.pi * (self.mass / len(self.hemispheres))
                              / self.args.density_water) / self.args.scale * self.args.width

    def generate_hemispheres(self):
        # Random walk to decide locations of hemispheres.
        if self.mass < self.args.m_static and self.args.enable_hemispheres:
            self.hemispheres = [(self.x, self.y)]
            num_hemispheres = random.randint(1, self.args.max_hemispheres)
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
        if self.mass > self.args.m_static:
            friction_constant_force = self.args.m_static * self.args.gravity
            acceleration = (self.mass * self.args.gravity - friction_constant_force)/self.mass
            self.velocity = self.velocity + acceleration * self.args.time_step
            if self.velocity > 0:
                self.direction = choose_direction(self)
                if self.direction == -1:
                    self.x -= math.floor(math.sqrt(self.velocity) * self.args.scale * self.args.width) + random.randint(-2,2)
                    self.y += math.floor(math.sqrt(self.velocity) * self.args.scale * self.args.width) + random.randint(-2,2)
                if self.direction == 0:
                    self.y += math.floor(self.velocity * args.scale * args.width) + random.randint(-2,2)
                if self.direction == 1:
                    self.x += math.floor(math.sqrt(self.velocity) * self.args.scale * self.args.width) + random.randint(-2,2)
                    self.y += math.floor(math.sqrt(self.velocity) * self.args.scale * self.args.width) + random.randint(-2,2)

                # This determines if we should leave a droplet behind
                self.t_i += 1
            self.hemispheres = [(self.x, self.y)]

    def intersects(self, drop):
        if drop.get_lowest_y() < self.get_highest_y() or drop.get_highest_y() > self.get_lowest_y():
            return False

        for self_x, self_y in self.hemispheres:
            for drop_x, drop_y in drop.hemispheres:
                delta_x = self_x - drop_x
                delta_y = self_y - drop_y
                if math.sqrt(delta_x**2 + delta_y**2) < drop.radius + self.radius + self.args.attraction:
                    return True
        return False

    def residual_probability(self):
        if self.velocity > 0:
            return min(1, self.args.beta * self.args.time_step / self.args.t_max * min(1, self.t_i / self.args.t_max))
        else:
            return 0

    def get_lowest_y(self):
        return min(self.hemispheres, key=lambda t: t[1])[1] - math.ceil(self.radius)

    def get_highest_y(self):
        return max(self.hemispheres, key=lambda t: t[1])[1] + math.ceil(self.radius)

    def get_lowest_x(self):
        return min(self.hemispheres, key=lambda t: t[0])[0] - math.ceil(self.radius)

    def get_highest_x(self):
        return max(self.hemispheres, key=lambda t: t[0])[0] + math.ceil(self.radius)

    def get_height(self, x, y):
        if len(self.hemispheres) == 1:
            return np.sqrt(self.radius ** 2 - (y - self.y) ** 2 - (x - self.x) ** 2)

        else:
            summation = 0.0

            for hemi_x, hemi_y in self.hemispheres:
                delta_x = x - hemi_x
                delta_y = y - hemi_y
                rad_sqr = self.radius ** 2
                distance_from_center_sqr = delta_x ** 2 + delta_y ** 2
                if rad_sqr >= distance_from_center_sqr:
                    summation += np.sqrt(rad_sqr - distance_from_center_sqr)

            return summation

    def delete(self):
        if self in self.drop_array:
            self.drop_array.remove(self)
        elif self in self.active_drops:
            self.active_drops.remove(self)
        elif self in self.new_drops:
            self.new_drops.remove(self)


def choose_direction(droplet, args):
    # Chooses three regions in front of the drop, calculates total water volume
    # and returns region with highest water volume, or random if equal quantities of
    # water present
    width = args.width
    height = args.height
    height_map = droplet.height_map
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
def add_drops(height_map, active_drops, drop_array, new_drops, args):
    for x in range(args.drops):
        mass = min(args.m_max,max(args.m_min, np.random.normal(args.m_avg, args.m_dev, 1)))
        drop_to_add = Droplet(random.randint(0, args.width), random.randint(0, args.height), mass, 0,
                              height_map, active_drops, drop_array, new_drops, args)


# Iterates over all active drops (which are moving faster than a given speed) to update their position
def iterate_over_drops(active_drops, args):
    for drop in active_drops:
        old_x = drop.x
        old_y = drop.y
        # TODO: handling for fast particles (and streaks of water)
        drop.iterate_position()
        delta_x_sqr = (old_x - drop.x) ** 2
        delta_y_sqr = (old_y - drop.y) ** 2
        if drop.radius**2 > delta_x_sqr + delta_y_sqr:
            leave_streaks(old_x, old_y, drop, args)


def leave_streaks(old_x, old_y, drop, args):
    height_map = drop.height_map
    if old_x == drop.x:
        for y in range(old_y,drop.y):
            center_x = drop.x + random.randint(-2, 2)
            for x in range(center_x - math.floor(0.8 * drop.radius), math.ceil(center_x + 0.8 * drop.radius)):
                if (0 <= y < args.height) and (0 <= x < args.width):
                    height_map[x, y] = max(np.sqrt(drop.radius ** 2 - (x - center_x) ** 2),height_map[x,y])
    # Todo: add support for diagonal streaks


# Goes over all active drops, and has them leave
def leave_residual_droplets(height_map, active_drops, drop_array, new_drops, args):
    for drop in active_drops:
        if drop.mass > args.m_static:
            if drop.residual_probability() < np.random.uniform():
                drop.t_i = 0
                a = np.random.uniform(0.05, 0.15)
                new_drop_mass = min(args.m_static, a*drop.mass)
                drop.set_mass(drop.mass - new_drop_mass)
                Droplet(drop.x, drop.y, new_drop_mass, 0,
                        height_map, active_drops, drop_array, new_drops)


def compute_height_map(height_map, args):
    height_map = smooth_height_map(height_map, args)
    floor_water(height_map)


def update_height_map(args, height_map, active_drops, new_drops=[]):
    for drop in itertools.chain(active_drops, new_drops):
        for y in range(drop.get_lowest_y(), drop.get_highest_y() + 1):
            for x in range(drop.get_lowest_x(), drop.get_highest_x() + 1):
                if (0 <= y < args.height) and (0 <= x < args.width):
                    new_height = drop.get_height(x, y)
                    if height_map[x][y] < new_height:
                        height_map[x][y] = new_height


def smooth_height_map(height_map, args):
    if args.kernel == "dwn":
        kernel = np.array([[4/27, 1/9, 2/27],
                       [4/27, 1/9, 2/27],
                       [4/27, 1/9, 2/27]])
    else:
        kernel = np.array([[1/9, 1/9, 1/9],
                        [1/9, 1/9, 1/9],
                        [1/9, 1/9, 1/9]])

    return ndimage.convolve(height_map, kernel, mode='constant', cval=0)


def floor_water(height_map):
    height_map[height_map < 0.2] = 0.0


def detect_intersections(active_drops, drop_array, new_drops):
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


def merge_drops(active_drops, drop_array, new_drops):
    intersecting_drops = detect_intersections(active_drops, drop_array, new_drops)

    for a, b in intersecting_drops:

        new_velocity = (a.velocity * a.mass + b.velocity * b.mass) / (a.mass + b.mass)

        if a.y < b.y:
            low_drop = a
            high_drop = b
        else:
            low_drop = b
            high_drop = b

        low_drop.velocity = new_velocity
        low_drop.set_mass(low_drop.mass + high_drop.mass)
        high_drop.delete()


def trim_drops(active_drops, args):
    drops_to_remove = []
    for drop in active_drops:
        radius = drop.radius
        x = drop.x
        y = drop.y
        if x + radius < 0 or x - radius > args.width or y + radius < 0 or y - radius > args.height:
            drops_to_remove.append(drop)

    for drop in drops_to_remove:
        drop.delete()

def run_loop(queue, args):
    import file_ops as fo
    while not queue.empty():
        curr_run = queue.get()

        drop_array = []
        active_drops = []
        print("Staring process " + str(curr_run))
        height_map = np.zeros(shape=(args.width, args.height))

        for ii in range(int(args.steps)):
            new_drops = []

            add_drops(height_map, active_drops, drop_array, new_drops, args)

            iterate_over_drops(active_drops, args)

            if args.leave_residuals:
                leave_residual_droplets(height_map, active_drops, drop_array, new_drops, args)
            merge_drops(active_drops, drop_array, new_drops)
            trim_drops(active_drops, args)
            update_height_map(args, height_map, active_drops)
            compute_height_map(height_map, args)

            drop_array.extend(new_drops)
            new_drops.clear()

        update_height_map(args, height_map, active_drops, drop_array)
        compute_height_map(height_map, args)

        fo.save(fo.choose_file_name(args, curr_run), height_map, args)

        print("\rRun " + str(curr_run + 1) + " out of " + str(args.runs) + " is complete.")


if __name__ == '__main__':
    import arg_parser
    import file_ops as fo
    import sim as sp
    args = arg_parser.parse_arguments()

    fo.set_up_directories(args)

    if args.runs >= 1:
        import multiprocessing
        from multiprocessing import Process, Queue
        processes = []
        q = Queue()
        for i in range(int(args.runs)):
            q.put(i)
        for i in range(multiprocessing.cpu_count()-1):
            p = Process(target=run_loop, args=(q, args))
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
