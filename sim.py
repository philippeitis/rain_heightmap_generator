import math
import scipy.stats as st
import numpy as np
import random
import itertools
from scipy import ndimage


class Droplet:
    def __init__(self, x, y, mass, velocity=0):
        self.x = x
        self.y = y
        self.mass = mass
        self.velocity = velocity
        self.direction = 0
        self.t_i = 0
        self.hemispheres = [(self.x, self.y)]  # (x,y) tuples (could add z to represent share of mass)
        if mass < m_static and hemispheres_enabled:
            self.generate_hemispheres()

        self.radius = 0
        self.calculate_radius()

    def calculate_radius(self):
        # Only called when mass changes to avoid excessive calculations
        self.radius = np.cbrt((3 / 2) / math.pi * (self.mass / len(self.hemispheres)) / density_water) / scale_factor * width

    def generate_hemispheres(self):
        # Random walk to decide locations of hemispheres.
        self.hemispheres = [(self.x, self.y)]
        num_hemispheres = random.randint(1, max_hemispheres)
        directions = (1,2,3,4)
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
            acceleration = (self.mass * gravity - friction_constant_force)/self.mass
            self.velocity = self.velocity + acceleration * time_step
            if self.velocity > 0:
                self.direction = choose_direction(self)
                if self.direction == -1:
                    self.x -= math.floor(math.sqrt(self.velocity) * scale_factor * width) + random.randint(-2,2)
                    self.y += math.floor(math.sqrt(self.velocity) * scale_factor * width) + random.randint(-2,2)
                if self.direction == 0:
                    self.y += math.floor(self.velocity * scale_factor * width) + random.randint(-2,2)
                if self.direction == 1:
                    self.x += math.floor(math.sqrt(self.velocity) * scale_factor * width) + random.randint(-2,2)
                    self.y += math.floor(math.sqrt(self.velocity) * scale_factor * width) + random.randint(-2,2)

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
                if math.sqrt(delta_x**2 + delta_y**2) < drop.radius + self.radius + attraction_radius:
                    return True
        return False

    def residual_probability(self):
        if self.velocity > 0:
            return min(1,beta * time_step / t_max * min(1, self.t_i / t_max))
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

        else:                                                                           # This does not work correctly
            summation = 0.0

            for hemi_x, hemi_y in self.hemispheres:                                               # for each hemisphere, check if point is in bounds
                delta_x = x - hemi_x
                delta_y = y - hemi_y
                rad_sqr = self.radius ** 2
                distance_from_center_sqr = delta_x ** 2 + delta_y ** 2  # distance (pythagoras)
                if rad_sqr >= distance_from_center_sqr:                                 # in bounds?
                    summation += np.sqrt(rad_sqr - distance_from_center_sqr)  # if yes, add height to total

            return summation


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
        mass = min(m_max,max(m_min, np.random.normal(average_mass, deviation_mass, 1)))
        drop_to_add = Droplet(random.randint(0, width), random.randint(0, height), mass)
        if mass > m_static:
            active_drops.append(drop_to_add)
        else:
            new_drops.append(drop_to_add)


# Iterates over all active drops (which are moving faster than a given speed) to update their position
def iterate_over_drops():
    for drop in active_drops:
        old_x = drop.x
        old_y = drop.y
        ## TODO: handling for fast particles (and streaks of water)
        drop.iterate_position()
        delta_x_sqr = (old_x - drop.x) ** 2
        delta_y_sqr = (old_y - drop.y) ** 2
        if drop.radius**2 > delta_x_sqr + delta_y_sqr:
            leave_streaks(old_x, old_y, drop)


def leave_streaks(old_x, old_y, drop):
    if old_x == drop.x:
        for y in range(old_y,drop.y):
            center_x = drop.x + random.randint(-2, 2)
            for x in range(center_x - math.floor(0.8 * drop.radius), math.ceil(center_x + 0.8 * drop.radius)):
                if (0 <= y < height) and (0 <= x < width):
                    height_map[x, y] = max(np.sqrt(drop.radius ** 2 - (x - center_x) ** 2),height_map[x,y])
    # Todo: add support for diagonal streaks

# Goes over all active drops, and has them leave
def leave_residual_droplets():
    for drop in active_drops:
        if drop.mass > m_static:
            if drop.residual_probability() < np.random.uniform():
                drop.t_i = 0
                a = np.random.uniform(0.05, 0.15)
                new_drop_mass = min(m_static, a*drop.mass)
                drop.mass -= new_drop_mass
                drop.calculate_radius()

                if drop.mass < m_static:
                    active_drops.remove(drop)
                    new_drops.append(drop)
                new_drops.append(Droplet(drop.x, drop.y, new_drop_mass, 0))


def compute_height_map():
    smooth_height_map()
    floor_water()


def update_height_map():
    for drop in itertools.chain(active_drops,new_drops):
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
    height_map[height_map < 0.2] = 0.0


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
                detections.append((drop_a,drop_b))

    return detections


def merge_drops():
    intersecting_drops = detect_intersections()

    for a, b in intersecting_drops:
        if a in drop_array and b in drop_array:

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
            if low_drop.mass <= m_static and hemispheres_enabled:
                low_drop.generate_hemispheres()
                if low_drop not in new_drops:
                    new_drops.append(low_drop)
                    drop_array.remove(low_drop)
            elif a.mass > m_static and a not in active_drops:
                active_drops.append(low_drop)

            if high_drop in drop_array:
                drop_array.remove(high_drop)
            elif high_drop in active_drops:
                active_drops.remove(high_drop)
            elif high_drop in new_drops:
                new_drops.remove(high_drop)


def trim_drops():
    drops_to_remove = []
    for drop in active_drops:
        radius = drop.radius
        x = drop.x
        y = drop.y
        if x + radius < 0 or x - radius > width or y + radius < 0 or y - radius > height:
            drops_to_remove.append(drop)

    for drop in drops_to_remove:
        active_drops.remove(drop)


def empty_new_drop_arr():
    global new_drops
    drop_array.extend(new_drops)
    new_drops = []


## File output
def save(filename, fformat):
    import PIL
    border = int(args.border)

    height_map[0:border] = 0
    height_map[width - border:] = 0
    height_map[:, 0:border] = 0
    height_map[:, height - border:] = 0

    if fformat == "txt":
        np.savetxt(filename + ".txt", height_map, delimiter=",")
        print("File saved to " + file_name + ".txt")

    elif fformat == "png":
        from PIL import Image
        maximum_drop_size = np.amax(height_map)
        im = PIL.Image.new('RGBA', (width, height), 0)
        pixels = im.load()
        for x in range(width):
            for y in range(height):
                pixel_val = math.floor(height_map[x, y] / maximum_drop_size * 255)
                pixels[x, y] = (pixel_val, pixel_val, pixel_val)

        if int(args.show) == 1:
            im.show()

        im.save(filename + ".png", 'PNG')
        print("File saved to " + file_name + ".png")


def set_up_directories():
    import os
    if args.path != "./":
        try:
            os.mkdir(args.path)
            print("Directory created.")
        except FileExistsError:
            print("Directory already exists.")


def padded_zeros(ref_string, curr_num):
    out_string = str(curr_num)
    while len(out_string) < len(ref_string):
        out_string = "0" + out_string
    return out_string


def generate_time_stamp():
    from datetime import datetime
    now = datetime.now()  # current date and time
    return now.strftime("%m-%d0%Y-%H-%M-%S")


if __name__ == '__main__':
    import argparse
    import cProfile

    parser = argparse.ArgumentParser(description='Create the height map for rain on a surface.')
    parser = argparse.ArgumentParser(description='Create the height map for rain on a surface.')
    parser.add_argument('steps', type=int, help='Number of simulation steps to run') # around 50 time steps is good

    parser.add_argument('--imw', dest='w', default=720, type=int,
                        help='Sets the width of the height map and the output file.')
    parser.add_argument('--imh', dest='h', default=480, type=int,
                        help='Sets the height of the height map and the output file.')
    parser.add_argument('--width', dest='scale', default=0.3, type=float,
                        help='width of height map in meters (default 0.3)')

    parser.add_argument('--w', dest='water', default=1000, type=int,
                        help='Sets the density of water, in kg/m^3')

    parser.add_argument('--drops', dest='drops', default=5, type=int,
                        help='Sets the number of drops added to the height map '
                             'each time step.')

    parser.add_argument('--merge_radius', dest='attraction', default=2, type=int,
                        help='Drops will now merge if they are separated by less than n pixels')

    parser.add_argument('--residual_drops', dest='leave_residuals', default=True, type=bool,
                        help='Enables leaving residual drops')
    parser.add_argument('--beta', dest='beta', default=0.5, type=float,
                        help='Sets value b in equation used to determine if drop should be left or not')
    parser.add_argument('--kernel', dest='kernel', default="dwn", type=str,
                        help='Type of kernel used in smoothing step. '
                             '(dwn for downward trending, avg for averaging kernel)')

    parser.add_argument('--mmin', dest='m_min', default=0.000001, type=float,
                        help='Minimum mass of droplets (kg)')
    parser.add_argument('--mavg', dest='m_avg', default=0.000034, type=float,
                        help='Average mass of drops (kg)')
    parser.add_argument('--mdev', dest='m_dev', default=0.000016, type=float,
                        help='Average deviation of drops (kg). Higher '
                             'values create more diverse drop sizes.')
    parser.add_argument('--mmax', dest='m_max', default=1, type=float,
                        help='Maximum mass of droplets (kg)')
    parser.add_argument('--mstatic', dest='m_static', default=0.8, type=float,
                        help='Sets the percentage of drops that are static.')

    parser.add_argument('--hemispheres', dest='enable_hemispheres', default=True, type=bool,
                        help='Enables drops with multiple hemispheres (on by default)')
    parser.add_argument('--numh', dest='max_hemispheres', default=5, type=int,
                        help='Maximum number of hemispheres per drop. '
                             'Performance drops off rapidly after 15 hemispheres.')

    parser.add_argument('--g', dest='g', default=9.81, type=float,
                        help='Gravitational constant (m/s)')

    parser.add_argument('--time', dest='time', default=0.001, type=float,
                        help='Duration of each time step (in seconds)')

    parser.add_argument('--path', dest='path', default="./", type=str,
                        help='Output file path. If not defined, program defaults to same folder.')

    parser.add_argument('--name', dest='name', type=str,
                        help='Output file name. If not defined, program defaults to using date-time string.')

    parser.add_argument('--s', dest='show', default=0, type=int,
                        help='Show image on program completion..')

    parser.add_argument('--f', dest='format', default="png", type=str,
                        help='Output file format (png, txt, or npy).')
    parser.add_argument('--border', dest='border', default=0, type=int,
                        help='Sets all values within border pixels of the edge to 0')

    parser.add_argument('--runs', dest='runs', default=1, type=int,
                        help='Will execute the program with the given parameters repeatedly.')

    parser.add_argument('--verbose', dest='verbose', default="", type=str,
                        help='Will output detailed information on program operation. '
                        't : time to execute each step, '
                        'd : number of droplets in each step, '
                        'a : average mass of droplets in each step.')

    args = parser.parse_args()

    width = int(args.w)
    height = int(args.h)
    scale_factor = float(args.scale)

    density_water = int(args.water)
    gravity = float(args.g)
    beta = 0.5

    m_min = float(args.m_min)
    m_max = float(args.m_max)
    average_mass = float(args.m_avg)
    deviation_mass = float(args.m_dev)  # normally distributed
    hemispheres_enabled = bool(args.enable_hemispheres)
    max_hemispheres = int(args.max_hemispheres)
    m_static = average_mass + deviation_mass * st.norm.ppf(float(args.m_static))
    friction_constant_force = m_static * gravity
    attraction_radius = int(args.attraction)

    time_step = float(args.time)
    t_max = time_step * 4

    temp_name = "temp"
    temp_path = "./temp/"

    set_up_directories()

    verbose = args.verbose
    show_time = "t" in verbose
    show_drop_count = "d" in verbose
    show_average_drop_mass = "a" in verbose

    if show_time:
        import time

    kernel = np.array(1)

    if args.kernel == "dwn":
        kernel = np.array([[4/27, 1/9, 2/27],
                       [4/27, 1/9, 2/27],
                       [4/27, 1/9, 2/27]])
    else:
        kernel = np.array([[1/9, 1/9, 1/9],
                        [1/9, 1/9, 1/9],
                        [1/9, 1/9, 1/9]])

    name = ""
    if args.name:
        name = args.name
    else:
        name = generate_time_stamp()

    for i in range(int(args.runs)):
        file_name = ""
        drop_array = []
        active_drops = []

        height_map = np.zeros(shape=(width, height))

        if int(args.runs) > 1:
            if not args.name:
                file_name = name + "_" + padded_zeros(args.runs, i)
            else:
                file_name = name + padded_zeros(args.runs, i)
        else:
            file_name = name

        for ii in range(int(args.steps)):
            new_drops = []

            if show_time:
                start_time = time.time()

            add_drops(int(args.drops))      # Very fast
            iterate_over_drops()            # Very fast

            if bool(args.leave_residuals):
                leave_residual_droplets()   # Very fast
            merge_drops()                   # Takes around 1/3 of the processing time
            trim_drops()                    # Very fast
            update_height_map()             # Takes around 1/3 of the processing time
            compute_height_map()            # Very fast
            empty_new_drop_arr()

            output_string = "\rStep " + str(ii+1) + " out of " + args.steps + " is complete."

            if show_time:
                elapsed_time = time.time()-start_time
                output_string = output_string + "\nCalculation took " + str(elapsed_time) + " seconds."
            if show_drop_count:
                output_string = output_string + "\nThere are currently " + str(len(drop_array)+len(active_drops)) + \
                                " drops in the height map, of which " + str(len(active_drops)) + " are in motion."
            if show_average_drop_mass:
                masses = 0.0
                for drop in drop_array:
                    masses += drop.mass
                avg_mass = masses/len(drop_array)
                output_string = output_string + "\nThe average mass of the drops is " + str(avg_mass) + " kg."
            print(output_string)

        new_drops = drop_array
        update_height_map()
        compute_height_map()

        if args.format != "none":
            if args.name:
                save(args.path + file_name, args.format)
            else:
                save(args.path + file_name, args.format)

        if (int(args.runs)) > 1:
            print("\rRun " + str(i + 1) + " out of " + args.runs + " is complete.")
