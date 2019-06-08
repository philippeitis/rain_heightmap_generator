import math
import scipy.stats as st
import numpy as np
import random
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

    def generate_hemispheres(self):
        self.hemispheres = [(self.x,self.y)]
        num_hemispheres = random.randint(1, max_hemispheres)

        while len(self.hemispheres) < num_hemispheres:
            old_x, old_y = self.hemispheres[-1]
            new_x, new_y = 0,0
            direction = random.randint(1, 4)
            if direction == 1:  # Down
                new_y = old_y - 1
                new_x = old_x
            if direction == 2:  # Left
                new_y = old_y
                new_x = old_x - 1
            if direction == 3:  # Up
                new_y = old_y + 1
                new_x = old_x
            if direction == 4:  # Right
                new_y = old_y
                new_x = old_x + 1

            if not (new_x, new_y) in self.hemispheres:
                self.hemispheres.append((new_x, new_y))

    def iterate_position(self):
        if self.mass > m_static:
            acceleration = (self.mass * gravity - friction_constant_force)/self.mass
            self.velocity = self.velocity + acceleration * time_step
            if self.velocity > 0:
                self.direction = choose_direction(self)
                if self.direction == -1:
                    self.x -= math.floor(math.sqrt(self.velocity) * scale_factor * width)
                    self.y += math.floor(math.sqrt(self.velocity) * scale_factor * width)
                if self.direction == 0:
                    self.y += math.floor(self.velocity * scale_factor * width)
                if self.direction == 1:
                    self.x += math.floor(math.sqrt(self.velocity) * scale_factor * width)
                    self.y += math.floor(math.sqrt(self.velocity) * scale_factor * width)

                # This determines if we should leave a droplet behind
                self.t_i += 1

    def intersects(self, drop):
        if drop.get_lowest_y() < self.get_highest_y() or drop.get_highest_y() > self.get_lowest_y():
            return False

        for self_x, self_y in self.hemispheres:
            for drop_x, drop_y in drop.hemispheres:
                delta_x = self_x - drop_x
                delta_y = self_y - drop_y
                if math.sqrt(delta_x**2 + delta_y**2) < drop.radius() + self.radius():
                    return True
        return False

    def radius(self):
        return np.cbrt((3 / 2) / math.pi * (self.mass / len(self.hemispheres)) / density_water) / scale_factor * width

    def residual_probability(self):
        if self.velocity > 0:
            return min(1,beta * time_step / t_max * min(1, self.t_i / t_max))
        else:
            return 0

    def get_lowest_y(self):
        return min(self.hemispheres, key=lambda t: t[1])[1] - math.floor(self.radius()) - 1

    def get_highest_y(self):
        return max(self.hemispheres, key=lambda t: t[1])[1] + math.floor(self.radius()) + 1

    def get_lowest_x(self):
        return min(self.hemispheres, key=lambda t: t[0])[0] - math.floor(self.radius()) - 1

    def get_highest_x(self):
        return max(self.hemispheres, key=lambda t: t[0])[0] + math.floor(self.radius()) + 1

    def get_height(self, x, y):
        if len(self.hemispheres) >= 1:
            return np.sqrt(self.radius() ** 2 - (y - self.y) ** 2 - (x - self.x) ** 2)
        else:
            summation = 0.0
            for x, y in self.hemispheres:
                distance_from_center = np.sqrt((y - self.y) ** 2 + (x - self.x) ** 2)
                if self.radius() >= distance_from_center > 0:
                    summation += np.sqrt(self.radius() ** 2 - distance_from_center ** 2)
            return summation


def choose_direction(droplet):
    # Chooses three regions in front of the drop, calculates total water volume
    # and returns region with highest water volume, or random if equal quantities of
    # water present
    radius = math.floor(droplet.radius())
    start_y = droplet.y + radius
    end_y = start_y + radius * 2
    x_1 = droplet.x + radius * 3
    x_2 = droplet.x + radius
    x_3 = droplet.x - radius
    x_4 = droplet.x - 3 * radius

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
        mass = min(m_max,max(m_min,np.random.normal(average_mass, deviation_mass, 1)))
        drop_to_add = Droplet(random.randint(0, width), random.randint(0, height), mass)
        drop_array.append(drop_to_add)


def iterate_over_drops():
    for drop in drop_array:
        ## TODO: handling for fast particles (and streaks of water)
        drop.iterate_position()


def leave_residual_droplets():
    drops_to_add = []
    for drop in drop_array:
        if drop.residual_probability() < np.random.uniform():
            global residual_count
            residual_count += 1
            print("Adding residual" + str(residual_count))
            drop.t_i = 0
            a = np.random.uniform(0.1, 0.3)
            new_drop_mass = min(m_static, a*drop.mass)
            drop.mass -= new_drop_mass
            drops_to_add.append(Droplet(drop.x, drop.y, new_drop_mass, 0))

    drop_array.extend(drops_to_add)


def compute_height_map():
    smooth_height_map()
    floor_water()


def update_height_map():
    for drop in drop_array:
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

    for a in range(len(drop_array)):
        for b in range(a+1, len(drop_array)):
            if drop_array[a].intersects(drop_array[b]):
                detections.append((drop_array[a], drop_array[b]))

    return detections


def merge_drops():
    intersecting_drops = detect_intersections()

    for a, b in intersecting_drops:
        if a in drop_array and b in drop_array:

            new_velocity = (a.velocity * a.mass + b.velocity * b.mass) / (a.mass + b.mass)

            if a.y < b.y:
                a.velocity = new_velocity
                a.mass += b.mass
                if a.mass < m_static and hemispheres_enabled:
                    a.generate_hemispheres()
                drop_array.remove(b)

            else:
                b.velocity = new_velocity
                b.mass += a.mass
                if b.mass < m_static and hemispheres_enabled:
                    b.generate_hemispheres()
                drop_array.remove(a)


def trim_drops():
    drops_to_remove = []
    for drop in drop_array:
        radius = drop.radius()
        x = drop.x
        y = drop.y
        if x + radius < 0 or x - radius > width or y + radius < 0 or y - radius > height:
            drops_to_remove.append(drop)

    for drop in drops_to_remove:
        drop_array.remove(drop)


def generate_time_stamp():
    from datetime import datetime
    now = datetime.now()  # current date and time
    return now.strftime("%m-%d0%Y-%H-%M-%S")


def save(filename, fformat):
    import PIL
    border = int(args.border)

    height_map[0:border] = 0
    height_map[width - border:] = 0
    height_map[:, 0:border] = 0
    height_map[:, height - border:] = 0

    if fformat == "txt":
        np.savetxt(filename + ".txt", height_map, delimiter=",")

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


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Create the height map for rain on a surface.')
    parser.add_argument('steps') # around 50 time steps is good

    parser.add_argument('--imw', dest='w', default=720,
                        help='Sets the width of the height map and the output file.')
    parser.add_argument('--imh', dest='h', default=480,
                        help='Sets the height of the height map and the output file.')
    parser.add_argument('--width', dest='scale', default=0.3,
                        help='width of height map in meters (default 0.3)')

    parser.add_argument('--w', dest='water', default=1000,
                        help='Sets the density of water, in kg/m^3')

    parser.add_argument('--drops', dest='drops', default=5,
                        help='Sets the number of drops added to the height map '
                             'each time step.')

    parser.add_argument('--residual_drops', dest='leave_residuals', default=False,
                        help='Enables leaving residual drops')
    parser.add_argument('--beta', dest='beta', default=0.5,
                        help='Sets value b in equation used to determine if drop should be left or not')
    parser.add_argument('--kernel', dest='kernel', default="dwn",
                        help='Type of kernel used in smoothing step. '
                             '(dwn for downward trending, avg for averaging kernel)')

    parser.add_argument('--mmin', dest='m_min', default=0.000001,
                        help='Minimum mass of droplets (kg)')
    parser.add_argument('--mavg', dest='m_avg', default=0.000034,
                        help='Average mass of drops (kg)')
    parser.add_argument('--mdev', dest='m_dev', default=0.000016,
                        help='Average deviation of drops (kg). Higher '
                             'values create more diverse drop sizes.')
    parser.add_argument('--mmax', dest='m_max', default=1,
                        help='Maximum mass of droplets (kg)')
    parser.add_argument('--mstatic', dest='m_static', default=0.8,
                        help='Sets the percentage of drops that are static.')

    parser.add_argument('--hemispheres', dest='enable_hemispheres', default=True,
                        help='Enables drops with multiple hemispheres (on by default)')
    parser.add_argument('--numh', dest='max_hemispheres', default=5,
                        help='Maximum number of hemispheres per drop. '
                             'Performance drops off rapidly after 15 hemispheres.')

    parser.add_argument('--g', dest='g', default=9.81,
                        help='Gravitational constant (m/s)')

    parser.add_argument('--time', dest='time', default=0.001,
                        help='Duration of each time step (in seconds)')

    parser.add_argument('--path', dest='path', default="./",
                        help='Output file path. If not defined, program defaults to same folder.')

    parser.add_argument('--name', dest='name',
                        help='Output file name. If not defined, program defaults to using date-time string.')

    parser.add_argument('--s', dest='show', default=0,
                        help='Show image on program completion..')

    parser.add_argument('--f', dest='format', default="png",
                        help='Output file format (png, txt, or npy).')
    parser.add_argument('--border', dest='border', default=0,
                        help='Sets all values within border pixels of the edge to 0')

    parser.add_argument('--runs', dest='runs', default=1,
                        help='Will execute the program with the given parameters repeatedly.')

    parser.add_argument('--verbose', dest='verbose', default=1,
                        help='Will output detailed information on program operation. '
                        't : time to execute each step, '
                        'd : number of droplets in each step, '
                        'a : average mass of droplets in each step.')
    args = parser.parse_args()

    width = int(args.w)
    height = int(args.h)
    scale_factor = args.scale

    density_water = args.water
    gravity = args.g
    beta = 0.5

    m_min = args.m_min
    m_max = args.m_max
    average_mass = args.m_avg
    deviation_mass = args.m_dev  # normally distributed
    hemispheres_enabled = bool(args.enable_hemispheres)
    max_hemispheres = int(args.max_hemispheres)
    m_static = average_mass + deviation_mass * st.norm.ppf(args.m_static)
    friction_constant_force = m_static * gravity

    time_step = args.time
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
        residual_count = 0
        file_name = ""
        drop_array = []
        height_map = np.zeros(shape=(width, height))

        if int(args.runs) > 1:
            if not args.name:
                file_name = name + "_" + padded_zeros(args.runs, i)
            else:
                file_name = name + padded_zeros(args.runs, i)
        else:
            file_name = name

        for ii in range(int(args.steps)):
            if show_time:
                start_time = time.time()

            add_drops(int(args.drops))
            iterate_over_drops()
            if bool(args.leave_residuals):
                leave_residual_droplets()
            update_height_map()
            compute_height_map()
            merge_drops()
            trim_drops()
            output_string = "\rStep " + str(ii+1) + " out of " + args.steps + " is complete."

            if show_time:
                elapsed_time = time.time()-start_time
                output_string = output_string + "\nCalculation took " + str(elapsed_time) + " seconds."
            if show_drop_count:
                output_string = output_string + "\nThere are currently " + str(len(drop_array)) + " active drops in the height map."
            if show_average_drop_mass:
                masses = 0.0
                for drop in drop_array:
                    masses += drop.mass
                avg_mass = masses/len(drop_array)
                output_string = output_string + "\nThe average mass of the drops is " + str(avg_mass) + " kg."
            print(output_string)

        if args.format != "none":
            if args.name:
                save(args.path + file_name, args.format)
            else:
                save(args.path + file_name, args.format)

        if (int(args.runs)) > 1:
            print("\rRun " + str(i + 1) + " out of " + args.runs + " is complete.")
