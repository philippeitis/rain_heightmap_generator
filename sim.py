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
        new_x = self.x
        new_y = self.y
        while len(self.hemispheres) < num_hemispheres:
            direction = random.randint(1, 4)
            if direction == 1:  # Down
                new_y -= 1
            if direction == 2:  # Left
                new_x -= 1
            if direction == 3:  # Up
                new_y += 1
            if direction == 4:  # Right
                new_x += 1

            if not (new_x, new_y) in self.hemispheres:
                self.hemispheres.append((new_x,new_y))

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
        lowest_y = height
        for x, y in self.hemispheres:
            if y < lowest_y:
                lowest_y = y
        return lowest_y - math.floor(self.radius())

    def get_highest_y(self):
        highest_y = 0
        for x, y in self.hemispheres:
            if y > highest_y:
                highest_y = y
        return highest_y + math.floor(self.radius())

    def get_lowest_x(self):
        lowest_x = width
        for x, y in self.hemispheres:
            if x < lowest_x:
                lowest_x = x
        return lowest_x - math.floor(self.radius())

    def get_highest_x(self):
        highest_x = 0
        for x, y in self.hemispheres:
            if x > highest_x:
                highest_x = x
        return highest_x + math.floor(self.radius())

    def get_height(self, x, y):
        if len(self.hemispheres) == 1:
            return np.sqrt(self.radius() ** 2 - (y - self.y) ** 2 - (x - self.x) ** 2)
        else:
            summation = 0.0
            for x, y in self.hemispheres:
                local_height_sqr = self.radius() ** 2 - (y - self.y) ** 2 - (x - self.x) ** 2
                if local_height_sqr > 0:
                    summation += np.sqrt(local_height_sqr)
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
        if drop.residual_probability() < random.random():
            drop.t_i = 0
            a = random.uniform(0.1,0.3)
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
                        help='width of height map')
    parser.add_argument('--imh', dest='h', default=480,
                        help='height of height map')
    parser.add_argument('--width', dest='scale', default=0.3,
                        help='width of height map in meters (default 0.3)')

    parser.add_argument('--w', dest='water', default=1000,
                        help='density of water, in kg/m^3')

    parser.add_argument('--drops', dest='drops', default=5,
                        help='average number of drops per time step')

    parser.add_argument('--residual_drops', dest='leave_residuals', default=False,
                        help='enable residual drops')
    parser.add_argument('--kernel', dest='kernel', default="dwn",
                        help='Type of kernel used in smoothing step. '
                             '(dwn for downward trending, avg for averaging kernel)')

    parser.add_argument('--mmin', dest='m_min', default=0.000001,
                        help='minimum mass of droplets (kg)')
    parser.add_argument('--mavg', dest='m_avg', default=0.000034,
                        help='average mass of drops (kg)')
    parser.add_argument('--mdev', dest='m_dev', default=0.000016,
                        help='average mass of drops (kg)')
    parser.add_argument('--mmax', dest='m_max', default=1,
                        help='maximum mass of droplets (kg)')
    parser.add_argument('--mstatic', dest='m_static', default=0.8,
                        help='percentage of drops that do not move ')

    parser.add_argument('--hemispheres', dest='enable_hemispheres', default=True,
                        help='enable hemisphere based drops')
    parser.add_argument('--numh', dest='max_hemispheres', default=5,
                        help='maximum number of hemispheres per drop')

    parser.add_argument('--g', dest='g', default=9.81,
                        help='gravitational constant (m/s)')

    parser.add_argument('--time', dest='time', default=0.001,
                        help='duration of each time step (in seconds)')

    parser.add_argument('--path', dest='path', default="./",
                        help='output file path. if --name is not defined, defaults to date-time string')

    parser.add_argument('--name', dest='name',
                        help='output file name')

    parser.add_argument('--s', dest='show', default=0,
                        help='flag to have image automatically appear once image is generated.')

    parser.add_argument('--f', dest='format', default="png",
                        help='file format (txt file with comma seperated rows), or png.')
    parser.add_argument('--border', dest='border', default=0,
                        help='all values within border pixels of the edge will be 0')

    parser.add_argument('--runs', dest='runs', default=1,
                        help='number of images to create.')

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
    max_hemispheres = args.max_hemispheres
    m_static = average_mass + deviation_mass * st.norm.ppf(args.m_static)
    friction_constant_force = m_static * gravity

    time_step = args.time
    t_max = time_step * 4

    temp_name = "temp"
    temp_path = "./temp/"

    set_up_directories()

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
        height_map = np.zeros(shape=(width, height))

        if int(args.runs) > 1:
            if not args.name:
                file_name = name + "_" + padded_zeros(args.runs, i)
            else:
                file_name = name + padded_zeros(args.runs, i)
        else:
            file_name = name

        for ii in range(int(args.steps)):
            add_drops(int(args.drops))
            iterate_over_drops()
            if bool(args.leave_residuals):
                leave_residual_droplets()
            update_height_map()
            compute_height_map()
            merge_drops()
            trim_drops()
            print("\rStep " + str(ii+1) + " out of " + args.steps + " is complete.")

        if args.format != "none":
            if args.name:
                save(args.path + file_name, args.format)
            else:
                save(args.path + file_name, args.format)

        if (int(args.runs)) > 1:
            print("\rRun " + str(i + 1) + " out of " + args.runs + " is complete.")
