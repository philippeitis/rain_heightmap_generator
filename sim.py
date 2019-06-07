## Constants per the paper
import PIL
import math
import scipy.stats as st
import numpy as np
import random
from scipy import ndimage

time_step = 0.01

# constants
gravity = 9.8

# max time for drops to split
t_max = time_step * 4
# may need adjustment
beta = 0.5

average_mass = 0.000034   # in kg
deviation_mass = 0.000005 # normally distributed

# percentage of drops that are created as static
m_static_below = 0.8
drop_array = []

# mass at which drops do not move
m_static = average_mass + deviation_mass * st.norm.ppf(m_static_below)
friction_constant_force = m_static * gravity


class Droplet:
    def __init__(self, x, y, mass, velocity):
        self.x = x
        self.y = y
        self.mass = mass
        self.velocity = velocity
        self.direction = 0

    def iterate_position(self):
        acceleration = (self.mass * gravity - friction_constant_force)/self.mass
        self.velocity = self.velocity + acceleration * time_step
        self.direction = choose_direction(self)
        if self.velocity > 0:
            if self.direction == -1:
                self.x -= math.floor(math.sqrt(self.velocity) * scale_factor * width)
                self.y += math.floor(math.sqrt(self.velocity) * scale_factor * width)
            if self.direction == 0:
                self.y += math.floor(self.velocity * scale_factor * width)
            if self.direction == 1:
                self.x += math.floor(math.sqrt(self.velocity) * scale_factor * width)
                self.y += math.floor(math.sqrt(self.velocity) * scale_factor * width)

    def intersects(self,drop):
        drop_radius = np.cbrt((3 / 2) / math.pi * drop.mass / 1000) / scale_factor * width
        self_radius = np.cbrt((3 / 2) / math.pi * self.mass / 1000) / scale_factor * width

        delta_x = self.x - drop.x
        delta_y = self.y - drop.y
        return math.sqrt(delta_x**2 + delta_y**2) < drop_radius + self_radius


def choose_direction(droplet):
    import random
    region_scan_size = droplet.mass
    return random.randint(-1,2)


def add_drops(avg):
    import random
    for i in range(avg):
        drop_to_add = Droplet(random.randint(0, width), random.randint(0, height), np.random.normal(average_mass, deviation_mass, 1),0)
        drop_array.append(drop_to_add)


def iterate_over_drops():
    for drop in drop_array:
        drop.iterate_position()


def leave_residual_droplets():
    return 0
    # TODO


def compute_height_map():
    smooth_height_map()
    floor_water()


def update_height_map():
    for drop in drop_array:
        radius = np.cbrt((3 / 2) / math.pi * drop.mass / 1000) / scale_factor * width
        round_radius = math.floor(radius)
        for y in range(drop.y - round_radius, drop.y + round_radius+1):
            for x in range(drop.x - round_radius, drop.x + round_radius +1):
                if (0 < y < height) and (0 < x < width):
                    new_height = np.sqrt(radius**2 - (y - drop.y)**2 - (x-drop.x)**2)
                    if height_map[x][y] < new_height:
                        height_map[x][y] = new_height


def smooth_height_map():
    kernel = np.array([[1/9, 1/9, 1/9],
                       [1/9, 1/9, 1/9],
                       [1/9, 1/9, 1/9]])
    global height_map
    height_map = ndimage.convolve(height_map, kernel, mode='constant', cval=0)



def floor_water():
    for x in range(width):
        for y in range(height):
            if height_map[x][y] < 0.1:
                height_map[x][y] = 0.0


def detect_intersections():
    detections = []
    for a in range(len(drop_array)):
        for b in range(a+1,len(drop_array)):
            if drop_array[a].intersects(drop_array[b]):
                detections.append((drop_array[a], drop_array[b]))

    return detections


def merge_drops():
    intersecting_drops = detect_intersections()

    for a, b in intersecting_drops:
        if a in drop_array and b in drop_array:
            drop_array.remove(a)
            drop_array.remove(b)
            new_velocity = (a.velocity * a.mass + b.velocity * b.mass) / (a.mass + b.mass)

            if a.y < b.y:
                drop_array.append(Droplet(a.x,a.y,a.mass + b.mass, new_velocity))
            else:
                drop_array.append(Droplet(b.x,b.y,a.mass + b.mass, new_velocity))



def trim_drops():
    drops_to_remove = []
    for drop in drop_array:
        radius = math.floor(np.cbrt((3 / 2) / math.pi * drop.mass / 1000) / scale_factor * width)
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


def find_max():
    maximum = 0
    for x in range(width):
        for y in range(height):
            if height_map[x][y] > maximum:
                maximum = height_map[x][y]
    return maximum


def save(filename):
    from PIL import Image
    im = PIL.Image.new('RGBA', (width, height), 0)
    pixels = im.load()
    for x in range(width):
        for y in range(height):
            pixel_val = math.floor(height_map[x][y] / maximum_drop_size * 255)
            pixels[x, y] = (pixel_val, pixel_val, pixel_val)

    im.show()
    im.save(filename + ".png", 'PNG')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Create the height map for rain on a surface.')
    parser.add_argument('steps')

    parser.add_argument('--imw', dest='w', default=720,
                        help='width of height map')
    parser.add_argument('--imh', dest='h', default=480,
                        help='height of height map')
    parser.add_argument('--width', dest='scale', default=0.3,
                        help='width of height map in meters (default 0.3)')

    parser.add_argument('--drops', dest='drops', default=5,
                        help='average number of drops per time step')

    parser.add_argument('--min', dest='min', default=5,
                        help='minimum mass of droplets (kg)')
    parser.add_argument('--max', dest='max', default=5,
                        help='maximum mass of droplets (kg)')

    parser.add_argument('--g', dest='g', default=9.8,
                        help='gravitational constant (m/s)')

    parser.add_argument('--mstatic', dest='m_static', default=0.1,
                        help='percentage of drops that do not move ')

    parser.add_argument('--time', dest='time', default=0.01,
                        help='duration of each time step (in seconds)')

    parser.add_argument('--path', dest='path', default="./",
                        help='output file path. if --name is not defined, defaults to date-time string')

    parser.add_argument('--name', dest='name',
                        help='output file name')

    args = parser.parse_args()

    width = int(args.w)
    height = int(args.h)
    scale_factor = args.scale

    gravity = args.g
    m_static = average_mass + deviation_mass * st.norm.ppf(args.m_static)
    friction_constant_force = m_static * gravity

    time_step = args.time
    t_max = time_step * 4
    height_map = np.zeros(shape=(width, height))

    name = generate_time_stamp()
    for i in range(int(args.steps)):
        add_drops(int(args.drops))
        iterate_over_drops()
        #leave_residual_droplets()
        update_height_map()
        compute_height_map()
        merge_drops()
        trim_drops()
        maximum_drop_size = find_max()

        print("Step " + str(i+1) + " out of " + args.steps + " is complete.")

    if args.name:
        save(args.path+args.name)
    else:
        save(args.path + generate_time_stamp())
