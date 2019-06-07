## Constants per the paper
import PIL
import math
import scipy.stats as st
import numpy as np
import random
from scipy import ndimage

class Droplet:
    def __init__(self, x, y, mass, velocity):
        self.x = x
        self.y = y
        self.mass = mass
        self.velocity = velocity
        self.direction = 0
        self.t_i = 0

    def iterate_position(self):
        acceleration = (self.mass * gravity - friction_constant_force)/self.mass
        self.velocity = self.velocity + acceleration * time_step
        self.direction = choose_direction(self)
        if self.velocity > 0:
            if self.direction == -1:
                self.x -= math.floor(math.sqrt(self.velocity) / scale_factor * width)
                self.y += math.floor(math.sqrt(self.velocity) / scale_factor * width)
            if self.direction == 0:
                self.y += math.floor(self.velocity * scale_factor * width)
            if self.direction == 1:
                self.x += math.floor(math.sqrt(self.velocity) / scale_factor * width)
                self.y += math.floor(math.sqrt(self.velocity) / scale_factor * width)

            # This determines if we should leave a droplet behind
            self.t_i += 1

    def intersects(self, drop):
        delta_x = self.x - drop.x
        delta_y = self.y - drop.y
        return math.sqrt(delta_x**2 + delta_y**2) < drop.radius() + self.radius()

    def radius(self):
        return np.cbrt((3 / 2) / math.pi * self.mass / 1000) / scale_factor * width

    def residual_probability(self):
        if self.velocity > 0:
            return min(1,beta * time_step / t_max * min(1, self.t_i / t_max))
        else:
            return 0


def choose_direction(droplet):
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
        drop_to_add = Droplet(random.randint(0, width), random.randint(0, height),
                              np.random.normal(average_mass, deviation_mass, 1), 0)
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
        radius = drop.radius()
        round_radius = math.floor(radius)
        for y in range(drop.y - round_radius, drop.y + round_radius+1):
            for x in range(drop.x - round_radius, drop.x + round_radius+1):
                if (0 <= y < height) and (0 <= x < width):
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
    height_map[height_map < 0.2] = 0.0


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


def save(filename):
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

    parser.add_argument('--drops', dest='drops', default=5,
                        help='average number of drops per time step')

    parser.add_argument('--mmin', dest='m_min',
                        help='minimum mass of droplets (kg)')
    parser.add_argument('--mavg', dest='m_avg', default=0.000034,
                        help='average mass of drops (kg)')
    parser.add_argument('--mdev', dest='m_dev', default=0.000005,
                        help='average mass of drops (kg)')
    parser.add_argument('--mmax', dest='m_max',
                        help='maximum mass of droplets (kg)')
    parser.add_argument('--mstatic', dest='m_static', default=0.8,
                        help='percentage of drops that do not move ')

    parser.add_argument('--g', dest='g', default=9.81,
                        help='gravitational constant (m/s)')

    parser.add_argument('--time', dest='time', default=0.005,
                        help='duration of each time step (in seconds)')

    parser.add_argument('--path', dest='path', default="./",
                        help='output file path. if --name is not defined, defaults to date-time string')

    parser.add_argument('--name', dest='name',
                        help='output file name')

    parser.add_argument('--s', dest='show', default=0,
                        help='flag to have image automatically appear once image is generated.')

    args = parser.parse_args()

    width = int(args.w)
    height = int(args.h)
    scale_factor = args.scale

    gravity = args.g
    beta = 0.5

    average_mass = args.m_avg
    deviation_mass = args.m_dev  # normally distributed

    m_static = average_mass + deviation_mass * st.norm.ppf(args.m_static)
    friction_constant_force = m_static * gravity

    time_step = args.time
    t_max = time_step * 4

    drop_array = []
    height_map = np.zeros(shape=(width, height))

    for i in range(int(args.steps)):
        add_drops(int(args.drops))
        iterate_over_drops()
        #leave_residual_droplets()
        update_height_map()
        compute_height_map()
        merge_drops()
        trim_drops()

        print("Step " + str(i+1) + " out of " + args.steps + " is complete.")

    if args.name:
        save(args.path+args.name)
    else:
        save(args.path + generate_time_stamp())
