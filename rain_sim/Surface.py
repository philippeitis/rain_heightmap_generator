import numpy as np
from scipy import ndimage
from Droplet import Droplet

class Surface():
    def __init__(self, args):
        self.width = args.width
        self.height = args.height
        self.scale = args.scale
        self.one_meter_in_pixels = self.width / self.scale

        # masses in kg
        self.m_min = args.m_min
        self.m_max = args.m_max
        self.p_static = args.p_static
        self.m_static = args.m_static
        self.max_hemispheres = args.max_hemispheres

        self.time_step = args.time_step
        self.drops_per_iter = args.drops_per_iter
        self.residual_t_max = self.time_step * 4
        self.beta = args.beta
        self.floor_val = args.floor_val
        self.residual_floor = args.residual_floor
        self.residual_ceil = args.residual_ceil

        self.exp_lambda = -np.log(1-self.p_static) / self.m_static

        self.water_density = 997 # kg/m^3 => radii will be calculated in m
        self.gravity = 9.81 # m/s^2
        self.friction = self.m_static * self.gravity

        self.height_map = np.zeros(shape=(self.height, self.width))
        self.id_map = np.zeros(shape=(self.height, self.width))
        self.affinity_map = np.reshape(np.random.uniform(size=self.height * self.width), (self.height, self.width))

        self.kernel = [[1/9,1/9,1/9], [1/9,1/9,1/9], [1/9,1/9,1/9]]
        self.kernel2 = [[0,1/3,0], [0,1/3,0], [0,1/3,0]]
        self.erosion_factor = 0.9

        self.drop_dict = {}
        self.active_drops = []
        self.static_drops = []

        self.curr_drop_id = 1
        self.collisions = {}
        self.merged_drops_to_delete = []
        self.dict_already_seen = []

        #self.current_drop_map = np.zeros(shape=(self.height, self.width))


    def add_drops(self):
        for i in range(self.drops_per_iter):
            mass = min(self.m_max, max(self.m_min, np.random.exponential(1/self.exp_lambda)))
            Droplet(x=np.random.randint(0, self.width), y=np.random.randint(0, self.height),
                                  mass=mass, surface=self)


    def iterate_active_drops(self):
        for drop in self.active_drops:
            assert drop.mass > self.m_static or drop.velocity > 0
            drop.iterate_position()


    def out_of_bounds(self, x, y):
        return (x < 0 or x >= self.width or y < 0 or y >= self.height)


    def update_drop_height(self, drop, hemi):
        hemi_x, hemi_y = hemi[0], hemi[1]

        for x in range(hemi_x - drop.radius, hemi_x + drop.radius + 1):
            for y in range(hemi_y - drop.radius, hemi_y + drop.radius + 1):
                if (self.out_of_bounds(x, y)): continue

                xy_delta_dist = drop.radius_sqr - ((x - hemi_x) ** 2 + (y - hemi_y) ** 2)
                if xy_delta_dist > 0:
                    if self.height_map[y, x] < np.sqrt(xy_delta_dist):
                        self.height_map[y, x] = np.sqrt(xy_delta_dist)
                        curr_id = self.id_map[y, x]

                        self.id_map[y, x] = drop.id

                        if (curr_id != 0) and (curr_id != drop.id):
                            curr_drop = self.drop_dict[curr_id]
                            if (curr_drop.parent_id != drop.id) and \
                                (drop.parent_id != curr_id):
                                if (drop.id in self.collisions.keys()):
                                    self.collisions[drop.id].append(curr_id)
                                else:
                                    self.collisions[drop.id] = [curr_id]
                                if (curr_id in self.collisions.keys()):
                                    self.collisions[curr_id].append(drop.id)
                                else:
                                    self.collisions[curr_id] = [drop.id]


    def update_height_map(self):
        for drop in self.active_drops:
            for hemi in drop.active_path:
                self.update_drop_height(drop, hemi)
            drop.active_path = []

        for drop in self.static_drops:
            for hemi in drop.hemispheres: self.update_drop_height(drop, hemi)
            for hemi in drop.active_path: self.update_drop_height(drop, hemi)
            drop.active_path = []


    def smooth_heightmap(self):
        self.height_map = ndimage.convolve(self.height_map, self.kernel, mode='constant', cval=0)
        self.height_map = ndimage.convolve(self.height_map, self.kernel2, mode='constant', cval=0)
        self.height_map[self.height_map < self.floor_val] = 0


    def update_drop_lists(self):
        for drop in self.static_drops:
            if (self.out_of_bounds(drop.x, drop.y) or drop.id in self.merged_drops_to_delete):
                self.static_drops.remove(drop)
                self.drop_dict.pop(drop.id)

        for drop in self.active_drops:
            if (self.out_of_bounds(drop.x, drop.y) or drop.id in self.merged_drops_to_delete):
                self.active_drops.remove(drop)
                self.drop_dict.pop(drop.id)


    def get_all_colliding(self, drop_id):
        collisions = [drop_id]

        # drop becomes seen once its list is visited
        if (drop_id in self.dict_already_seen): return collisions

        # visit its list
        my_collisions = self.collisions[drop_id]
        self.dict_already_seen.append(drop_id)
        collisions.extend(my_collisions)

        for drop in my_collisions:
            drop_collisions = self.get_all_colliding(drop)

            for new_drop in drop_collisions:
                if new_drop not in collisions:
                    collisions.append(new_drop)

        unique_collisions = []
        for drop in collisions:
            if drop not in unique_collisions: unique_collisions.append(drop)
        return unique_collisions


    def merge_drops(self):
        len_collisions = len(self.collisions.keys())
        for i, drop_id in enumerate(self.collisions.keys()):
            if drop_id in self.dict_already_seen: continue
            if i >= len_collisions: return

            drop_collisions = self.get_all_colliding(drop_id)

            lowest_pos = (-1,-1)
            total_mass = 0
            v_numerator = 0

            for merged_drop_id in drop_collisions:
                merged_drop = self.drop_dict[merged_drop_id]
                total_mass += merged_drop.mass
                v_numerator += merged_drop.mass * merged_drop.velocity

                if (merged_drop.y > lowest_pos[1]):
                    lowest_pos = (merged_drop.x, merged_drop.y)
                self.merged_drops_to_delete.append(merged_drop_id)


            Droplet(lowest_pos[0], lowest_pos[1], total_mass, self, velocity= v_numerator / total_mass)

    def erode_heightmap(self):
        for x in range(self.width):
            for y in range(self.height):
                if (self.height_map[y,x] != 0 and self.id_map[y,x] == 0):
                    self.height_map[y,x] = self.height_map[y,x]*self.erosion_factor



    def step(self):
        self.collisions = {}
        self.id_map = np.zeros(shape=(self.height, self.width))

        self.add_drops()
        self.iterate_active_drops()
        self.update_height_map()

        self.erode_heightmap()
        self.smooth_heightmap()
        self.merge_drops()

        self.update_drop_lists()



    def save(self, file_name, height_map, args):

        fformat = args.format
        fname = args.path + file_name

        import PIL
        border = 1

        height_map[0:border] = 0
        height_map[args.height - border:] = 0
        height_map[:, 0:border] = 0
        height_map[:, args.width - border:] = 0

        if fformat == "txt":
            np.savetxt(fname + ".txt", height_map, delimiter=",")

        elif fformat == "png":
            from PIL import Image
            import random
            maximum_drop_size = np.amax(height_map)
            im = PIL.Image.new('RGBA', (args.width, args.height), 0)
            pixels = im.load()
            height_map_copy = np.floor(np.copy(height_map) * 255 / maximum_drop_size).astype(int)

            max = height_map[0,0]
            for y in range(args.height):
                for x in range(args.width):
                    if (height_map[y,x] > max): print(height_map[y,x])
                    pixels[x, y] = (int(height_map_copy[y, x]), int(height_map_copy[y, x]), int(height_map_copy[y, x]))

            im.save(fname + ".png", 'PNG')

        print("File saved to " + fname + "." + fformat)