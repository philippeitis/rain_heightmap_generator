import math
import numpy as np

class Droplet():
    def __init__(self, x, y, mass, surface, velocity = 0):
        self.x = x
        self.y = y
        self.mass = mass
        self.surface = surface

        self.residual_t = 0
        self.active_path = []
        # hemispheres
        self.hemispheres = [(x, y)]
        self.parent_id = None
        # in pixels (radius for each hemisphere
        self.radius = None
        self.radius_sqr = None
        self.velocity = velocity

        self.id = self.surface.curr_drop_id
        self.surface.drop_dict[self.id] = self
        self.surface.curr_drop_id += 1

        if (self.mass > self.surface.m_static) or self.velocity > 0:
            self.surface.active_drops.append(self)
        else:
            self.generate_static_hemispheres()
            self.surface.static_drops.append(self)

        self.update_radius()

    def iterate_position(self):
        # in m/s^2
        acceleration = (self.mass * self.surface.gravity - self.surface.friction)/self.mass
        # in m/s
        self.velocity = self.velocity + acceleration*self.surface.time_step
        # in pixels travelled in one time step
        pixels_travelled = self.velocity * self.surface.one_meter_in_pixels * self.surface.time_step * 10

        self.active_path = [(self.x, self.y)]
        if self.velocity <= 0:
            self.surface.active_drops.remove(self)
            self.generate_static_hemispheres()
            self.update_radius()
            self.surface.static_drops.append(self)
            return
        for step in range(math.floor(pixels_travelled)):
            self.x += np.random.randint(-1, 2)
            self.y += 1
            self.active_path.append((self.x, self.y))

            if (self.get_residual_probability() > np.random.uniform()):
                self.leave_residual_drop()
                self.residual_t = 0
            else: self.residual_t += self.surface.time_step


    def leave_residual_drop(self):
        alpha = np.random.uniform(self.surface.residual_floor, self.surface.residual_ceil)
        new_mass = min(self.surface.m_min, alpha * self.mass)
        new_droplet = Droplet(self.x, self.y, new_mass, self.surface)
        self.mass -= new_mass
        self.update_radius()

    def generate_static_hemispheres(self):
        num_hemispheres = np.random.randint(1, self.surface.max_hemispheres)
        directions = (1, 2, 3, 4)
        next_dirs = directions
        self.hemispheres = [(self.x, self.y)]
        while len(self.hemispheres) < num_hemispheres:
            new_x, new_y = self.hemispheres[-1]
            direction = np.random.choice(next_dirs)
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

            if new_y > self.y:
                self.y = new_y
                self.x = new_x

            if (new_x, new_y) in self.hemispheres:  # To avoid infinite loops
                break
            else:
                self.hemispheres.append((new_x, new_y))


    def update_radius(self):
        self.radius = np.cbrt((3 / 2) / math.pi * (self.mass / len(
            self.hemispheres)) / self.surface.water_density) * self.surface.one_meter_in_pixels
        self.radius = int(math.ceil(self.radius))
        self.radius_sqr = self.radius ** 2

    def get_residual_probability(self):
        assert self.mass > self.surface.m_static or self.velocity > 0

        temp = min(1, self.residual_t / self.surface.residual_t_max)
        return min(1, self.surface.beta * self.surface.time_step / self.surface.residual_t_max * temp)

