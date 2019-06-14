class Droplet:
    def __init__(self, x, y, mass, velocity):
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


'''
def add_drops(height_map, active_drops, drop_array, new_drops, args):
    for x in range(args.drops):
        mass = min(args.m_max,max(args.m_min, np.random.normal(args.m_avg, args.m_dev, 1)))
        drop_to_add = Droplet(random.randint(0, args.width), random.randint(0, args.height), mass, 0,
                              height_map, active_drops, drop_array, new_drops, args)

def merge_drops():
    intersecting_drops = []

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
'''


def call_sp(queue, args):
    import src.single_processing as sp
    while not queue.empty():
        curr_run = queue.get()
        sp.main(args, True, curr_run)


def main(args):
    import multiprocessing
    from multiprocessing import Process, Queue
    processes = []
    q = Queue()
    for i in range(int(args.runs)):
        q.put(i)
    for i in range(min(args.runs, multiprocessing.cpu_count() - 1)):
        p = Process(target=call_sp, args=(q, args))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()
