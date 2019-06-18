def call_sp(queue, args):
    from src import surface
    while not queue.empty():
        surface = surface.Surface(args, True, queue.get())
        for j in range(int(args.steps)):
            surface.step()
        surface.save()


def multi_processing(args):
    import multiprocessing
    from multiprocessing import Process, Queue

    args.show = False

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


def single_process(args):
    from src import surface
    runs = args.runs

    for i in range(runs):
        surface = surface.Surface(args, curr_run=i)

        for j in range(int(args.steps)):
            string = surface.step()
            print(string)

        surface.blur_masked()

        surface.save()


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
def merge_drops():
        ...
        low_drop.velocity = new_velocity
        low_drop.set_mass(low_drop.mass + high_drop.mass)
'''

