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
    from src import file_ops as fo
    import math
    for i in range(args.runs):
        surface = surface.Surface(args, curr_run=i)

        for j in range(int(args.steps)):
            string = surface.step()
            if not args.silent:
                print(string)
            if args.video:
                surface.save_temp()

        for i in range(math.ceil(args.width / 720)):
            surface.blur_masked()
        if not args.video:
            surface.save()
        else:
            fo.save_as_video("./temp/", fo.choose_file_name(args, i))
