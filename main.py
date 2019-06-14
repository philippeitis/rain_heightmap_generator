def call_sp(queue, args):
    import single_processing as sp
    while not queue.empty():
        curr_run = queue.get()
        sp.main(args, True, curr_run)


if __name__ == '__main__':
    import arg_parser as ap
    import file_ops as fo
    import single_processing as sp
    args = ap.parse_arguments()

    fo.set_up_directories(args)

    if args.runs >= 1 and args.mt:
        args.show = False

        import multiprocessing
        from multiprocessing import Process, Queue
        processes = []
        q = Queue()
        for i in range(int(args.runs)):
            q.put(i)
        for i in range(min(args.runs, multiprocessing.cpu_count()-1)):
            p = Process(target=call_sp, args=(q, args))
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
    else:
        sp.main(args, False)
