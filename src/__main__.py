if __name__ == '__main__':
    from src import arg_parser as ap, file_ops as fo
    from src import single_processing as sp

    args = ap.parse_arguments()

    fo.set_up_directories(args)

    if not args.profile:
        if not args.name:
            args.name = fo.generate_time_stamp()

        if args.runs > 1 and args.mt:
            sp.multi_processing(args)

        else:
            sp.single_process(args)
    else:
        import cProfile
        cProfile
        cProfile.run('sp.single_process(args)',"data.txt")
        import pstats
        p = pstats.Stats('data.txt')
        p.sort_stats('cumulative').print_stats(10)
