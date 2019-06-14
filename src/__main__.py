if __name__ == '__main__':
    from src import arg_parser as ap, file_ops as fo

    args = ap.parse_arguments()

    fo.set_up_directories(args)
    if args.runs >= 1 and args.mt:
        from src import multi_processing as mp
        args.show = False
        mp.main(args)

    else:
        import src.single_processing as sp
        sp.main(args, False)
