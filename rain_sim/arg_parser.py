def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser(description='Create the height map for rain on a surface.')
    parser.add_argument('steps', type=int, help='Number of simulation steps to run') # around 50 time steps is good

    parser.add_argument('--width', dest='width', default=720, type=int,
                        help='Sets the width of the height map and the output file.')
    parser.add_argument('--height', dest='height', default=480, type=int,
                        help='Sets the height of the height map and the output file.')
    parser.add_argument('--scale', dest='scale', default=0.3, type=float,
                        help='Width of image corresponds to this number of meters.')

    parser.add_argument('--drops_per_iter', dest='drops_per_iter', default=2, type=int,
                        help='Sets the number of drops added to the height map '
                             'each time step.')

    parser.add_argument('--beta', dest='beta', default=0.5, type=float,
                        help='Sets value b in equation used to determine if drop should be left or not')
    parser.add_argument('--floor_val', dest='floor_val', default=1.2, type=float,
                        help='Drops below the given height will be set to zero.')
    parser.add_argument('--residual_floor', dest='residual_floor', default=0.1, type=float,
                        help='Lower bound for mass drops will lose to residual drops.')
    parser.add_argument('--residual_ceil', dest='residual_ceil', default=0.3, type=float,
                        help='Upper bound for mass drops will lose to residual drops.')

    parser.add_argument('--dist', dest='dist', default="exp", choices=["normal", "uniform"],
                        help='Distribution used for determining drop masses.')
    parser.add_argument('--m_min', dest='m_min', default=0.000001, type=float,
                        help='Minimum mass of droplets (kg)')
    parser.add_argument('--m_max', dest='m_max', default=0.000240, type=float,
                        help='Maximum mass of droplets (kg)')
    parser.add_argument('--p_static', dest='p_static', default=0.5, type=float,
                        help='Sets the percentage of drops that are static.')
    parser.add_argument('--m_static', dest='m_static', default=0.00008, type=float,
                        help='Sets the mass of drops that are static.')

    parser.add_argument('--max_hem', dest='max_hemispheres', default=5, type=int,
                        help='Maximum number of hemispheres per drop. '
                             'Performance drops off rapidly after 15 hemispheres.')

    parser.add_argument('--time_step', dest='time_step', default=0.001, type=float,
                        help='Duration of each time step (in seconds)')

    parser.add_argument('--path', dest='path', default="./", type=str,
                        help='Output file path. If not defined, program defaults to same folder.')

    parser.add_argument('--name', dest='name', type=str,
                        help='Output file name. If not defined, program defaults to using date-time string.')

    parser.add_argument('--f', dest='format', default="png", type=str, choices=['png', 'txt', 'npy'],
                        help='Output file format (png, txt, or npy).')

    args = parser.parse_args()
    return args
