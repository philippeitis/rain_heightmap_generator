def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser(description='Create the height map for rain on a surface.')
    parser.add_argument('steps', type=int, help='Number of simulation steps to run') # around 50 time steps is good

    parser.add_argument('--imw', dest='width', default=720, type=int,
                        help='Sets the width of the height map and the output file.')
    parser.add_argument('--imh', dest='height', default=480, type=int,
                        help='Sets the height of the height map and the output file.')
    parser.add_argument('--scale', dest='scale', default=3, type=float,
                        help='Scale factor of height map')

    parser.add_argument('--w', dest='density_water', default=1000, type=int,
                        help='Sets the density of water, in kg/m^3')

    parser.add_argument('--drops', dest='drops', default=5, type=int,
                        help='Sets the number of drops added to the height map '
                             'each time step.')

    parser.add_argument('--merge_radius', dest='attraction', default=2, type=int,
                        help='Drops will now merge if they are separated by less than n pixels')

    parser.add_argument('--enable_residuals', dest='leave_residuals', action='store_true',
                        help='Enables leaving residual drops')
    parser.add_argument('--disable_residuals', dest='leave_residuals', action='store_false',
                        help='Disables leaving residual drops')
    parser.set_defaults(leave_residuals=True)

    parser.add_argument('--beta', dest='beta', default=3, type=float,
                        help='Sets value b in equation used to determine if drop should be left or not')
    parser.add_argument('--floorval', dest='floor_value', default=1.2, type=float,
                        help='Drops below the given height will be set to zero.')
    parser.add_argument('--residual_floor', dest='residual_floor', default=0.1, type=float,
                        help='Lower bound for mass drops will lose to residual drops.')
    parser.add_argument('--residual_ceil', dest='residual_ceil', default=0.3, type=float,
                        help='Upper bound for mass drops will lose to residual drops.')

    parser.add_argument('--kernel', dest='kernel', default="dwn", type=str, choices=['dwn','avg'],
                        help='Type of kernel used in smoothing step. '
                             '(dwn for downward trending, avg for averaging kernel)')

    parser.add_argument('--dist', dest='dist', default="norm", choices=["norm", "unif", "exp"],
                        help='Distribution used for determining drop masses.')
    parser.add_argument('--mmin', dest='m_min', default=0.000001, type=float,
                        help='Minimum mass of droplets (kg)')
    parser.add_argument('--mavg', dest='m_avg', default=0.000034, type=float,
                        help='Average mass of drops (kg)')
    parser.add_argument('--mdev', dest='m_dev', default=0.000016, type=float,
                        help='Average deviation of drops (kg). Higher '
                             'values create more diverse drop sizes.')
    parser.add_argument('--mmax', dest='m_max', default=0.000240, type=float,
                        help='Maximum mass of droplets (kg)')
    parser.add_argument('--mstatic', dest='m_static', default=None, type=float,
                        help='Sets the mass of static drops. Will override --pstatic if set.')
    parser.add_argument('--pstatic', dest='p_static', default=0.8, type=float,
                        help='Sets the percentage of drops that are static.')

    parser.add_argument('--enable_hemispheres', dest='enable_hemispheres', action='store_true',
                        help='Enables drops with multiple hemispheres (on by default)')
    parser.add_argument('--disable_hemispheres', dest='enable_hemispheres', action='store_false',
                        help='Disables drops with multiple hemispheres (on by default)')
    parser.set_defaults(enable_hemispheres=True)

    parser.add_argument('--numh', dest='max_hemispheres', default=5, type=int,
                        help='Maximum number of hemispheres per drop. '
                             'Performance drops off rapidly after 15 hemispheres.')

    parser.add_argument('--g', dest='g', default=9.81, type=float,
                        help='Gravitational constant (m/s)')

    parser.add_argument('--time', dest='time', default=0.001, type=float,
                        help='Duration of each time step (in seconds)')

    parser.add_argument('--path', dest='path', default="./", type=str,
                        help='Output file path. If not defined, program defaults to same folder.')

    parser.add_argument('--name', dest='name', type=str,
                        help='Output file name. If not defined, program defaults to using date-time string.')

    parser.add_argument('--s', dest='show', action='store_true',
                        help='Show image on program completion.')
    parser.set_defaults(show=False)

    parser.add_argument('--silent', dest='silent', action='store_true',
                        help='Suppress all command line printing.')
    parser.set_defaults(silent=False)

    parser.add_argument('--f', dest='format', default="png", nargs='+', type=str, choices=['png', 'txt', 'npy'],
                        help='Output file format (png, txt, or npy).')

    parser.add_argument('--color', dest='color', default=False, action='store_true',
                        help='Colors in image according to drop ids.')

    parser.add_argument('--border', dest='border', default=0, type=int,
                        help='Sets all values within border pixels of the edge to 0')

    parser.add_argument('--runs', dest='runs', default=1, type=int,
                        help='Will execute the program with the given parameters repeatedly.')
    parser.add_argument('--mt', dest='mt', default=False, action='store_true',
                        help='Enables multithreading for the program. Does not support video')
    parser.add_argument('--verbose', dest='verbose', default="", type=str,
                        help='Will output detailed information on program operation. '
                        't : time to execute each step, '
                        'd : number of droplets in each step, '
                        'a : average mass of droplets in each step.')
    parser.add_argument('--profile', dest='profile', default=False, action='store_true',
                        help='Profiles code if enabled, using the cProfile library. Will save results as data.txt.')
    parser.add_argument('--video', dest='video', default=False, action='store_true',
                        help='Saves series of images as video.')
    parser.add_argument('--video_load', dest='video_load', default=False, action='store_true',
                        help="In the case that the program crashes or you end execution early,"
                             "this option will take the images in temp and finish making the video.")
    parser.add_argument('--graph', dest='graph', default=False, action='store_true',
                        help="Will produce graphs of average drop mass and time to compute each step.")

    args = parser.parse_args()
    return args
