from Surface import Surface
import arg_parser

if __name__ == '__main__':

    args = arg_parser.parse_arguments()
    surface = Surface(args)
    surface.add_drops()
    for step in range(args.steps):
        surface.step()
        surface.save("attempt" + str(step), surface.height_map, args)
        print(step)
        #print(len(surface.static_drops), len(surface.active_drops))

    # for drop in surface.drop_dict:
    #     print(surface.drop_dict[drop].x, surface.drop_dict[drop].y, surface.drop_dict[drop].radius)
    # surface.save("attempt1", surface.height_map, args)

