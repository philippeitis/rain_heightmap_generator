## File output
def save(file_name, height_map, id_map, args):
    fformat = args.format
    file_name = args.path + file_name

    import numpy as np
    import math
    import PIL
    border = int(args.border)

    height_map[0:border] = 0
    height_map[args.width - border:] = 0
    height_map[:, 0:border] = 0
    height_map[:, args.height - border:] = 0

    if fformat == "txt":
        np.savetxt(file_name + ".txt", height_map, delimiter=",")

    elif fformat == "png":
        from PIL import Image
        import random
        maximum_drop_size = np.amax(height_map)
        im = PIL.Image.new('RGBA', (args.width, args.height), 0)
        pixels = im.load()
        color_dict = {0 : (255,255,255)}
        for x in range(args.width):
            for y in range(args.height):
                if args.color:
                    if id_map[x, y] not in color_dict.keys():
                        color_dict[id_map[x, y]] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

                    pixel = color_dict[id_map[x, y]]
                    height = height_map[x, y] / maximum_drop_size
                    pixels[x, y] = tuple([math.floor(height * x) for x in pixel])
                else:
                    height = math.floor(height_map[x, y] / maximum_drop_size * 255)
                    pixels[x, y] = (height, height, height)

        if args.show:
            im.show()

        im.save(file_name + ".png", 'PNG')

    elif fformat == "npy":
        np.save(file_name, height_map)

    print("File saved to " + file_name + "." + fformat)


def set_up_directories(args):
    import os
    if args.path != "./":
        try:
            os.mkdir(args.path)
            print("Directory created.")
        except FileExistsError:
            print("Directory already exists.")


def padded_zeros(ref_string, curr_num):
    out_string = str(curr_num)
    while len(out_string) < len(str(ref_string)):
        out_string = "0" + out_string
    return out_string


def generate_time_stamp():
    from datetime import datetime
    now = datetime.now()  # current date and time
    return now.strftime("%m-%d0%Y-%H-%M-%S")


def choose_file_name(args, curr_run):
    if args.name:
        name = args.name
    else:
        name = generate_time_stamp()

    if int(args.runs) > 1:
        name += padded_zeros(args.runs,curr_run)

    return name


def choose_file_name_per_run(args, curr_run):
    if args.name:
        name = args.name
    else:
        name = generate_time_stamp()

    if int(args.runs) > 1:
        name += padded_zeros(args.steps, curr_run)

    return name

