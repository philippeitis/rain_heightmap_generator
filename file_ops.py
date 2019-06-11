## File output
def save(file_name, height_map, args):
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
        print("File saved to " + file_name + ".txt")

    elif fformat == "png":
        from PIL import Image
        maximum_drop_size = np.amax(height_map)
        im = PIL.Image.new('RGBA', (args.width, args.height), 0)
        pixels = im.load()
        for x in range(args.width):
            for y in range(args.height):
                pixel_val = math.floor(height_map[x, y] / maximum_drop_size * 255)
                pixels[x, y] = (pixel_val, pixel_val, pixel_val)

        if int(args.show) == 1:
            im.show()

        im.save(file_name + ".png", 'PNG')
        print("File saved to " + file_name + ".png")


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
