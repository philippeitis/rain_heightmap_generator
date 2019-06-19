## File output
def save(file_name, height_map, id_map, args):
    ## TODO: add functionality for saving specific args used as metadata (eg. json)

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
        color_dict = {0:(255, 255, 255)}
        for y in range(args.height):
            for x in range(args.width):
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


def save_temp(height_map, id_map, color_dict, args, curr_step):
    ## TODO: add functionality for saving specific args used as metadata (eg. json)

    file_name = "./temp/" + padded_zeros(args.steps,curr_step) + ".png"

    import numpy as np
    import math
    import PIL
    border = int(args.border)

    height_map[0:border] = 0
    height_map[args.width - border:] = 0
    height_map[:, 0:border] = 0
    height_map[:, args.height - border:] = 0

    from PIL import Image
    import random
    maximum_drop_size = np.amax(height_map)
    im = PIL.Image.new('RGBA', (args.width, args.height), 0)
    pixels = im.load()

    for x in range(args.width):
        for y in range(args.height):
            if args.color:
                if id_map[x, y] not in color_dict.keys():
                    color_dict[id_map[x, y]] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

                pixel = color_dict[id_map[x, y]]
                height = height_map[y, x] / maximum_drop_size
                pixels[x, y] = tuple([math.floor(height * x) for x in pixel])
            else:
                height = math.floor(height_map[x, y] / maximum_drop_size * 255)
                pixels[x, y] = (height, height, height)
    im.save(file_name, 'PNG')


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
        name = args.path + args.name
    else:
        name = args.path + generate_time_stamp()

    if int(args.runs) > 1:
        name += padded_zeros(args.steps, curr_run)

    return name

def save_as_video(folder_name,file_name):
    import cv2
    import os

    images = [img for img in os.listdir(folder_name) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(folder_name, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(file_name + ".avi", 0, 24, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(folder_name, image)))

    cv2.destroyAllWindows()
    video.release()
    print("File saved to " + file_name + ".avi")
    clear_temp()


def clear_temp():
    import os
    filelist = [f for f in os.listdir("./temp") if f.endswith(".png")]
    for f in filelist:
        os.remove(os.path.join("./temp", f))

