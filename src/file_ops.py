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

    if "txt" in fformat:
        np.savetxt(file_name + ".txt", height_map, delimiter=",")
        print("File saved to " + file_name + ".txt")

    if "png" in fformat:
        from PIL import Image
        import random
        maximum_drop_size = np.amax(height_map)
        im = PIL.Image.new('RGBA', (args.width, args.height), 0)
        pixels = im.load()
        color_dict = {0:(255, 255, 255)}
        if not args.color:
            height_map_copy = np.floor(np.copy(height_map) * 255 / maximum_drop_size).astype(int)
            for y in range(args.height):
                for x in range(args.width):
                    pixels[x, y] = (height_map_copy[x, y], height_map_copy[x, y], height_map_copy[x, y])
        else:
            height_map_copy = np.copy(height_map) / maximum_drop_size
            for y in range(args.height):
                for x in range(args.width):
                    if args.color:
                        if id_map[x, y] not in color_dict.keys():
                            color_dict[id_map[x, y]] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                        pixel = color_dict[id_map[x, y]]
                        pixels[x, y] = tuple([math.floor(height_map_copy[x,y] * x) for x in pixel])
        if args.show:
            im.show()
        im.save(file_name + ".png", 'PNG')
        print("File saved to " + file_name + ".png")

    if "npy" in fformat:
        np.save(file_name, height_map)

        print("File saved to " + file_name + ".npy")


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
                height = height_map[x, y] / maximum_drop_size
                pixels[x, y] = tuple([math.floor(height * x) for x in pixel])
            else:
                height = math.floor(height_map[x, y] / maximum_drop_size * 255)
                pixels[x, y] = (height, height, height)
    im.save(file_name, 'PNG')
    return color_dict


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

    video = cv2.VideoWriter(file_name + ".mp4", cv2.VideoWriter_fourcc(*'H264'), 24, (width, height))

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


def generate_graph(file_name, graph_data):
    import numpy as np
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots()
    t = np.arange(1, len(graph_data) + 1, 1)
    s1 = [row[0] for row in graph_data]
    ax1.plot(t, s1, 'b-')
    ax1.set_xlabel('Step #')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('Time (s)', color='b')
    ax1.tick_params('y', colors='b')

    ax2 = ax1.twinx()
    s2 = [row[1] for row in graph_data]
    ax2.plot(t, s2, 'r.')
    ax2.set_ylabel('# of Drops', color='r')
    ax2.tick_params('y', colors='r')

    fig.tight_layout()

    fig.savefig(file_name + "graph" + ".png")
