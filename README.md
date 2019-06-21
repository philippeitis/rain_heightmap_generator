# rain_heightmap_generator
This is an implementation of the process outlined by "A heuristic approach to the simulation of water drops and flows on glass panes", by Kai-Chun Chen, Pei-Shan Chen, and Sai-Keung Wong, for the generation of rain on glass panes. The abstract can be found at https://www.sciencedirect.com/science/article/pii/S0097849313001295.

Quick Start:
Call the code with 
python -m src steps
Where steps is the number of steps the simulation will run.

In general, the image width is assumed to correspond to 1 meter. To change this, use --scale n, where n corresponds to an image width of 1m/n (so --scale 3 would correspond to an image width of 33cm). 
For more detailed operation, --verbose can be used with strings "a": average drop mass, "d": total drops and drops in motion, and "t": time per simulation step, or any of these strings in combination. Additionally, to profile the code, the flag --profile can be used in combination with preferred settings to see what methods are consuming the most time. Graphs can also be generated with --graph, which will show the time each step took and the number of drops present in each step.

Additionally, files can be output as either png, txt files (delimited by ","), or numpy's npy files, or any combination of the three with --f png txt npy. It is also possible to output videos with the --video flag, and to see drops colored based on their ids (which also works for the --video flag), use the --color flag.
