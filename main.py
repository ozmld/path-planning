import numpy as np
import cv2
from ObjectDetector import ObjectDetector
from MapBuilder import MapBuilder
from GUI import GUI, Param
from tkinter import *
import time
from graphs import Graph
import keyboard

global lower_bound, upper_bound, detalization, approximation_coef, morphology_coef


def gr():
    if 1 == 1:
        path, mp = graph.search_path_dijkstra([p1[1], p1[0], 1], [p2[1], p2[0], 1])
        # path, mp = graph.search_path_random()
        # path, mp = self.graph.astar((p1[1], p1[0]), (p2[1], p2[0]), True)
        return path, mp
    else:
        print("no path")
        return None, None


def on_params_change(ind, val):
    #print("changing params, new val", val)
    if (ind <= Param.LOWER_BLUE):
        lower_bound[ind] = int(255 * val)
        gui.lower_bound = lower_bound
    elif (ind <= Param.UPPER_BLUE):
        upper_bound[ind - Param.UPPER_RED] = int(255 * val)
        gui.upper_bound = upper_bound
    elif (ind == Param.DETALIZATION):
        detalization = int(20 * val) + 10
    elif (ind == Param.APPROXIMATION_COEF):
        approximation_coef = 0.1 * val
        gui.approximation_coef = approximation_coef
    elif (ind == Param.MORPHOLOGY_COEF):
        morphology_coef = int(30 * val)
        gui.morphology_coef = morphology_coef


def gui_init():
    gui.detalization = detalization
    gui.map_builder = map_builder
    gui.lower_bound = lower_bound
    gui.upper_bound = upper_bound
    gui.approximation_coef = approximation_coef
    gui.morphology_coef = morphology_coef


def map_init():
    contours_frame, surface, objects = ObjectDetector.detect_object(frame, lower_bound, upper_bound,
                                                                    approximation_coef, morphology_coef)
    map, scale = map_builder.build_map(frame, objects)
    gui.map = map
    gui.map_builder = map_builder


# cam1 = cv2.VideoCapture('1234.mov')
frame = cv2.imread("ggwp.png")
lower_bound = np.array([0, 0, 0])
upper_bound = np.array([255, 255, 255])
detalization = 10
approximation_coef = 0.0
morphology_coef = 11
k = 0

map_builder = MapBuilder()
gui = GUI(on_params_change)
gui_init()

graph = Graph()

files = ['frame_10']  # 0,2,5,7,
filename = 'frame_10'
frame = cv2.imread('./' + filename + '.png')
frame = frame[:480, :, :]
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

map_init()
graph.gen_graph(gui.map, detalization)
gui.graph = graph

imgp1 = (30, 190)
imgp2 = (590, 310)
p1 = map_builder.getPoint(imgp1, 0)
p2 = map_builder.getPoint(imgp2, 0)
gui.p1 = p1
gui.p2 = p2

path, mp = gr()
gui.path = path
gui.mp = mp
print(mp)
while keyboard.is_pressed('space') == False:
    # _, frame = cam1.read()

    # frame update
    frame = cv2.imread('./' + filename + '.png')
    frame = frame[:480, :, :]
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gui.frame = frame

    if gui.map is None:
        print(1488)
        gui.set_image(frame)
        gui.root.update()
        time.sleep(0.01)
        continue

    mp = np.zeros(frame.shape, np.uint8)

    if gui.flag == 1:
        # t = time.time()
        # start and finish points update
        # p1 = gui.map_builder.getPoint(imgp1, 0)
        # p2 = gui.map_builder.getPoint(imgp2, 0)
        # gui.p1 = p1
        # gui.p2 = p2

        gui.map_builder = map_builder
        path = gui.path
        mp = gui.mp

        print("mp type", type(mp), type(path))
        if path is None and mp is None:
            print("path and mp None")
        else:
            # drawing start and finish points on real word frame
            frame = cv2.circle(frame, imgp1, 5, (255, 0, 0), 2)
            frame = cv2.circle(frame, imgp2, 5, (0, 0, 255), 2)

            # drawing points on black&white map
            mp = cv2.circle(mp, (p1[0], p1[1]), 5, (255, 0, 0), 2)
            mp = cv2.circle(mp, (p2[0], p2[1]), 5, (0, 0, 255), 2)



            # drawing path on real word frame
            for i in range(1, len(path)):
                point1 = path[i - 1]
                point2 = path[i]
                img_point1 = gui.map_builder.getPointBack(point1)
                img_point2 = gui.map_builder.getPointBack(point2)
                cv2.line(gui.frame, img_point1, img_point2, (0, 255, 0), 3)
                cv2.line(mp, point1, point2, (0, 255, 0), 3)

            # show real word frame and map in gui with paths
            mp = np.flip(mp, axis=0)
            res = np.concatenate((frame, mp), axis=1)
            gui.set_image(res)
            gui.root.update()

            print("OK")
        # print(time.time()-t, "total end")
        print(lower_bound, k)
        frame = cv2.imread('./' + filename + '.png')
        frame = frame[:480, :, :]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        gui.flag = 0
    time.sleep(0.01)
    gui.root.update()

    # cv2.imshow('tk', frame)
    # print(frame.shape)
    # time.sleep(0.01)
    # cv2.imwrite(filename + '-initial_picture.png', frame)
    # break
    # cv2.imwrite(filename + '-contours.png', contours_frame)
    # cv2.imwrite(filename + '-pic2r.png', q)
    # cv2.imwrite(filename + '-map.png', map)
    # break
