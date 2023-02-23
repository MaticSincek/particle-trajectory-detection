import random
import math
from PIL import Image, ImageDraw
import numpy as np

def draw_point(image, xy, size, color):
    if size % 2 != 1:
        width = int(size / 2) + 1
    else:
        width = size / 2
    bbox = [(xy[0] - width, xy[1] - width), (xy[0] + width, xy[1] + width)]
    image.ellipse(bbox, fill= color)

def draw_concentric(image, origin, r, w):
    bbox = [(origin[0] - r, origin[1] - r), (origin[0] + r, origin[1] + r)]
    image.ellipse(bbox, outline = 'gray', width = w)

# how many sensor should be on layer proportional to the circumference of the layer
def number_of_sensors_on_layer(original_r, r, sensor_density):
    return sensor_density * (r/original_r)

def circle_from_points(p1, p2, p3):
    """
    Returns the center and radius of the circle passing the given 3 points.
    In case the 3 points form a line, returns (None, infinity).
    """
    temp = p2[0] * p2[0] + p2[1] * p2[1]
    bc = (p1[0] * p1[0] + p1[1] * p1[1] - temp) / 2
    cd = (temp - p3[0] * p3[0] - p3[1] * p3[1]) / 2
    det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])

    if abs(det) < 1.0e-6:
        return (None, np.inf)

    # Center of circle
    cx = (bc*(p2[1] - p3[1]) - cd*(p1[1] - p2[1])) / det
    cy = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det

    radius = np.sqrt((cx - p1[0])**2 + (cy - p1[1])**2)
    return ((cx, cy), radius)

W = 1500
H = 1500
N_CONCENTRIC = 5
N_TRAJECTORIES = 10
SENSOR_DENSITY = 10

#trajectory info
#radius of trajectory
radii = []
#angle of trajectory center from origin
angles = []
#negative or positive magnetic effect
directions = []
#center of circle the curve is based on
centers = []
#list of points where the particle colided with the sensors
detections = []

origin = (W/2, H/2)

#minimum and maximum radius of the trajectories
rmin = int(W * 2 / 3 / 2)
rmax = W * 2 / 3 * 2

img = Image.new("RGB", (W, H))
canvas = ImageDraw.Draw(img)

draw_point(canvas, origin, 6, "white")

concentric_step = int(W / 2 / 5)
layer_radii = [r for r in range(concentric_step, int(W/2), concentric_step)]

#detector layer drawing and sensor generation
for r in layer_radii:
    draw_concentric(canvas, origin, r, 3)

    number_of_sensors = number_of_sensors_on_layer(concentric_step, r, SENSOR_DENSITY)

    for angle in [a / 1000 for a in range(0, 360 * 1000, int(360 * 1000 / number_of_sensors))]:
        x = origin[0] + int(r * math.cos(math.pi * 2 * angle / 360));
        y = origin[1] + int(r * math.sin(math.pi * 2 * angle / 360));
        
        #draw_point(canvas, (x,y), 10, "green")

#generation of arcs
for i in range(N_TRAJECTORIES):
    angle = random.randint(0, 359)
    r = radius = random.randint(rmin, rmax)
    dir = random.randint(0,1)
    if dir == 1:
        astart = angle
        aend = (angle + 180) % 360
    else:
        astart = (360 + angle - 180) % 360
        aend = angle

    x = origin[0] + int(r * math.cos(math.pi * 2 * angle / 360));
    y = origin[1] + int(r * math.sin(math.pi * 2 * angle / 360));
    bbox = [(x - r, y - r), (x + r, y + r)]

    radii.append(r)
    angles.append(angle)
    directions.append(dir)
    centers.append((x,y))

    canvas.arc(
        bbox, 
        start = astart, 
        end = aend, 
        fill = (
            random.randint(50,200), 
            random.randint(50,200), 
            random.randint(50,200)
            ),
        width=3)

    detection = []
    for d2 in layer_radii:
        d = math.sqrt((x - origin[0]) ** 2 + (y - origin[1]) ** 2)
        a = (d2 ** 2 - r ** 2 + d ** 2) / (2 * d)
        h = math.sqrt(d2 ** 2 - a ** 2)
        x2 = origin[0] + a * (x - origin[0]) / d   
        y2 = origin[1] + a * (y - origin[1]) / d

        if dir == 0:  
            x3 = x2 + h * (y - origin[1]) / d
            y3 = y2 - h * (x - origin[0]) / d
        else:
            x3 = x2 - h * (y - origin[1]) / d
            y3 = y2 + h * (x - origin[0]) / d

        detection.append((x3,y3))
        draw_point(canvas, (x3,y3), 7, "red")

    detections.append(detection)



#sorting detections by layer
detections_on_layer = [[] for _ in layer_radii]

for i in range(N_TRAJECTORIES):
    for lr in range(len(layer_radii)):
        detections_on_layer[lr].append(detections[i][lr])

#finding the trajectories from points by combinatorics
for p0 in detections_on_layer[0]:
    for p1 in detections_on_layer[1]:
        for p2 in detections_on_layer[2]:

            #find the r and center of these 3 points
            (center, r) = circle_from_points(p0, p1, p2)

            x = 0 + center[0];
            y = 0 + center[1];
            bbox = [(x - r, y - r), (x + r, y + r)]

            canvas.arc(
                bbox, 
                start = 0, 
                end = 340, 
                fill = "red",
                width=3)
            
            break
        break
    break

img.show()