import random
from decimal import *
import math
from PIL import Image, ImageDraw
import numpy as np
import sys

random.seed(5)

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

def circle_from_points(p1, p2, p3):
    """
    Returns the center and radius of the circle passing the given 3 points.
    In case the 3 points form a line, returns (None, infinity).
    """
    temp = p2[0] * p2[0] + p2[1] * p2[1]
    bc = (p1[0] * p1[0] + p1[1] * p1[1] - temp) / 2
    cd = (temp - p3[0] * p3[0] - p3[1] * p3[1]) / 2
    det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])

    # colinear, return none
    if abs(det) < 1.0e-6:
        return (None, np.inf)

    # center of circle
    cx = (bc*(p2[1] - p3[1]) - cd*(p1[1] - p2[1])) / det
    cy = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det

    radius = np.sqrt((cx - p1[0])**2 + (cy - p1[1])**2)
    return ((cx, cy), radius)

def angle_of_point_relative_to_origin(x, y):
    angle_rad = math.atan2(y, x)
    angle_rad = (angle_rad + 2 * math.pi) % (2 * math.pi)
    return angle_rad

def get_orientation(p1, p2, p3):
	val = (float(p2[1] - p1[1]) * (p3[0] - p2[0])) - \
		(float(p2[0] - p1[0]) * (p3[1] - p2[1]))
	if (val > 0):
		# clockwise orientation
		return 1
	elif (val < 0):
		# anti-clockwise orientation
		return 0
	else:
		# collinear orientation
		return 0
    
def cartesian2polar(p):
    x,y = p
    angle = angle_of_point_relative_to_origin(x, y)
    distance = math.sqrt(x ** 2 + y ** 2)
    return (distance, angle)

def polar2cartesian(distance, angle):
    x = distance * math.cos(angle)
    y = distance * math.sin(angle)
    return (x,y)

def centralize_point_on_sensor(p, sensor_segment_angle, layer_list, layer):
    distance, angle = cartesian2polar(p)
    new_angle = (int(angle / sensor_segment_angle) * sensor_segment_angle) + sensor_segment_angle / 2
    new_distance = layer_list[layer]
    return polar2cartesian(new_distance, new_angle)

def random_point_on_sensor(p, sensor_segment_angle):
    distance, angle = cartesian2polar(p)
    a_min = int(angle / sensor_segment_angle) * sensor_segment_angle
    a_max = (int(angle / sensor_segment_angle) + 1) * sensor_segment_angle
    a_delta = a_max - a_min
    a = a_min + random.random() * a_delta
    return polar2cartesian(distance, a)

def scale_for_drawing (x):
    return W/2 / 10000 * x


W = 1500
H = 1500
realW = 20000
realH = 20000
N_CONCENTRIC = 23
N_TRAJECTORIES = 3
SENSOR_DENSITY = 3600
N_SEED_CORRECTIONS = 30 ** 2
TOLERANCE = 50 ** 2
CENTER_TOLERANCE = 10
TRAJECTORY_ANGLE_TOLERANCE = math.pi / 4 # originally 50deg, now 45deg
SEED_ANGLE_TOLERANCE = math.pi / 18 # 10deg
MIN_PERC_COVERAGE_FOR_TRAJ = 0.9
DETECTION_FAIL_RATE = 0
WITH_SENSORS = True

# trajectory info
# radius of trajectory
radii = []
# angle of trajectory center from origin; absolete if we have the center point
angles = []
# negative or positive magnetic effect
directions = []
# center of circle the curve is based on
centers = []
# list of points where the particle colided with the sensors
detections = []

origin = (W/2, H/2)

# minimum and maximum radius of the trajectories
rmin = realW * 2 / 3 / 2
rmax = realW * 2 / 3

img = Image.new("RGB", (W, H))
canvas = ImageDraw.Draw(img)

draw_point(canvas, origin, 6, "white")

layer_radii = [400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000]

# detector layer drawing and sensor generation
for r in layer_radii:
    draw_concentric(canvas, origin, scale_for_drawing(r), 3)

# generation of arcs
for i in range(N_TRAJECTORIES):
    angle = random.uniform(0, 2 * math.pi)
    r = radius = random.uniform(rmin, rmax)
    dir = random.randint(0,1)

    # we should always only draw half of the circle
    angledeg = math.degrees(angle)
    if dir == 1:
        astart = angledeg
        aend = (angledeg + 180) % 360
    else:
        astart = (360 + angledeg - 180) % 360
        aend = angledeg

    # points r away from the origin with an angle "angle" relative to the origin 
    x = r * math.cos(angle)
    y = r * math.sin(angle)

    radii.append(r)
    angles.append(angle)
    directions.append(dir)
    centers.append((x,y))

    x_draw = scale_for_drawing(x)
    y_draw = scale_for_drawing(y)
    r_draw = scale_for_drawing(r)

    # center of bbox is in (x,y) with corners r away from the center in both directions
    bbox = [(x_draw - r_draw + origin[0], y_draw - r_draw + origin[1]), (x_draw + r_draw + origin[0], y_draw + r_draw + origin[1])]

    canvas.arc(
        bbox, 
        start = astart, 
        end = aend, 
        fill = (
            random.randint(50,200), 
            random.randint(50,200), 
            random.randint(50,200)
            ),
        width = 5)

    # we already have the trajectory now let's see where it intersects each concentric circle
    detection = []
    for radius in layer_radii:
        d = math.sqrt(x ** 2 + y ** 2)
        a = (radius ** 2 - r ** 2 + d ** 2) / (2 * d)
        h = math.sqrt(radius ** 2 - a ** 2)
        x2 = a * x / d   
        y2 = a * y / d

        if dir == 0:  
            x3 = x2 + h * y / d
            y3 = y2 - h * x / d
        else:
            x3 = x2 - h * y / d
            y3 = y2 + h * x / d

        detection.append((x3,y3))

        if not WITH_SENSORS:
            x3_real = scale_for_drawing(x3) + origin[0]
            y3_real = scale_for_drawing(y3) + origin[1]
            draw_point(canvas, (x3_real,y3_real), 7, "red")

    detections.append(detection)

# sorting detections by layer
detections_on_layer = [[] for _ in layer_radii]

for i in range(N_TRAJECTORIES):
    for lr in range(len(layer_radii)):
        detections_on_layer[lr].append(detections[i][lr])

#determine the angle span of sensors
segment_angle = 2 * math.pi / SENSOR_DENSITY

#centralize points on sensors
if WITH_SENSORS:
    for i in range(len(detections_on_layer)):
        for j in range(len(detections_on_layer[i])):
            p = centralize_point_on_sensor(detections_on_layer[i][j], segment_angle, layer_radii, i)
            detections_on_layer[i][j] = (float(p[0]), float(p[1]))

# remove points that translated to the same sensor to avoid duplicate trajectories
for i in range(len(detections_on_layer)):
    detections_on_layer[i] = list(dict.fromkeys(detections_on_layer[i]))

marked_for_deletion = []

#delete some points to simulate the real environment whene not every point on trajectory is detected
#important that we start from the back of the rows for deleting - otherwise we can get indexes that can't be deleted
if WITH_SENSORS:
    for i in range(len(detections_on_layer)-1, -1, -1):
        for j in range(len(detections_on_layer[i])-1, -1, -1):
            chance = random.random()
            if chance <= DETECTION_FAIL_RATE:
                marked_for_deletion.append((i,j))
            else:
                x_draw = scale_for_drawing(detections_on_layer[i][j][0]) + origin[0]
                y_draw = scale_for_drawing(detections_on_layer[i][j][1]) + origin[1]
                draw_point(canvas, (x_draw, y_draw), 6, "yellow")
                
for (i,j) in marked_for_deletion:
    del detections_on_layer[i][j]
              
# radius of trajectory
trajectory_radii = []
# negative or positive magnetic effect
trajectory_directions = []
# center of circle the curve is based on
trajectory_centers = []
# points used in trajectory
trajectory_points = []

# number of points we need to find on the seed trajectory for us to count it as an actual trajectory
points_needed = int(N_CONCENTRIC * MIN_PERC_COVERAGE_FOR_TRAJ)

# finding the trajectories from points by combinatorics
for p0 in detections_on_layer[N_CONCENTRIC-1]:
    for p1 in detections_on_layer[N_CONCENTRIC-2]:
        for p2 in detections_on_layer[N_CONCENTRIC-3]:

            # p0 = actual sensor; pp0 = randomly generated point in sensor space

            r_best = None
            pp0_best = None
            pp1_best = None
            pp2_best = None
            center_best = None

            min_avg_error = sys.float_info.max

            angle_p0 = angle_reference = angle_of_point_relative_to_origin(p0[0], p0[1])
            angle_p1 = angle_of_point_relative_to_origin(p1[0], p1[1])
            angle_p2 = angle_of_point_relative_to_origin(p2[0], p2[1])

            # if both points are roughly in the same direction from the origin we continue
            if (abs(angle_p0 - angle_p1) < SEED_ANGLE_TOLERANCE or \
               abs(angle_p0 - angle_p1) > (2 * math.pi - SEED_ANGLE_TOLERANCE)) and \
               (abs(angle_p0 - angle_p2) < SEED_ANGLE_TOLERANCE or \
               abs(angle_p0 - angle_p2) > (2 * math.pi - SEED_ANGLE_TOLERANCE)):
                
                # we generate random points in sensos space and figure out which one fits best
                for _ in range(N_SEED_CORRECTIONS):
                    pp0 = random_point_on_sensor(p0, segment_angle)
                    pp1 = random_point_on_sensor(p1, segment_angle)
                    pp2 = random_point_on_sensor(p2, segment_angle)

                    # find the r and center of these 3 points
                    (center, r) = circle_from_points(pp0, pp1, pp2)

                    if center == None:
                        continue

                    # check if point goes roughly through the center
                    distance_center_origin = math.sqrt(center[0] ** 2 + center[1] ** 2)
                    center_error = abs(distance_center_origin-r)
                    
                    if center_error > CENTER_TOLERANCE:
                        continue
                    
                    # variables that hold how many points we found and cumulative distance of points to sensor centers
                    points_on_seed_trajectory = 0
                    cumul_error = 0
                    
                    # if r makes sense it could be a trajectory
                    if r < rmax and r > rmin:

                        # check how the seed is supported by points by goung through all the layers and finding the best fit on every layer
                        for layer in range(N_CONCENTRIC-1, -1, -1):
                            min_error = sys.float_info.max

                            for det in range(len(detections_on_layer[layer])):
                                p = detections_on_layer[layer][det]

                                # calculate distance from the seed center to point to see if it is on the trajectory
                                distance_center_current_point = math.sqrt((center[0] - p[0]) ** 2 + (center[1] - p[1]) ** 2)

                                # if distance from the seed center to point is approx. the same as r 
                                # and the angle from origin is similar it is on the trajectory of seed s
                                error = abs(distance_center_current_point-r) ** 2
                                if error < TOLERANCE:
                                    detection_angle = angle_of_point_relative_to_origin(p[0], p[1])

                                    if (abs(angle_reference - detection_angle) < TRAJECTORY_ANGLE_TOLERANCE) or \
                                        abs(angle_reference - detection_angle) > (2 * math.pi - TRAJECTORY_ANGLE_TOLERANCE):

                                        # if it's the best fit as of now save the error as minimal
                                        if error < min_error:
                                            min_error = error

                            # if we ever found a point that is close, then means we found the closest one then add it to the cumulative
                            if (min_error < sys.float_info.max):
                                cumul_error += min_error
                                points_on_seed_trajectory += 1

                    # calculate the average aquared error. If it has at least the amount of points we need and the lowest error yes, save it as the best
                    if points_on_seed_trajectory >= points_needed:
                        avg_err = cumul_error / points_on_seed_trajectory
                        if avg_err < min_avg_error:
                            min_avg_error = avg_err
                            r_best = r
                            pp0_best = pp0
                            pp1_best = pp1
                            pp2_best = pp2
                            center_best = center

            # if we found at least one trajectory we can be sure its the best one and we can save it
            if center_best != None:
                o = get_orientation(pp2_best, pp1_best, pp0_best) 
                trajectory_radii.append(r_best)
                trajectory_centers.append(center_best)
                trajectory_directions.append(o)
                trajectory_points.append((pp0_best, pp1_best, pp2_best))

print("Found " + str(len(trajectory_radii)) + " out of " + str(N_TRAJECTORIES) + " trajectories." )

# draw the trajectories
for i in range(len(trajectory_radii)):
    r = trajectory_radii[i]
    dir = trajectory_directions[i]
    center = trajectory_centers[i]
    angle = angle_of_point_relative_to_origin(center[0], center[1])

    # we should always only draw half of the circle
    angledeg = math.degrees(angle)
    if dir == 1:
        astart = angledeg
        aend = (angledeg + 180) % 360
    else:
        astart = (360 + angledeg - 180) % 360
        aend = angledeg

    r_draw = scale_for_drawing(r)
    center_draw = (scale_for_drawing(center[0]) + origin[0], scale_for_drawing(center[1]) + origin[1])

    # calculate bbox from center and r
    bbox = [(center_draw[0] - r_draw, center_draw[1] - r_draw), (center_draw[0] + r_draw, center_draw[1] + r_draw)]

    canvas.arc(
        bbox, 
        start = astart, 
        end = aend, 
        fill = (255, 255, 255),
        width = 2)
    
# for layer in detections_on_layer:
#     print("{ ", end = "")
#     for detection in layer:
#         print(str(detection[0]) + ", ", end = "")
#     print("}, ")
    
# print("")

# for layer in detections_on_layer:
#     print("{ ", end = "")
#     for detection in layer:
#         print(str(detection[1]) + ", ", end = "")
#     print("}, ")

# for a in range(len(trajectory_radii)):
#     print(trajectory_radii[a])
#     print(trajectory_centers[a])
#     print()

img.show()