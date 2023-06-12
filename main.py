# NOTES
# Currently we calculate the trajectory only from the three seed points. 
# We could improove the trajectory fitting even better if we used all the points in the calculation of the trajectory

import random
from decimal import *
import math
from PIL import Image, ImageDraw
import numpy as np

random.seed(2)

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

    # colinear, return none
    if abs(det) < 1.0e-6:
        return (None, np.inf)

    # center of circle
    cx = (bc*(p2[1] - p3[1]) - cd*(p1[1] - p2[1])) / det
    cy = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det

    radius = np.sqrt((cx - p1[0])**2 + (cy - p1[1])**2)
    return ((cx, cy), radius)

def angle_of_point_relative_to_origin(x_origin, y_origin, x, y):
    angle_rad = math.atan2(y - y_origin, x - x_origin)
    angle_deg = math.degrees(angle_rad)
    angle_deg = (angle_deg + 360) % 360
    return angle_deg

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
    
def cartesian2polar(orig, p):
    x,y = p
    angle = angle_of_point_relative_to_origin(orig[0], orig[1], x, y)
    distance = math.sqrt((orig[0] - x) ** 2 + (orig[1] - y) ** 2)
    return (distance, angle)

def polar2cartesian(orig, distance, angle):
    x = distance * math.cos(math.radians(angle)) + orig[0]
    y = distance * math.sin(math.radians(angle)) + orig[1]
    return (x,y)

def centralize_point_on_sensor(orig, p, sensor_segment_angle):
    distance, angle = cartesian2polar(orig, p)
    new_angle = (int(angle / sensor_segment_angle) * sensor_segment_angle) + sensor_segment_angle / 2
    return polar2cartesian(orig, distance, new_angle)

def point_sensor_subspace(orig, p, sensor_segment_angle, n_segments):
    distance, angle = cartesian2polar(orig, p)
    a_min = int(angle / sensor_segment_angle) * sensor_segment_angle
    a_max = (int(angle / sensor_segment_angle) + 1) * sensor_segment_angle
    a_range = [x / 1000 for x in range(int(a_min * 1000), int(a_max * 1000), int((a_max * 1000 - a_min * 1000) / n_segments))]
    p_range =  [polar2cartesian(orig, distance, a) for a in a_range]
    return p_range

def random_point_sensor_subspace(orig, p, sensor_segment_angle, n_segments):
    distance, angle = cartesian2polar(orig, p)
    a_min = int(angle / sensor_segment_angle) * sensor_segment_angle
    a_max = (int(angle / sensor_segment_angle) + 1) * sensor_segment_angle
    a_delta = a_max - a_min
    array = [-1] * n_segments
    for i in range(len(array)):
        a = a_min + random.random() * a_delta
        array[i] = polar2cartesian(orig, distance, a)
    return array

W = 1500
H = 1500
N_CONCENTRIC = 20
N_TRAJECTORIES = 40
SENSOR_DENSITY = 360
SUBSENSOR_SPACE = 10
TOLERANCE = 10
CENTER_TOLERANCE = 3
TRAJECTORY_ANGLE_TOLERANCE = 50
SEED_ANGLE_TOLERANCE = 10
MIN_PERC_COVERAGE_FOR_TRAJ = 0.9
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
rmin = int(W * 2 / 3 / 2)
rmax = W * 2 / 3 * 2

img = Image.new("RGB", (W, H))
canvas = ImageDraw.Draw(img)

draw_point(canvas, origin, 6, "white")

#calculate the number of concentric circles by dividing half of the screen by N_CONCENTRIC
concentric_space = int(W / 2 - 15)
concentric_step = int(concentric_space / N_CONCENTRIC)
layer_radii = [r for r in range(concentric_step, concentric_space + 1, concentric_step)]

# detector layer drawing and sensor generation
for r in layer_radii:
    draw_concentric(canvas, origin, r, 3)

# generation of arcs
for i in range(N_TRAJECTORIES):
    angle = random.randint(0, 359)
    r = radius = random.randint(rmin, rmax)
    dir = random.randint(0,1)

    # we should always only draw half of the circle
    if dir == 1:
        astart = angle
        aend = (angle + 180) % 360
    else:
        astart = (360 + angle - 180) % 360
        aend = angle

    # points r away from the origin with an angle "angle" relative to the origin 
    x = origin[0] + int(r * math.cos(math.pi * 2 * angle / 360))
    y = origin[1] + int(r * math.sin(math.pi * 2 * angle / 360))

    # center of bbox is in (x,y) with corners r away from the center in both directions
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
        width = 5)

    # we already have the trajectory now let's see where it intersects each concentric circle
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

        if not WITH_SENSORS:
            draw_point(canvas, (x3,y3), 7, "red")

    detections.append(detection)

# sorting detections by layer
detections_on_layer = [[] for _ in layer_radii]

for i in range(N_TRAJECTORIES):
    for lr in range(len(layer_radii)):
        detections_on_layer[lr].append(detections[i][lr])

segment_angle = 360 / SENSOR_DENSITY

if WITH_SENSORS:
    for i in range(len(detections_on_layer)):
        for j in range(len(detections_on_layer[i])):
            # we have to round as sometimes sometimes centralize_point_on_sensor 
            # gives different 8th decimal for two points on the same coordinates
            p = centralize_point_on_sensor(origin, detections_on_layer[i][j], segment_angle)
            new_x = Decimal(str(p[0])).quantize(Decimal('1e-2'))
            new_y = Decimal(str(p[1])).quantize(Decimal('1e-2'))
            detections_on_layer[i][j] = (float(new_x), float(new_y))

# remove points that translated to the same sensor to avoid duplicate trajectories
for i in range(len(detections_on_layer)):
    detections_on_layer[i] = list(dict.fromkeys(detections_on_layer[i]))

if WITH_SENSORS:
    for i in range(len(detections_on_layer)):
        for j in range(len(detections_on_layer[i])):
            draw_point(canvas, centralize_point_on_sensor(origin, detections_on_layer[i][j], segment_angle), 6, "yellow")
              
# radius of seed
seed_radii = []
# negative or positive magnetic effect
seed_directions = []
# center of circle the curve is based on
seed_centers = []
# approximate angles of the point on the trajectory
seed_trajectory_angles = []
# points used in seed
seed_points = []

# number of points we need to find on the seed trajectory for us to count it as an actual trajectory
points_needed = int(N_CONCENTRIC * MIN_PERC_COVERAGE_FOR_TRAJ)

# finding the trajectories from points by combinatorics
for p0 in detections_on_layer[N_CONCENTRIC-1]:
    for p1 in detections_on_layer[N_CONCENTRIC-2]:

        r_best = None
        pp0_best = None
        pp1_best = None
        center_best = None
        seed_traj_angle_best = None

        min_avg_error = 999999

        angle_p0 = angle_reference = angle_of_point_relative_to_origin(origin[0], origin[1], p0[0], p0[1])
        angle_p1 = angle_of_point_relative_to_origin(origin[0], origin[1], p1[0], p1[1])

        # if both points are roughly in the same direction from the origin we continue
        if (abs(angle_p0 - angle_p1) < SEED_ANGLE_TOLERANCE) or \
            abs(angle_p0 - angle_p1) > (360 - SEED_ANGLE_TOLERANCE):

            #generate arrays of possible actual points of intersectioin with the sensor
            p0_space = random_point_sensor_subspace(origin, p0, segment_angle, SUBSENSOR_SPACE)
            p1_space = random_point_sensor_subspace(origin, p1, segment_angle, SUBSENSOR_SPACE)

            for pp in p0_space:
                draw_point(canvas,pp,3,"blue")
            for pp in p1_space:
                draw_point(canvas,pp,3,"green")

            # try every possible combination
            for pp0 in p0_space:
                for pp1 in p1_space:

                    # find the r and center of these 3 points
                    (center, r) = circle_from_points(pp0, pp1, origin)

                    if center == None:
                        continue

                    pp0_angle = angle_of_point_relative_to_origin(origin[0], origin[1], pp0[0], pp0[1])

                    points_on_seed_trajectory = 2
                    cumul_error = 0
                    
                    # if r makes sense it could be a trajectory
                    if r < rmax and r > rmin:

                        # check how the seed is supported by points
                        for l in range(N_CONCENTRIC-3, -1, -1):
                            min_error = 999999
                            for det in range(len(detections_on_layer[l])):
                                p = detections_on_layer[l][det]
                                # calculate distance from the seed center to point to see if it is on the trajectory
                                d = math.sqrt((center[0] - p[0]) ** 2 + (center[1] - p[1]) ** 2)
                                # is distance to origin is approx. the same as r 
                                # and the angle from origin is similar it is on the trajectory of seed s
                                error = abs(d-r)
                                if error < TOLERANCE:
                                    p_angle = angle_of_point_relative_to_origin(origin[0], origin[1], p[0], p[1])

                                    if (abs(angle_reference - p_angle) < TRAJECTORY_ANGLE_TOLERANCE) or \
                                        abs(angle_reference - p_angle) > (360 - TRAJECTORY_ANGLE_TOLERANCE):
                                        if error < min_error:
                                            min_error = error

                            if (min_error < 999999):
                                cumul_error += min_error
                                points_on_seed_trajectory += 1

                    # if we have a trajectory we save it as best for the two points; if it has the least error that is
                    if points_on_seed_trajectory >= points_needed:
                        avg_err = cumul_error / points_on_seed_trajectory
                        if avg_err < min_avg_error:
                            min_avg_error = avg_err
                            r_best = r
                            pp0_best = pp0
                            pp1_best = pp1
                            center_best = center
                            seed_traj_angle_best = pp0_angle

        if center_best != None:
            o = get_orientation(origin,pp1_best,pp0_best) 
            seed_radii.append(r_best)
            seed_centers.append(center_best)
            seed_directions.append(o)
            seed_trajectory_angles.append(seed_traj_angle_best)
            seed_points.append((origin,pp0_best,pp1_best))

print("Found " + str(len(seed_radii)) + " out of " + str(N_TRAJECTORIES) + " trajectories." )

# draw the trajectories
for i in range(len(seed_radii)):
    r = seed_radii[i]
    dir = seed_directions[i]
    center = seed_centers[i]
    angle = angle_of_point_relative_to_origin(origin[0], origin[1], center[0], center[1])

    # we should always only draw half of the circle
    if dir == 1:
        astart = angle
        aend = (angle + 180) % 360
    else:
        astart = (360 + angle - 180) % 360
        aend = angle

    # calculate bbox from center and r
    bbox = [(center[0] - r, center[1] - r), (center[0] + r, center[1] + r)]

    canvas.arc(
        bbox, 
        start = astart, 
        end = aend, 
        fill = (255, 255, 255),
        width = 2)
    
"""    
for a in range(len(seed_centers)):
    if seed_points[a][2] == (1047.6980719706721, 1365.8180396391003):
        print(seed_points[a])
        print(seed_trajectory_angles[a])
        print("++++++++++++++++++++++++++++++")

    draw_point(canvas, (1084.6862729217262, 1387.4834105431794), 6, "green")
    draw_point(canvas, (1082.6534002096673, 1347.6602005562581), 6, "green")
    draw_point(canvas, (1077.9923332081312, 1390.9532193200112), 6, "green")
    draw_point(canvas, (1047.6980719706721, 1365.8180396391003), 6, "green")

print("+++++++++++++++++++++++++++++++")
for a in range(len(seed_centers)):
    if seed_trajectory_angles[a] < 90 and seed_trajectory_angles[a] > 45:
        print(seed_trajectory_angles[a])

print("+++++++++++++++++++++++++++++++")
for p in detections_on_layer[19]:
    a = angle_of_point_relative_to_origin(origin[0], origin[1], p[0], p[1])
    if(a > 45 and a <90):
        print(a)
print("+++++++++++++++++++++++++++++++")
for p in detections_on_layer[18]:
    a = angle_of_point_relative_to_origin(origin[0], origin[1], p[0], p[1])
    if(a > 180 + 30 and a < 180 + 45):
        print(a)
print("+++++++++++++++++++++++++++++++")
"""

img.show()