# NOTES
# Currently we calculate the trajectory only from the three seed points. 
# We could improove the trajectory fitting even better if we used all the points in the calculation of the trajectory

import random
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

W = 1500
H = 1500
N_CONCENTRIC = 20
N_TRAJECTORIES = 19
SENSOR_DENSITY = 10
TOLERANCE = H / 100
TRAJECTORY_ANGLE_TOLERANCE = 50
MIN_PERC_COVERAGE_FOR_TRAJ = 1

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
concentric_step = int(((W / 2) - 50) / N_CONCENTRIC)
layer_radii = [r for r in range(concentric_step, int(W/2), concentric_step)]

# detector layer drawing and sensor generation
for r in layer_radii:
    draw_concentric(canvas, origin, r, 3)

    number_of_sensors = SENSOR_DENSITY

    for angle in [a / 1000 for a in range(0, 360 * 1000, int(360 * 1000 / number_of_sensors))]:
        x = origin[0] + int(r * math.cos(math.pi * 2 * angle / 360))
        y = origin[1] + int(r * math.sin(math.pi * 2 * angle / 360))
        
        #draw_point(canvas, (x,y), 10, "green")

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
        draw_point(canvas, (x3,y3), 7, "red")

    detections.append(detection)



# sorting detections by layer
detections_on_layer = [[] for _ in layer_radii]

for i in range(N_TRAJECTORIES):
    for lr in range(len(layer_radii)):
        detections_on_layer[lr].append(detections[i][lr])

combinations_checked = 0
seeds_found = 0

# radius of seed
seed_radii = []
# negative or positive magnetic effect
seed_directions = []
# center of circle the curve is based on
seed_centers = []
# approximate angles of the point on the trajectory
seed_trajectory_angles = []

# finding the trajectories from points by combinatorics
for p0 in detections_on_layer[N_CONCENTRIC-1]:
    for p1 in detections_on_layer[N_CONCENTRIC-2]:
        for p2 in detections_on_layer[N_CONCENTRIC-3]:

            combinations_checked += 1

            # find the r and center of these 3 points
            (center, r) = circle_from_points(p0, p1, p2)
            
            # calculate distance from the calculated center to origin
            d = math.sqrt((center[0] - origin[0]) ** 2 + (center[1] - origin[1]) ** 2)
            
            # if distance to origin is approx. the same as r it could be a trajectory
            if abs(d-r) < TOLERANCE:
                approximate_trajectory_angle = angle_of_point_relative_to_origin(origin[0], origin[1], p0[0], p0[1])
                angle_p1 = angle_of_point_relative_to_origin(origin[0], origin[1], p1[0], p1[1])
                angle_p2 = angle_of_point_relative_to_origin(origin[0], origin[1], p2[0], p2[1])
                # if all three point are roughly in the same direction from the origin we have a trajectory
                if ((approximate_trajectory_angle - TRAJECTORY_ANGLE_TOLERANCE) < angle_p1 < (approximate_trajectory_angle + TRAJECTORY_ANGLE_TOLERANCE)) and \
                    ((approximate_trajectory_angle - TRAJECTORY_ANGLE_TOLERANCE) < angle_p2 < (approximate_trajectory_angle + TRAJECTORY_ANGLE_TOLERANCE)):
                    seeds_found += 1

                    o = get_orientation(p2,p1,p0)

                    seed_radii.append(r)
                    seed_centers.append(center)
                    seed_directions.append(o)
                    seed_trajectory_angles.append(approximate_trajectory_angle)

            # if it could be a trajectory check how many of other points are on the path of the trajectory
            # for each point first check trajectory compliance then check angle relative to center (must have the similar to other three points)

            # for debugging purposes of seed finding uncomment this block
            """
            x = 0 + center[0]
            y = 0 + center[1]
            bbox = [(x - r, y - r), (x + r, y + r)]

            canvas.arc(
                bbox, 
                start = 0, 
                end = 360, 
                fill = "red",
                width = 1)

        break
    break
    
            
print("Number of checked combinations: " + str(combinations_checked))
print("Number of found track seeds: " + str(seeds_found))
"""

# radius of trajectory
trajectory_radii = []
# negative or positive magnetic effect
trajectory_directions = []
# center of circle the curve is based on
trajectory_centers = []

# number of points we need to find on the seed trajectory for us to count it as an actual trajectory
points_needed = int(N_CONCENTRIC * MIN_PERC_COVERAGE_FOR_TRAJ)

# for each seed we should check how many hits on the other layers we get
# s as the seed number
for s in range(len(seed_radii)):
    center = seed_centers[s]
    r = seed_radii[s]
    angle = seed_trajectory_angles[s]

    points_on_seed_trajectory = 3

    for l in range(N_CONCENTRIC-4, -1, -1):
        for p in detections_on_layer[l]:
            # calculate distance from the seed center to point to see if it is on the trajectory
            d = math.sqrt((center[0] - p[0]) ** 2 + (center[1] - p[1]) ** 2)
            # is distance to origin is approx. the same as r 
            # and the angle from origin is similar it is on the trajectory of seed s
            if abs(d-r) < TOLERANCE:
                p_angle = angle_of_point_relative_to_origin(origin[0], origin[1], p[0], p[1])
                #print(angle- p_angle, end = " ")
                #print(l)
                #if l == 0:
                #    print(p_angle)
                #    print(angle)
                #    print(p)
                #    draw_point(canvas, p, 7, "green")

                if ((angle - TRAJECTORY_ANGLE_TOLERANCE) < p_angle < (angle + TRAJECTORY_ANGLE_TOLERANCE)):
                    print(angle- p_angle, end = " ")
                    print(l)
                    points_on_seed_trajectory += 1
                    break
        if points_on_seed_trajectory >= points_needed:
            break

    # in this case we are certai of the trajectory
    if points_on_seed_trajectory >= points_needed:
        trajectory_radii.append(r)
        trajectory_directions.append(seed_directions[s])
        trajectory_centers.append(center)
    print(points_on_seed_trajectory / points_needed)
    print("----------")

print("Found " + str(len(trajectory_radii)) + " out of " + str(N_TRAJECTORIES) + " trajectories." )

# draw the trajectories
for i in range(len(trajectory_radii)):
    r = trajectory_radii[i]
    dir = trajectory_directions[i]
    center = trajectory_centers[i]
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

img.show()