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
N_TRAJECTORIES = 100
SENSOR_DENSITY = 3600
N_SEED_CORRECTIONS = 30 #* 30 * 5
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
    
for layer in detections_on_layer:
    print("{ ", end = "")
    for detection in layer:
        print(str(detection[0]) + ", ", end = "")
    print("}, ")
    
print("")
print("")
print("")

for layer in detections_on_layer:
    print("{ ", end = "")
    for detection in layer:
        print(str(detection[1]) + ", ", end = "")
    print("}, ")

print("")
print("")
print("")

print("{ ", end="")
for layer in detections_on_layer:
    print(len(layer), end=", ")
print("};")

# for a in range(len(trajectory_radii)):
#     print(trajectory_radii[a])
#     print(trajectory_centers[a])
#     print()

# PRINT PASTED POINTS
# pnts = []
# for p in pnts:
#     x = scale_for_drawing(p[0]) + origin[0]
#     y = scale_for_drawing(p[1]) + origin[1]

#     draw_point(canvas, (x,y), 4, "red")

# PRINT PASTED TRAJECTORIES
trajs = "-8142.927460,-7379.452078,10885.860942,0,3.877845:6035.062597,-4175.847772,7272.016612,1,5.677895:11234.566774,-5739.826974,12830.347305,0,5.810850:4649.307404,4941.194749,6829.984714,0,0.815824:6052.739578,-6195.921516,8657.292093,0,5.486098:-10828.863600,6093.848033,12356.483078,1,2.629020:4336.159399,-9477.436555,10422.291083,1,5.141483:4058.778388,-9105.520365,10020.071004,1,5.131702:-2011.457729,-6720.438493,7000.184837,0,4.421570:-9183.878615,-2758.562042,9663.050037,1,3.433389:7566.391329,2279.176201,7878.257259,0,0.292579:-111.235075,-11901.349153,11917.051936,1,4.703043:2763.572217,9819.055210,10260.289620,1,1.296444:-2013.569705,8618.992877,8865.825853,0,1.800300:9275.419569,-5007.238866,10608.605009,1,5.788176:3337.027817,-10156.459310,10650.964801,1,5.029839:4338.332610,-8795.162336,9928.682491,1,5.170633:2915.762521,-8470.289082,8984.221347,0,5.043918:-10882.822963,5185.450069,12079.459378,1,2.696937:-8140.677173,-9592.820067,12802.602329,0,4.008696:9083.075198,-447.461830,8565.925740,1,6.233962:7092.585862,9840.453812,12102.070203,1,0.946274:10090.271288,-1955.154862,10383.639114,1,6.091791:-1260.373718,8506.785932,8111.846325,0,1.717887:-10161.886326,540.269131,10164.161011,1,3.088476:10660.140842,-2879.812976,11071.234089,1,6.019336:-5568.685246,-9789.905774,11211.072793,1,4.195212:5901.098519,-8137.096040,10066.682347,1,5.339834:-2360.565839,9695.380657,10027.882972,1,1.809623:9555.794978,-2786.844390,9931.719247,0,5.999417:-1386.927613,7207.440172,7360.933080,1,1.760903:-1441.212777,7183.609986,7305.288340,1,1.768793:-1584.136856,6881.565599,7019.275792,1,1.797055:-4597.991433,-8065.604889,9269.694039,1,4.194265:-9471.712844,-6079.258973,11339.419007,0,3.712205:-3257.576750,7237.924251,7907.212818,0,1.993709:-4843.660024,4914.922249,6914.077672,1,2.348892:-8720.968465,5473.382181,10263.935588,1,2.581117:-13072.589556,-1061.592061,13079.881775,1,3.222622:-8397.272126,-5554.314236,10050.171690,0,3.725970:8886.395402,-5212.541678,10333.308335,1,5.752695:4813.018984,-9612.684630,10716.683536,1,5.176592:-7422.388182,3577.001938,8350.294270,1,2.692513:-7680.695416,4952.895259,9299.633218,1,2.568846:12406.688641,-3092.982589,12783.632735,1,6.038866:-7482.895680,1219.248563,7632.703805,1,2.980074:-7280.278716,477.145394,6870.986033,1,3.076147:-905.566067,11953.569551,11616.627291,0,1.646409:6424.680308,-8807.454115,10755.216177,1,5.342614:2711.243057,-8356.632968,8764.007504,0,5.026116:-2207.412776,11300.145101,11556.403514,1,1.763711:-12255.103141,-1308.103542,12287.059853,0,3.247930:-7635.570161,9457.135344,12320.753847,0,2.250025:9487.433963,-797.400356,9075.974713,1,6.199334:-10746.747952,-3747.976333,11353.944626,0,3.477157:-11482.163709,6122.543179,13173.130324,1,2.651722:-4214.241198,-10181.944471,10891.568006,1,4.319963:2823.190775,-8385.399885,8874.184611,0,5.037148:-9083.440288,-7430.980396,11575.968875,0,3.827261:6002.381544,-8859.485277,10462.588974,1,5.307860:6431.465614,-4259.122298,7823.923532,1,5.698259:-9332.900207,-2391.345607,9674.146113,0,3.392424:7109.474136,-5935.361912,9463.007035,1,5.587551:-6707.973169,-3200.414303,7455.811348,1,3.586758:11660.831710,139.839806,11750.372485,0,0.011992:5486.173979,-11136.884993,12466.726324,0,5.170110:-2258.710127,11632.381049,11619.170672,0,1.762584:6790.055120,2306.830402,6951.296993,1,0.327502:7060.056070,8784.258953,11478.145745,1,0.893793:6496.091587,-3782.081988,7504.892198,1,5.755950:-4253.797668,-5718.125114,6917.636180,0,4.072794:-666.979265,9974.180826,9420.901545,1,1.637567:1184.414749,12419.994097,12456.403628,1,1.475720:1821.333023,-6685.647348,7049.525243,1,4.978359:-3986.031078,-6315.729198,7556.700289,0,4.149395:6755.026609,8203.210204,10662.969503,1,0.881912:-6882.655018,-4609.051628,8276.173682,1,3.731666:-6812.540112,-9456.278893,11616.298577,1,4.088086:-8837.908842,-5936.559717,10670.755467,0,3.733082:7705.002296,1799.722228,7882.870415,0,0.229464:-4425.373923,8818.619164,9861.021321,0,2.035900:-5078.724935,-7647.222312,9311.852768,1,4.126147:-10168.584249,-4982.033076,11352.734085,1,3.597163:8565.850182,7600.498650,11360.522693,1,0.725755:-8242.862803,2188.073634,8616.704629,0,2.882126:10710.790653,4506.734003,11563.887465,1,0.398279:-4490.870798,-8975.984778,10072.675833,0,4.248485:-8286.538719,1741.896587,8560.135721,1,2.934401:7919.660176,-4936.237490,8975.297196,1,5.725817:-12219.464039,-4106.598717,12809.775975,1,3.465804:-5042.453761,-6124.498099,7915.165616,0,4.023586:-2471.702131,6620.105553,7069.244405,0,1.928131:-8293.697071,2929.772821,8690.919111,1,2.802023:-8580.381481,3491.672412,9304.918901,1,2.755121:"

s = trajs.split(":")[0:-1]
for traj in s:
    split = traj.split(",")
    cenx = float(split[0])
    ceny = float(split[1])
    center = (cenx,ceny)
    r = float(split[2])
    dir = int(split[3])
    angle = float(split[4])

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
        fill = (0, 255, 0),
        width = 2)
    
img.show()