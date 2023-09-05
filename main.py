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
trajs = "-8339.115489,-8163.026406,11675.231933,0,3.916321:-8327.541378,-8114.365487,11628.178452,0,3.914026:6666.930181,4964.803325,8315.670232,0,0.640095:6017.779153,-4277.773065,7376.436735,1,5.665211:10251.776574,-5910.302006,11839.377118,0,5.760213:5949.475008,-6171.975441,8567.850454,0,5.479433:4669.964091,4782.129773,6686.872201,0,0.797264:10532.631439,-4104.203572,11297.980773,1,5.911620:-11142.802608,6588.925715,12937.933360,1,2.607582:-8916.010933,-2897.746589,9368.954337,1,3.455829:4341.733761,-9707.165993,10639.002878,1,5.132971:-11125.762993,6591.466642,12928.552636,1,2.606743:-8999.167249,-2866.030289,9444.860073,1,3.449914:4127.676218,-8934.555005,9835.291422,1,5.145169:-1974.099981,-6772.982413,7056.841706,0,4.428779:2583.757363,9714.916404,10055.707651,1,1.310856:-5039.708787,7499.240170,9042.736198,1,2.162502:-2018.096506,8648.347240,8883.454818,0,1.800045:-171.415774,-12093.915399,12101.506900,1,4.698216:7534.714076,2276.492074,7870.564411,0,0.293413:9307.813202,5206.724301,10657.468391,1,0.510026:9174.803005,-4740.600709,10326.287625,1,5.806269:2851.720404,-8445.095260,8912.839038,0,5.038044:3256.823156,-10425.995329,10927.680881,1,5.015160:-5077.717089,7790.411636,9307.806734,1,2.148429:2915.521280,-8513.128660,8997.490999,0,5.042343:9524.286651,-1412.928239,9620.810294,1,6.135909:4479.600737,-8544.081915,9642.043237,1,5.195281:-10871.548447,5123.183475,12015.064695,1,2.701211:-8120.084305,-8664.921305,11870.594069,0,3.959439:-2423.076854,9712.504637,10007.706836,1,1.815286:-2447.537187,9101.903870,9434.123317,0,1.833486:-10113.203583,529.284437,10126.054534,1,3.089304:-1487.362534,7024.375479,7173.171352,1,1.779457:7283.385625,9917.625350,12306.764056,1,0.937362:-5944.672786,-10005.841759,11632.131461,1,4.176304:9850.017178,-1644.836116,9995.366352,1,6.117724:5872.751984,-7917.001812,9860.647212,1,5.350615:10552.882071,-2758.437527,10913.966873,1,6.027514:13237.054366,-1095.875779,13285.680034,0,6.200585:9769.044949,-2726.333924,10143.642555,0,6.011031:-1243.822398,13095.892798,13152.674427,1,1.665490:-1435.793245,7164.575796,7301.545392,1,1.768578:-1454.705866,7093.388967,7245.251219,1,1.773071:9390.979585,28.463084,9392.465885,1,0.003031:-9238.703709,-5549.297665,10774.278976,0,3.682496:-977.725857,-12239.887338,12282.162581,1,4.632678:-4756.340607,-8151.416985,9433.159200,1,4.184191:-8974.251450,1991.411774,9192.822988,0,2.923228:-8772.040754,5616.282938,10415.602849,1,2.572103:-12932.545299,-1040.142498,12974.559178,1,3.221848:-3250.732256,7196.853413,7896.185787,0,1.995053:8906.218011,-5188.224680,10311.618770,1,5.755703:-4854.054687,4963.729353,6952.520116,1,2.345024:4789.354311,-9955.632604,11047.411130,1,5.160778:-4855.115284,4924.990791,6914.509613,1,2.349050:-7584.609232,4454.812644,8796.287786,1,2.610527:2831.126854,-8435.811445,8899.060809,0,5.036185:-1959.808023,12823.643514,12975.407940,0,1.722451:-8573.749624,-5880.362248,10393.256139,0,3.742763:12449.833682,-3131.905352,12840.824121,1,6.036737:7122.502979,-5406.760745,8936.028048,1,5.633879:-2299.934179,11109.343314,11345.863918,1,1.774939:-7421.600255,1098.993181,7506.072299,1,2.994581:-7427.668556,3389.156444,8161.426606,1,2.713522:-10799.493004,5065.465605,11929.933451,1,2.703013:6371.123737,-9429.378010,11375.045772,1,5.306597:2478.925817,-12042.247023,12301.672375,0,4.915405:9800.862390,-1627.987992,9943.184377,1,6.118582:-9213.263971,-8137.904407,12295.329132,0,3.865094:-6506.309681,9049.989068,11138.142525,0,2.194114:6445.367417,-4053.701010,7614.856784,1,5.721763:-7420.622574,1101.670635,7504.165960,1,2.994208:-12304.437653,-1399.990408,12386.708182,0,3.254885:-6724.646552,-3217.543788,7456.756128,1,3.587869:-4743.512091,-10455.918739,11482.737858,1,4.286489:-5044.596126,-6284.238995,8057.257459,0,4.035981:11421.165327,-80.863042,11424.213149,0,6.276105:7107.240345,2002.314386,7380.542241,1,0.274611:-3234.829106,12532.132484,12940.640205,0,1.823405:-9245.235063,-2254.539375,9517.662394,0,3.380784:-10689.242341,-3760.860487,11337.078722,0,3.479902:6525.684830,-3876.837593,7592.430616,1,5.747123:5830.095474,-9624.233726,11243.758333,1,5.257042:-4067.772798,-5994.804385,7244.084253,0,4.116205:2828.034863,-8419.089640,8887.047271,0,5.036454:5056.249436,-10823.278584,11946.201070,0,5.149425:-3429.126610,-7380.599428,8132.603058,0,4.277449:10358.856332,-4872.482755,11450.146326,1,5.843522:6333.714382,8677.298625,10744.774584,1,0.940272:6322.217798,-7285.073380,9645.038757,1,5.427145:9442.511984,6110.374743,11245.175332,1,0.574343:1973.293570,-6576.914888,6867.220229,1,5.003876:7678.957068,1813.094766,7889.164659,0,0.231866:870.462867,11655.325347,11689.129312,1,1.496251:6873.705820,8326.565844,10789.103181,1,0.880690:1063.099387,12104.540600,12149.183206,1,1.483195:-6520.652280,6559.356401,9250.311828,1,2.353235:-9835.362088,-4997.009466,11030.050741,1,3.611672:-8768.857956,-5773.197374,10500.195664,0,3.723833:-7238.836968,-9667.752742,12076.265014,1,4.069681:10919.900236,4431.720925,11780.707340,1,0.385530:-4335.321131,8732.550742,9749.126692,0,2.031604:-11921.995013,170.884735,11926.648101,0,3.127260:-4063.752507,-6232.086573,7442.484152,0,4.134560:-8375.673517,1690.243500,8537.647339,1,2.942463:-6865.391046,-4597.311506,8265.482899,1,3.731648:-7938.162050,2353.998435,8285.451965,0,2.853311:11337.271501,6895.575688,13272.293354,0,0.546443:-12544.079284,-3980.912860,13155.477642,1,3.448893:-6100.209008,-11442.915930,12968.676684,1,4.222614:-5021.575643,-6108.433253,7907.198107,0,4.024333:8000.855933,-5916.593807,9953.976030,1,5.646441:-2423.625665,6589.770195,7022.253164,0,1.923228:-4769.951653,-7640.768991,9006.013611,1,4.154310:-8381.640522,3187.099094,8957.956255,1,2.778229:-8482.415896,3314.006247,9109.767452,1,2.769137:9289.164864,7725.861418,12083.650129,1,0.693778:-4521.588163,-8694.019257,9797.704803,0,4.232806:"

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