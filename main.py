import random
import math
from PIL import Image, ImageDraw

def draw_point(image, xy, size, color):
    if size % 2 != 1:
        width = int(size / 2) + 1
    else:
        width = size / 2
    bbox = [(xy[0] - width, xy[1] - width), (xy[0] + width, xy[1] + width)]
    image.ellipse(bbox, fill= color)

def draw_concentric(image, origin, r, w):
    bbox = [(origin[0] - r, origin[1] - r), (origin[0] + r, origin[1] + r)]
    image.ellipse(bbox, outline = 'white', width = w)

W = 1500
H = 1500
N_CONCENTRIC = 5
SENSOR_DENSITY = 100

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

#detectors drawing and generation
for r in range(concentric_step, int(W/2), concentric_step):
    draw_concentric(canvas, origin, r, 3)

    for angle in [a / 1000 for a in range(0, 360 * 1000, int(360 * 1000 / SENSOR_DENSITY))]:
        x = 750 + int(r * math.cos(math.pi * 2 * angle / 360));
        y = 750 + int(r * math.sin(math.pi * 2 * angle / 360));
        
        #draw_point(canvas, (x,y), 10, "green")

for i in range(20):
    
    angle = random.randint(0, 359)
    r = radius = random.randint(rmin, rmax)
    dir = random.randint(0,1)
    if dir == 1:
        astart = angle
        aend = (angle + 180) % 360
    else:
        astart = (360 + angle - 180) % 360
        aend = angle

    x = 750 + int(radius * math.cos(math.pi * 2 * angle / 360));
    y = 750 + int(radius * math.sin(math.pi * 2 * angle / 360));
    bbox = [(x - r, y - r), (x + r, y + r)]

    radii.append(r)
    angles.append(angle)
    directions.append(dir)
    centers.append((x,y))

    canvas.arc(bbox, start = astart, end = aend , fill = (random.randint(50,200), random.randint(50,200), random.randint(50,200)), width=3)

    detection = []
    for d2 in range(concentric_step, int(W/2), concentric_step):
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

    


img.show()