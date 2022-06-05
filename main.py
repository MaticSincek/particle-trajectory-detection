import random
import math
import numpy
from PIL import Image, ImageDraw

def draw_point(image, xy, size, color):
    if size % 2 != 1:
        width = int(size / 2) + 1
    else:
        width = size / 2
    bbox = [(xy[0] - width, xy[1] - width), (xy[0] + width, xy[1] + width)]
    image.ellipse(bbox, fill= color)

def draw_concentric(image, center, r, w):
    bbox = [(center[0] - r, center[1] - r), (center[0] + r, center[1] + r)]
    image.ellipse(bbox, outline = 'white', width = w)

W = 1500
H = 1500

center = (W/2, H/2)

rmin = int(W * 2 / 3 / 2)
rmax = W * 2 / 3 * 2

img = Image.new("RGB", (W, H))
canvas = ImageDraw.Draw(img)

draw_point(canvas, center, 6, "white")

for i in range(100, int(W/2), 100):
    draw_concentric(canvas, center, i, 3)

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

    canvas.arc(bbox, start = astart, end = aend , fill = (random.randint(50,200), random.randint(50,200), random.randint(50,200)), width=4)

    for d2 in range(100, int(W/2), 100):
        d=math.sqrt((x-center[0])**2 + (y-center[1])**2)
        a=(d2**2-r**2+d**2)/(2*d)
        h=math.sqrt(d2**2-a**2)
        x2=center[0]+a*(x-center[0])/d   
        y2=center[1]+a*(y-center[1])/d   
        x3=x2+h*(y-center[1])/d
        y3=y2-h*(x-center[0])/d
        x4=x2-h*(y-center[1])/d
        y4=y2+h*(x-center[0])/d

        if dir == 0:
            draw_point(canvas, (x3,y3), 7, "red")
        else:
            draw_point(canvas, (x4,y4), 7, "red")

img.show()