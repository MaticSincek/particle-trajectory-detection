import math

def inRange(p1, p2):
    TOLERANCE = 0.01 * math.dist([0,0], p1)

    if math.dist(p1, p2) < TOLERANCE:
        return True
    return False

def main():
    
    inp = ""

    t = inp.split(":")

    trajs = []
    for i in range(len(t)-1):
        trajs.append(list(map(lambda x: float(x), t[i].split(","))))

    final_trajs = []

    for tr in trajs:
        included = False
        for ftr in final_trajs:
            if inRange([tr[0], tr[1]], [ftr[0], ftr[1]]):
                included = True
        if not included:
            final_trajs.append(tr)

    for x in final_trajs:
        print("r: " + str(x[2]) + ", (" + str(x[0]) + ", " + str(x[1]) + ")")

    print(len(final_trajs))


if __name__ == "__main__":
    main()