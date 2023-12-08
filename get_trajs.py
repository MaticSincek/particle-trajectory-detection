import math

def inRange(p1, p2):
    TOLERANCE = 0.01 * math.dist([0,0], p1)

    if math.dist(p1, p2) < TOLERANCE:
        return True
    return False

def main():
    
    inp = "-8142.927460,-7379.452078,10885.860942,0,3.877845:6035.062597,-4175.847772,7272.016612,1,5.677895:11234.566774,-5739.826974,12830.347305,0,5.810850:4649.307404,4941.194749,6829.984714,0,0.815824:6052.739578,-6195.921516,8657.292093,0,5.486098:-10828.863600,6093.848033,12356.483078,1,2.629020:4336.159399,-9477.436555,10422.291083,1,5.141483:4058.778388,-9105.520365,10020.071004,1,5.131702:-2011.457729,-6720.438493,7000.184837,0,4.421570:-9183.878615,-2758.562042,9663.050037,1,3.433389:7566.391329,2279.176201,7878.257259,0,0.292579:-111.235075,-11901.349153,11917.051936,1,4.703043:2763.572217,9819.055210,10260.289620,1,1.296444:-2013.569705,8618.992877,8865.825853,0,1.800300:9275.419569,-5007.238866,10608.605009,1,5.788176:3337.027817,-10156.459310,10650.964801,1,5.029839:4338.332610,-8795.162336,9928.682491,1,5.170633:2915.762521,-8470.289082,8984.221347,0,5.043918:-10882.822963,5185.450069,12079.459378,1,2.696937:-8140.677173,-9592.820067,12802.602329,0,4.008696:9083.075198,-447.461830,8565.925740,1,6.233962:7092.585862,9840.453812,12102.070203,1,0.946274:10090.271288,-1955.154862,10383.639114,1,6.091791:-1260.373718,8506.785932,8111.846325,0,1.717887:-10161.886326,540.269131,10164.161011,1,3.088476:10660.140842,-2879.812976,11071.234089,1,6.019336:-5568.685246,-9789.905774,11211.072793,1,4.195212:5901.098519,-8137.096040,10066.682347,1,5.339834:-2360.565839,9695.380657,10027.882972,1,1.809623:9555.794978,-2786.844390,9931.719247,0,5.999417:-1386.927613,7207.440172,7360.933080,1,1.760903:-1441.212777,7183.609986,7305.288340,1,1.768793:-1584.136856,6881.565599,7019.275792,1,1.797055:-4597.991433,-8065.604889,9269.694039,1,4.194265:-9471.712844,-6079.258973,11339.419007,0,3.712205:-3257.576750,7237.924251,7907.212818,0,1.993709:-4843.660024,4914.922249,6914.077672,1,2.348892:-8720.968465,5473.382181,10263.935588,1,2.581117:-13072.589556,-1061.592061,13079.881775,1,3.222622:-8397.272126,-5554.314236,10050.171690,0,3.725970:8886.395402,-5212.541678,10333.308335,1,5.752695:4813.018984,-9612.684630,10716.683536,1,5.176592:-7422.388182,3577.001938,8350.294270,1,2.692513:-7680.695416,4952.895259,9299.633218,1,2.568846:12406.688641,-3092.982589,12783.632735,1,6.038866:-7482.895680,1219.248563,7632.703805,1,2.980074:-7280.278716,477.145394,6870.986033,1,3.076147:-905.566067,11953.569551,11616.627291,0,1.646409:6424.680308,-8807.454115,10755.216177,1,5.342614:2711.243057,-8356.632968,8764.007504,0,5.026116:-2207.412776,11300.145101,11556.403514,1,1.763711:-12255.103141,-1308.103542,12287.059853,0,3.247930:-7635.570161,9457.135344,12320.753847,0,2.250025:9487.433963,-797.400356,9075.974713,1,6.199334:-10746.747952,-3747.976333,11353.944626,0,3.477157:-11482.163709,6122.543179,13173.130324,1,2.651722:-4214.241198,-10181.944471,10891.568006,1,4.319963:2823.190775,-8385.399885,8874.184611,0,5.037148:-9083.440288,-7430.980396,11575.968875,0,3.827261:6002.381544,-8859.485277,10462.588974,1,5.307860:6431.465614,-4259.122298,7823.923532,1,5.698259:-9332.900207,-2391.345607,9674.146113,0,3.392424:7109.474136,-5935.361912,9463.007035,1,5.587551:-6707.973169,-3200.414303,7455.811348,1,3.586758:11660.831710,139.839806,11750.372485,0,0.011992:5486.173979,-11136.884993,12466.726324,0,5.170110:-2258.710127,11632.381049,11619.170672,0,1.762584:6790.055120,2306.830402,6951.296993,1,0.327502:7060.056070,8784.258953,11478.145745,1,0.893793:6496.091587,-3782.081988,7504.892198,1,5.755950:-4253.797668,-5718.125114,6917.636180,0,4.072794:-666.979265,9974.180826,9420.901545,1,1.637567:1184.414749,12419.994097,12456.403628,1,1.475720:1821.333023,-6685.647348,7049.525243,1,4.978359:-3986.031078,-6315.729198,7556.700289,0,4.149395:6755.026609,8203.210204,10662.969503,1,0.881912:-6882.655018,-4609.051628,8276.173682,1,3.731666:-6812.540112,-9456.278893,11616.298577,1,4.088086:-8837.908842,-5936.559717,10670.755467,0,3.733082:7705.002296,1799.722228,7882.870415,0,0.229464:-4425.373923,8818.619164,9861.021321,0,2.035900:-5078.724935,-7647.222312,9311.852768,1,4.126147:-10168.584249,-4982.033076,11352.734085,1,3.597163:8565.850182,7600.498650,11360.522693,1,0.725755:-8242.862803,2188.073634,8616.704629,0,2.882126:10710.790653,4506.734003,11563.887465,1,0.398279:-4490.870798,-8975.984778,10072.675833,0,4.248485:-8286.538719,1741.896587,8560.135721,1,2.934401:7919.660176,-4936.237490,8975.297196,1,5.725817:-12219.464039,-4106.598717,12809.775975,1,3.465804:-5042.453761,-6124.498099,7915.165616,0,4.023586:-2471.702131,6620.105553,7069.244405,0,1.928131:-8293.697071,2929.772821,8690.919111,1,2.802023:-8580.381481,3491.672412,9304.918901,1,2.755121:"

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