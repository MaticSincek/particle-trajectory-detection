#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h> 
#include <limits.h>
#include <float.h>
#include <math.h>

#define PI 3.141592654
#define N_CONCENTRIC 20
#define N_TRAJECTORIES 3

int W = 1500;
int H = 1500;
int SENSOR_DENSITY = 3600;
int N_SEED_CORRECTIONS = 30*30;
int TOLERANCE = 10;
int CENTER_TOLERANCE = 2;
int TRAJECTORY_ANGLE_TOLERANCE = 0.87266;
int SEED_ANGLE_TOLERANCE = 0.17453;
int MIN_PERC_COVERAGE_FOR_TRAJ = 0.9;
int DETECTION_FAIL_RATE = 0;
bool WITH_SENSORS = true;

// double detections_x [N_CONCENTRIC][N_TRAJECTORIES] = 
// {
//     { 725.1055299918384, 771.5396856205438, 771.6904224552521 }, 
//     { 698.4262600734249, 795.2622213877456, 795.6520272728758 }, 
//     { 670.693168980946, 820.7832184584738, 821.2093480308074 },  
//     { 642.0671499920185, 847.9317137615013, 848.29963105924 },   
//     { 612.4153789518864, 876.7226504020744, 876.7226504020744 }, 
//     { 582.0179121590032, 907.3281294945984, 906.5510527147919 }, 
//     { 550.4772638107207, 939.173347626747, 937.7133022610888 },
//     { 518.3398170422156, 972.3876592703886, 970.1353936769818 },
//     { 485.40813612761605, 1006.8742611611831, 1004.0922088642806 },
//     { 451.37091191340573, 1042.8990418202834, 1038.8268714753897 },
//     { 416.9485508661006, 1080.0279240952095, 1074.5822997839978 },
//     { 381.8565665459982, 1118.1434334540018, 1111.6884722179034 },
//     { 345.72498356434284, 1157.1252013759333, 1149.6744049982265 },
//     { 309.4050658298307, 1197.2563773178788, 1188.4425245586974 },
//     { 272.10667576995746, 1238.0030032992593, 1227.8933242300425 },
//     { 234.74228745329765, 1279.2237370457656, 1268.3646438815367 },
//     { 196.47345098192682, 1321.1596096461878, 1309.3068333085625 },
//     { 157.7927647321102, 1363.6107239128285, 1350.6030739747982 },
//     { 118.75383879261847, 1406.0019784046433, 1392.135092507062 },
//     { 79.41080901727196, 1448.4606529159532, 1434.175862449718 }
// };

// double detections_y [N_CONCENTRIC][N_TRAJECTORIES] =
// {
//     { 776.0051026341512, 778.8451372568782, 721.2680391599756 },
//     { 800.240913108601, 805.9940292803323, 694.3233226038352 },
//     { 823.3104805175577, 831.5704357267998, 668.8013007922705 },
//     { 845.3231340711927, 855.5716791556115, 644.7708094984225 },
//     { 866.0623627669494, 877.8333676121913, 622.1666323878087 },
//     { 885.7866641632368, 897.9995259104935, 601.17873843469 },
//     { 903.930756327458, 916.4855685868596, 581.8699427400409 },
//     { 921.1068661157872, 932.9965273010325, 564.3002195728809 },
//     { 936.9950415720637, 947.4629432397287, 548.9697798974721 },
//     { 951.0489187948275, 959.308746355139, 535.1069142253788 },
//     { 964.2258906617257, 968.8551331758218, 523.1468962810274 },
//     { 976.0407317381072, 976.0407317381071, 513.7682301961081 },
//     { 985.7662212573064, 980.8095977306959, 506.5161812577401 },
//     { 994.7286333545467, 982.331084765873, 501.4358178284813 },
//     { 1001.4318409755602, 981.1991971675144, 498.5681590244397 },
//     { 1007.4596855065658, 977.372461277338, 498.85443270173425 },
//     { 1011.0600688196926, 969.8196995467293, 501.56516706720095 },
//     { 1013.0486466347284, 958.2927735164487, 506.7307098460164 },
//     { 1013.4089671232633, 943.6940998822467, 514.376310675376 },
//     { 1012.1261851420617, 924.7933532152196, 525.7158292674565 },
// };

double detections_x [N_CONCENTRIC][N_TRAJECTORIES] = 
{
    { 725.1055299918384 }, 
    { 698.4262600734249 },  
    { 670.693168980946 },   
    { 642.0671499920185 },  
    { 612.4153789518864 },  
    { 582.0179121590032 },  
    { 550.4772638107207 },  
    { 518.3398170422156 },  
    { 485.40813612761605 }, 
    { 451.37091191340573 }, 
    { 416.9485508661006 },  
    { 381.8565665459982 },  
    { 345.72498356434284 }, 
    { 309.4050658298307 },  
    { 272.10667576995746 }, 
    { 234.74228745329765 }, 
    { 196.47345098192682 }, 
    { 157.7927647321102 },  
    { 118.75383879261847 },
    { 79.41080901727196 }
};

double detections_y [N_CONCENTRIC][N_TRAJECTORIES] =
{
    {776.0051026341512},
    {800.240913108601},
    {823.3104805175577},
    {845.3231340711927},
    {866.0623627669494},
    {885.7866641632368},
    {903.930756327458},
    {921.1068661157872},
    {936.9950415720637},
    {951.0489187948275},
    {964.2258906617257},
    {976.0407317381072},
    {985.7662212573064},
    {994.7286333545467},
    {1001.4318409755602},
    {1007.4596855065658},
    {1011.0600688196926},
    {1013.0486466347284},
    {1013.4089671232633},
    {1012.1261851420617}
};

double angle_of_point_relative_to_origin(double origin_x, double origin_y, double x, double y) 
{
    double angle_rad = atan2(y - origin_y, x - origin_x);
    angle_rad = fmod((angle_rad + 2 * PI), (2 * PI));
    return angle_rad;
}

void cartesian2polar(double origin_x, double origin_y, double x, double y, double* distance, double* angle)
{
    *angle = angle_of_point_relative_to_origin(origin_x, origin_y, x, y);
    *distance = sqrt(pow((origin_x - x), 2) + pow((origin_y - y), 2));
}

void polar2cartesian(double origin_x, double origin_y, double distance, double angle, double* x, double* y)
{
    *x = distance * cos(angle) + origin_x;
    *y = distance * sin(angle) + origin_y;
}
    
double random_point_on_sensor(double origin_x, double origin_y, double x, double y, 
    double sensor_segment_angle, double* cart_x, double* cart_y)
{
    double distance, angle;
    cartesian2polar(origin_x, origin_y, x, y, &distance, &angle);
    double a_min = (int) angle / sensor_segment_angle * sensor_segment_angle;
    double a_max = (int) ((angle / sensor_segment_angle) + 1) * sensor_segment_angle;
    double a_delta = a_max - a_min;
    double a = a_min + ((rand() % 10000) / 10000.0) * a_delta;
    polar2cartesian(origin_x, origin_y, distance, a, cart_x, cart_y);
}

int circle_from_points(double p1_x, double p1_y, double p2_x, double p2_y, 
double p3_x, double p3_y, double* r, double* center_x, double* center_y)
{
    double temp = p2_x * p2_x + p2_y * p2_y;
    double bc = (p1_x * p1_x + p1_y * p1_y - temp) / 2;
    double cd = (temp - p3_x * p3_x - p3_y * p3_y) / 2;
    double det = (p1_x - p2_x) * (p2_y - p3_y) - (p2_x - p3_x) * (p1_y - p2_y);

    if (abs(det) < 1.0e-6)
        return 0;

    *center_x = (bc*(p2_y - p3_y) - cd*(p1_y - p2_y)) / det;
    *center_y = ((p1_x - p2_x) * cd - (p2_x - p3_x) * bc) / det;

    *r = sqrt(pow((*center_x - p1_x), 2) + pow((*center_y - p1_y), 2));

    return 1;
}

// ------------
// --- MAIN ---
// ------------

int main(int argc, char *argv[])
{
    srand((unsigned int)time(NULL));

    int trajectory_found = 0;
    int* trajectory_radii;
    int* trajectory_directions;
    int* trajectory_centers_x;
    int* trajectory_centers_y;
    int** trajectory_points_x;
    int** trajectory_points_y;

    int points_needed = (int) N_CONCENTRIC * MIN_PERC_COVERAGE_FOR_TRAJ;

    double origin_x = W / 2;
    double origin_y = H / 2;
    double rmin = W * 2 / 3 / 2;
    double rmax = W * 2 / 3;

    double sensor_segment_angle = 2 * PI / SENSOR_DENSITY;

    int i;
    int j;
    int k;

    for (int i = 0; i < N_TRAJECTORIES; i++) {
        double p0x = detections_x[N_CONCENTRIC - 1][i];
        double p0y = detections_y[N_CONCENTRIC - 1][i];

        for (int j = 0; j < N_TRAJECTORIES; j++) {
            double p1x = detections_x[N_CONCENTRIC - 2][j];
            double p1y = detections_y[N_CONCENTRIC - 2][j];

            for (int k = 0; k < N_TRAJECTORIES; k++) {
                double p2x = detections_x[N_CONCENTRIC - 3][k];
                double p2y = detections_y[N_CONCENTRIC - 3][k];

                double best_r;

                double best_pp0x;
                double best_pp0y;
                double best_pp1x;
                double best_pp1y;
                double best_pp2x;
                double best_pp2y;

                double best_center_x;
                double best_center_y;

                double min_avg_error = DBL_MAX;

                double angle_p0 = angle_of_point_relative_to_origin(origin_x, origin_y, p0x, p0y);
                double angle_p1 = angle_of_point_relative_to_origin(origin_x, origin_y, p1x, p1y);
                double angle_p2 = angle_of_point_relative_to_origin(origin_x, origin_y, p2x, p2y);
                double angle_reference = angle_p0;

                if ((abs(angle_p0 - angle_p1) < SEED_ANGLE_TOLERANCE || 
                    abs(angle_p0 - angle_p1) > (2 * PI - SEED_ANGLE_TOLERANCE)) &&
                    (abs(angle_p0 - angle_p2) < SEED_ANGLE_TOLERANCE ||
                    abs(angle_p0 - angle_p2) > (2 * PI - SEED_ANGLE_TOLERANCE)))
                {
                    int ii;
                    for (ii = 0; ii < N_SEED_CORRECTIONS; ii++) 
                    {
                        double pp0x, pp0y, pp1x, pp1y, pp2x, pp2y;
                        random_point_on_sensor(origin_x, origin_y, p0x, p0y, sensor_segment_angle, &pp0x, &pp0y);
                        random_point_on_sensor(origin_x, origin_y, p1x, p1y, sensor_segment_angle, &pp1x, &pp1y);
                        random_point_on_sensor(origin_x, origin_y, p2x, p2y, sensor_segment_angle, &pp2x, &pp2y);

                        double r, center_x, center_y;
                        int success = circle_from_points(pp0x, pp0y, pp1x, pp1y, pp2x, pp2y, &r, &center_x, &center_y);
                        if (!success)
                            continue;

                        double distance_center_origin = sqrt(pow((center_x - origin_x), 2) + pow((center_y - origin_y), 2));
                        double center_error = abs(distance_center_origin - r);
                        if (center_error > CENTER_TOLERANCE)
                            continue;

                        int points_on_seed_trajectory = 3;
                        double cumul_error = 
                                    pow(sqrt(pow((pp0x - p0x), 2) + pow((pp0y - p0y), 2)), 2) +
                                    pow(sqrt(pow((pp1x - p1x), 2) + pow((pp1y - p1y), 2)), 2) +
                                    pow(sqrt(pow((pp2x - p2x), 2) + pow((pp2y - p2y), 2)), 2);

                        if (r < rmax && r > rmin)
                        {
                            int layer, detection;
                            for (layer = N_CONCENTRIC-4; layer >= 0; layer--)
                            {
                                double min_error = DBL_MAX;
                                
                                for (detection = 0; detection < N_CONCENTRIC; detection++) 
                                {
                                    double det_x = detections_x[layer][detection];
                                    double det_y = detections_y[layer][detection];

                                    double distance_center_detection = sqrt(pow((center_x - det_x), 2) + pow((center_y - det_y), 2));

                                    double error = abs(distance_center_detection - r);
    	                            if (error < TOLERANCE)
                                    {
                                        double detection_angle = angle_of_point_relative_to_origin(origin_x, origin_y, det_x, det_y);
                                        double sq_err = error * error;

                                        if ((abs(angle_reference - detection_angle) < TRAJECTORY_ANGLE_TOLERANCE) ||
                                            abs(angle_reference - detection_angle) > (2 * PI - TRAJECTORY_ANGLE_TOLERANCE))
                                        {
                                            if (sq_err < min_error)
                                                min_error = sq_err;
                                        }
                                    }
                                }

                                if (min_error < DBL_MAX)
                                {
                                    cumul_error += min_error;
                                    points_on_seed_trajectory ++;
                                }
                            }
                        }

                        if (points_on_seed_trajectory >= points_needed)
                        {
                            double avg_err = cumul_error / points_on_seed_trajectory;
                            if (avg_err < min_avg_error)
                            {
                                min_avg_error = avg_err;
                                best_r = r;
                                best_pp0x = pp0x;
                                best_pp0y = pp0y;
                                best_pp1x = pp1x;
                                best_pp1y = pp1y;
                                best_pp2x = pp2x;
                                best_pp2y = pp2y;
                                best_center_x = center_x;
                                best_center_y = center_y;
                                printf("%f", center_x);
                                printf("%f", center_y);
                                printf("nekej");
                            }
                                
                        }
                        
                    }
                }
            }
        }
    }

    printf("H")
    
    return 0;
}