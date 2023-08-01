#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h> 
#include <limits.h>
#include <float.h>
#include <math.h>

#define PI 3.141592654
#define N_CONCENTRIC 23
#define N_TRAJECTORIES 3

double realW = 20000;
double realH = 20000;
double SENSOR_DENSITY = 3600;
int N_SEED_CORRECTIONS = 30*30;
double TOLERANCE = 50 * 50;
double CENTER_TOLERANCE = 10;
double TRAJECTORY_ANGLE_TOLERANCE = 0.87266;
double SEED_ANGLE_TOLERANCE = 0.17453;
double MIN_PERC_COVERAGE_FOR_TRAJ = 0.9;
double DETECTION_FAIL_RATE = 0;
bool WITH_SENSORS = true;

double detections_x [N_CONCENTRIC][N_TRAJECTORIES] = 
{
    { -284.0741502493142, 242.11759521715774, 246.53956396858283 }, 
    { -429.78116605479255, 368.98414312611857, 375.5540833104356 },
    { -577.8911696478044, 500.73877774724724, 508.32368694731247 },
    { -728.3709698824001, 636.7513474701522, 644.7909042608296 },
    { -881.1870113228226, 776.9461041832825, 784.8971494911285 },
    { -1036.305378481565, 919.4043424324232, 928.5827361856593 },
    { -1191.8304905465952, 1067.4924270473593, 1075.7868918954691 },
    { -1351.2381973339072, 1219.5325399591984, 1226.4477731159407 },
    { -1512.8491008695964, 1375.4426102278676, 1380.5024804688749 },
    { -1927.6730167078151, 1778.5325770449492, 1778.5325770449494 },
    { -2352.7056376322275, 2206.5183635844014, 2195.8457367258497 },
    { -2793.38508166904, 2655.4608873064235, 2635.463030605256 },
    { -3242.212153413043, 3123.9029490082917, 3093.1487442313764 },
    { -3706.3421648724243, 3610.3358934423727, 3562.906003379987 },
    { -4176.639651925584, 4113.202590104299, 4052.765191766302 },
    { -4661.719323282076, 4636.07233304047, 4557.020979923081 },
    { -5156.438469008716, 5172.43045846175, 5079.947196829491 },
    { -5654.516685776853, 5720.411438892358, 5609.207100223201 },
    { -6166.235167437699, 6283.503201602818, 6154.632242860129 },
    { -6685.517747382228, 6854.250408193168, 6709.0847987851885 },
    { -7211.640936897965, 7430.476422975394, 7276.792290491385 },
    { -7749.977370405253, 8019.853394239645, 7844.433289857813 },
    { -8287.609321269236, 8609.035808170529, 8427.295971319409 }
};

double detections_y [N_CONCENTRIC][N_TRAJECTORIES] =
{
    { 281.6058897823873, 318.4008010138488, -314.98927505262947 },
    { 418.67427590500813, 473.1286316864411, -467.930689855729 },
    { 553.2104446258155, 623.9075864743054, -617.7434979733017 },
    { 685.1829903263593, 771.0692066831261, -764.3590058228539 },
    { 814.56089463953, 914.5243305644852, -907.7094605217579 },
    { 941.3135304085353, 1055.791482782658, -1047.728066846524 },
    { 1067.4924270473593, 1191.8304905465952, -1184.3490039789313 },
    { 1189.1826327632825, 1323.9110181506408, -1317.5074420355093 },
    { 1308.1619158185483, 1451.9495948446527, -1447.1395583768983 },
    { 1591.8783686753807, 1756.9353637484369, -1756.9353637484364 },
    { 1861.390926875258, 2032.5542332653306, -2044.0796218599 },
    { 2108.7910720383966, 2280.027955088661, -2303.1141123081507 },
    { 2342.660955465122, 2498.2454573514788, -2536.2237373819885 },
    { 2552.0634312040115, 2686.163572182764, -2748.763505847465 },
    { 2748.759941857953, 2842.8092536713193, -2928.3261943314046 },
    { 2918.625181646061, 2959.194708500388, -3079.538892194882 },
    { 3067.758516462313, 3040.717539064944, -3192.825782504301 },
    { 3205.688857370777, 3086.5665341588647, -3284.3257613710653 },
    { 3313.237670293904, 3085.0587539700336, -3334.741662430016 },
    { 3399.0958282221554, 3044.5445212352824, -3352.339655033407 },
    { 3462.980652163313, 2964.4595000786903, -3323.8974053127963 },
    { 3491.1102472145567, 2816.3720519318535, -3273.3570475844103 },
    { 3509.3491900936947, 2623.833541526142, -3159.221836431192 }
};

double angle_of_point_relative_to_origin(double x, double y) 
{
    double angle_rad = atan2(y, x);
    angle_rad = fmod((angle_rad + 2 * PI), (2 * PI));
    return angle_rad;
}

void cartesian2polar(double x, double y, double* distance, double* angle)
{
    *angle = angle_of_point_relative_to_origin(x, y);
    *distance = sqrt(pow(x, 2) + pow(y, 2));
}

void polar2cartesian(double distance, double angle, double* x, double* y)
{
    *x = distance * cos(angle);
    *y = distance * sin(angle);
}
    
double random_point_on_sensor(double x, double y, double sensor_segment_angle, double* cart_x, double* cart_y)
{
    double distance, angle;
    cartesian2polar(x, y, &distance, &angle);
    double a_min = ((int) (angle / sensor_segment_angle)) * sensor_segment_angle;
    double a_max = ((int) ((angle / sensor_segment_angle) + 1)) * sensor_segment_angle;
    double a_delta = a_max - a_min;
    double a = a_min + ((rand() % 10000) / 10000.0) * a_delta;
    polar2cartesian(distance, a, cart_x, cart_y);
}

int circle_from_points(double p1_x, double p1_y, double p2_x, double p2_y, 
double p3_x, double p3_y, double* r, double* center_x, double* center_y)
{
    double temp = p2_x * p2_x + p2_y * p2_y;
    double bc = (p1_x * p1_x + p1_y * p1_y - temp) / 2;
    double cd = (temp - p3_x * p3_x - p3_y * p3_y) / 2;
    double det = (p1_x - p2_x) * (p2_y - p3_y) - (p2_x - p3_x) * (p1_y - p2_y);

    if (fabs(det) < 1.0e-6)
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

    int trajectory_count = 0;
    double* trajectory_radii;
    int* trajectory_directions;
    double* trajectory_centers_x;
    double* trajectory_centers_y;
    double** trajectory_points_x;
    double** trajectory_points_y;

    trajectory_centers_x = (double*) malloc(N_CONCENTRIC * sizeof(double)); 
    trajectory_centers_y = (double*) malloc(N_CONCENTRIC * sizeof(double)); 
    trajectory_radii = (double*) malloc(N_CONCENTRIC * sizeof(double));

    int points_needed = (int) (N_CONCENTRIC * MIN_PERC_COVERAGE_FOR_TRAJ);

    double rmin = realW * 2 / 3 / 2;
    double rmax = realW * 2 / 3;

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

                bool found_trajectory = false;

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

                double angle_p0 = angle_of_point_relative_to_origin(p0x, p0y);
                double angle_p1 = angle_of_point_relative_to_origin(p1x, p1y);
                double angle_p2 = angle_of_point_relative_to_origin(p2x, p2y);
                double angle_reference = angle_p0;

                if ((fabs(angle_p0 - angle_p1) < SEED_ANGLE_TOLERANCE || 
                    fabs(angle_p0 - angle_p1) > (2 * PI - SEED_ANGLE_TOLERANCE)) &&
                    (fabs(angle_p0 - angle_p2) < SEED_ANGLE_TOLERANCE ||
                    fabs(angle_p0 - angle_p2) > (2 * PI - SEED_ANGLE_TOLERANCE)))
                {
                    int ii;
                    for (ii = 0; ii < N_SEED_CORRECTIONS; ii++) 
                    {
                        double pp0x, pp0y, pp1x, pp1y, pp2x, pp2y;
                        random_point_on_sensor(p0x, p0y, sensor_segment_angle, &pp0x, &pp0y);
                        random_point_on_sensor(p1x, p1y, sensor_segment_angle, &pp1x, &pp1y);
                        random_point_on_sensor(p2x, p2y, sensor_segment_angle, &pp2x, &pp2y);

                        double r, center_x, center_y;
                        int success = circle_from_points(pp0x, pp0y, pp1x, pp1y, pp2x, pp2y, &r, &center_x, &center_y);
                        if (!success)
                            continue;
                            
                        double distance_center_origin = sqrt(pow(center_x, 2) + pow(center_y, 2));
                        double center_error = fabs(distance_center_origin - r);

                        if (center_error > CENTER_TOLERANCE)
                            continue;

                        int points_on_seed_trajectory = 0;
                        double cumul_error = 0;

                        if (r < rmax && r > rmin)
                        {
                            int layer, detection;
                            for (layer = N_CONCENTRIC-1; layer >= 0; layer--)
                            {
                                double min_error = DBL_MAX;
                                
                                for (detection = 0; detection < N_CONCENTRIC; detection++) 
                                {
                                    double det_x = detections_x[layer][detection];
                                    double det_y = detections_y[layer][detection];

                                    double distance_center_detection = sqrt(pow((center_x - det_x), 2) + pow((center_y - det_y), 2));

                                    double error = pow(fabs(distance_center_detection - r), 2);
    	                            if (error < TOLERANCE)
                                    {
                                        double detection_angle = angle_of_point_relative_to_origin(det_x, det_y);

                                        if ((fabs(angle_reference - detection_angle) < TRAJECTORY_ANGLE_TOLERANCE) ||
                                            fabs(angle_reference - detection_angle) > (2 * PI - TRAJECTORY_ANGLE_TOLERANCE))
                                        {
                                            if (error < min_error)
                                                min_error = error;
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
                            if(!found_trajectory)
                            {
                                found_trajectory = true;
                                trajectory_count++;
                            }

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
                                trajectory_centers_x[trajectory_count-1] = center_x;
                                trajectory_centers_y[trajectory_count-1] = center_y;
                                trajectory_radii[trajectory_count-1] = r;
                            }
                                
                        }
                        
                    }
                }
            }
        }
    }

    for (int i = 0; i < trajectory_count; i++)
    {
        printf("\n\n%f, ", trajectory_centers_x[i]);
        printf("%f", trajectory_centers_y[i]);
        printf("\n%f", trajectory_radii[i]);
    }

    return 0;

    // 11619.427754191292
    // (-8324.073546007934, -8107.666118289865)

    // 7416.712571756438
    // (6027.151661446574, -4323.844916261201)

    // 8318.929267998605
    // (6673.505109694799, 4968.541566283304)
}