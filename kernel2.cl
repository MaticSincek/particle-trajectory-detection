#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics:enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define PI 3.14159265
#define N_CONCENTRIC 23

int random(int r1, int r2)
{
    int t = r1 ^ (r1 << 11);
    int result = r2 ^ (r2 >> 19) ^ (t ^ (t >> 8));
    return result;
}

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

int get_orientation(double x1, double x2, double x3, double y1, double y2, double y3)
{
	double val = ((y2 - y1) * (x3 - x2)) - ((x2 - x1) * (y3 - y2));
	if (val > 0)
		return 1;
	else if (val < 0)
		return 0;
	else
		return 0;
}
    
void random_point_on_sensor(double x, double y, double sensor_segment_angle, double* cart_x, double* cart_y, int* r1, int* r2)
{
    double distance, angle;
    cartesian2polar(x, y, &distance, &angle);
    double a_min = ((int) (angle / sensor_segment_angle)) * sensor_segment_angle;
    double a_max = ((int) ((angle / sensor_segment_angle) + 1)) * sensor_segment_angle;
    int irnd = random(*r1, *r2);
    *r1 = *r2;
    *r2 = irnd;
    double a_delta = a_max - a_min;
    double a = a_min + ((double)(irnd % 100000) / 100000) * a_delta;
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

__kernel void seed_calculation
                       (__global double *det_x,
                        __global double *det_y,
                        __global int    *arr_data,
                        __local  double *ldet_x,
                        __local  double *ldet_y,
                        __local  int    *larr_data,
                                 int     nlayers,
                                 int     ngroups,
                        __global double *x0,
                        __global double *x1,
                        __global double *x2,
                        __global double *y0,
                        __global double *y1,
                        __global double *y2,
                        __global int    *ntrajectories)
{
    double realW = 20000;
    double realH = 20000;
    double SENSOR_DENSITY = 3600;
    int    N_SEED_CORRECTIONS = 30 * 30 * 5;
    double TOLERANCE = 50 * 50;
    double CENTER_TOLERANCE = 10;
    double INITIAL_CENTER_TOLERANCE = 2250; //2250
    double TRAJECTORY_ANGLE_TOLERANCE = PI / 4;
    double GPU_SLICE_ANGLE = PI / 6;
    double SEED_ANGLE_TOLERANCE = PI / 18;
    double MIN_PERC_COVERAGE_FOR_TRAJ = 0.9;
    double DETECTION_FAIL_RATE = 0;
    bool   WITH_SENSORS = true;
    int    NUM_GRPS = ngroups;
    int    NTHREADS = 512;

    int lid = get_local_id(0);
    int lsize = get_local_size(0);
    int gid = get_global_id(0);
    int grp = get_group_id(0);
    int grplo = (grp - 1 + NUM_GRPS) % NUM_GRPS;
    int grphi = (grp + 1 + NUM_GRPS) % NUM_GRPS;

    int r1 = 12 + gid;
    int r2 = 27 + gid;

    if(lid == 1)
    {
        int i;
        for (i = 0; i < nlayers; i++) 
        {
            larr_data[i] = 0;
        }
    }

    if(lid == 1) 
    {
        int l, i;
        int p = 0;
        int lp = 0;
        for(l = 0; l < nlayers; l++) 
        {
            for (i = 0; i < arr_data[l]; i++)
            {
                double angle = angle_of_point_relative_to_origin(det_x[p], det_y[p]);
                int nsegment = (int)(angle / 2 / PI * NUM_GRPS);
                
                if ( nsegment == grp || nsegment == grplo || nsegment == grphi)
                {
                    ldet_x[lp] = det_x[p];
                    ldet_y[lp] = det_y[p];
                    larr_data[l] = larr_data[l] + 1;
                    lp++;
                }
                p++;
            }
        }

        int sequential = -1;
        for(l = 0; l < nlayers; l++) 
        {
            larr_data[l] = sequential + larr_data[l];
            sequential = larr_data[l];
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    int istart = larr_data[nlayers-2] + 1;
    int jstart = larr_data[nlayers-3] + 1;
    int kstart = larr_data[nlayers-4] + 1;

    int iend = larr_data[nlayers-1];
    int jend = larr_data[nlayers-2];
    int kend = larr_data[nlayers-3];

    int irange = iend - istart + 1;
    int jrange = jend - jstart + 1;
    int krange = kend - kstart + 1;

    int combinations = irange * jrange * krange;
    if (irange < 0 || jrange < 0 || krange < 0)
        combinations = 0;

    int npasses = -1;
    if (combinations % NTHREADS == 0) 
    {
        npasses = combinations / NTHREADS;
    } else 
    {
        npasses = combinations / NTHREADS + 1;
    }

    int pass;
    for (pass = 0; pass < npasses; pass++)
    {
        int iteration = pass * NTHREADS + lid;
        if(iteration >= combinations)
            break;

        int i = iteration % irange;
        iteration = iteration / irange;
        int j = iteration % jrange;
        iteration = iteration / jrange;
        int k = iteration % krange;

        double p0x = ldet_x[istart + i];
        double p0y = ldet_y[istart + i];

        double p1x = ldet_x[jstart + j];
        double p1y = ldet_y[jstart + j];

        double p2x = ldet_x[kstart + k];
        double p2y = ldet_y[kstart + k];

        double angle_p0 = angle_of_point_relative_to_origin(p0x, p0y);
        double angle_p1 = angle_of_point_relative_to_origin(p1x, p1y);
        double angle_p2 = angle_of_point_relative_to_origin(p2x, p2y);
        double angle_reference = angle_p0;

        double r, center_x, center_y;
        int success = circle_from_points(p0x, p0y, p1x, p1y, p2x, p2y, &r, &center_x, &center_y);
        if (!success)
            continue;
    
        double distance_center_origin = sqrt(pow(center_x, 2) + pow(center_y, 2));
        double center_error = fabs(distance_center_origin - r);

        if (center_error > INITIAL_CENTER_TOLERANCE)
            continue;

        if ((fabs(angle_p0 - angle_p1) < SEED_ANGLE_TOLERANCE || 
            fabs(angle_p0 - angle_p1) > (2 * PI - SEED_ANGLE_TOLERANCE)) &&
            (fabs(angle_p0 - angle_p2) < SEED_ANGLE_TOLERANCE ||
            fabs(angle_p0 - angle_p2) > (2 * PI - SEED_ANGLE_TOLERANCE)))
        {
            int old_ntrajectories = atomic_add(ntrajectories, 1);
            x0[old_ntrajectories] = p0x;
            x1[old_ntrajectories] = p1x;
            x2[old_ntrajectories] = p2x;
            y0[old_ntrajectories] = p0y;
            y1[old_ntrajectories] = p1y;
            y2[old_ntrajectories] = p2y;
        }
    }
}

// =============================================================================================================
// =============================================================================================================
// =============================================================================================================
// =============================================================================================================

__kernel void trajectory_calculation
                       (__global double *det_x,
                        __global double *det_y,
                        __global int    *arr_data,
                                 int     nlayers,
                                 int     ngroups,	
                        __global double *traj_x,
                        __global double *traj_y,
                        __global double *traj_r,
                        __global int    *ntrajectories,
                        __global double *x0,
                        __global double *x1,
                        __global double *x2,
                        __global double *y0,
                        __global double *y1,
                        __global double *y2
                        )
{
    double realW = 20000;
    double realH = 20000;
    double SENSOR_DENSITY = 3600;
    int    N_SEED_CORRECTIONS = 4;
    double TOLERANCE = 50 * 50;
    double CENTER_TOLERANCE = 10;
    double INITIAL_CENTER_TOLERANCE = 2250;
    double TRAJECTORY_ANGLE_TOLERANCE = PI / 4;
    double GPU_SLICE_ANGLE = PI / 6;
    double SEED_ANGLE_TOLERANCE = PI / 18;
    double MIN_PERC_COVERAGE_FOR_TRAJ = 0.9;
    double DETECTION_FAIL_RATE = 0;
    bool   WITH_SENSORS = true;
    int    NUM_GRPS = ngroups;
    int    NTHREADS = 512 * 16;

    int gid = get_global_id(0);

    int r1 = gid * gid * gid + (gid + 1) * 587684321;
    int r2 = 14;

    int nseeds = *ntrajectories;

    int points_needed = (int) ((nlayers - 1) * MIN_PERC_COVERAGE_FOR_TRAJ);

    double rmin = realW * 2 / 3 / 2;
    double rmax = realW * 2 / 3;

    double sensor_segment_angle = 2 * PI / SENSOR_DENSITY;

    int l;
    if(gid == 1)
    {
        int sequential = -1;
        for(l = 0; l < nlayers; l++) 
        {
            arr_data[l] = sequential + arr_data[l];
            sequential = arr_data[l];
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int seed = 1; seed < nseeds; seed ++)
    {
        double p0x = x0[seed];
        double p0y = y0[seed];

        double p1x = x1[seed];
        double p1y = y1[seed];

        double p2x = x2[seed];
        double p2y = y2[seed];

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

        double angle_reference = angle_of_point_relative_to_origin(p0x, p0y);
        
        int passes = N_SEED_CORRECTIONS;
        for (int p = 0; p < passes; p++)
        {

            double pp0x, pp0y, pp1x, pp1y, pp2x, pp2y;
            random_point_on_sensor(p0x, p0y, sensor_segment_angle, &pp0x, &pp0y, &r1, &r2);
            random_point_on_sensor(p1x, p1y, sensor_segment_angle, &pp1x, &pp1y, &r1, &r2);
            random_point_on_sensor(p2x, p2y, sensor_segment_angle, &pp2x, &pp2y, &r1, &r2);

            double r, center_x, center_y;
            int success = circle_from_points(pp0x, pp0y, pp1x, pp1y, pp2x, pp2y, &r, &center_x, &center_y);

            if (!success)
                continue;

            int points_on_seed_trajectory = 0;
            double cumul_error = 0;

            if (r < rmax && r > rmin)
            {
                int layer, detection;
                for (layer = nlayers-1; layer > 0; layer--)
                {
                    double min_error = DBL_MAX;
            
                    for (detection = arr_data[layer-1] + 1; detection <= arr_data[layer]; detection++)  
                    {
                        double x = det_x[detection];
                        double y = det_y[detection];

                        double distance_center_detection = sqrt(pow((center_x - x), 2) + pow((center_y - y), 2));

                        double error = pow(fabs(distance_center_detection - r), 2);
                        if (error < TOLERANCE)
                        {
                            double detection_angle = angle_of_point_relative_to_origin(x, y);

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
                double avg_err = cumul_error / points_on_seed_trajectory;
                if (avg_err < min_avg_error)
                {
                    best_r = r;
                    best_pp0x = pp0x;
                    best_pp0y = pp0y;
                    best_pp1x = pp1x;
                    best_pp1y = pp1y;
                    best_pp2x = pp2x;
                    best_pp2y = pp2y;
                    best_center_x = center_x;
                    best_center_y = center_y;

                    found_trajectory = true;
                    min_avg_error = avg_err;
                }
            
            }
        }
        if(found_trajectory && gid == 0) 
        {
            int orientation = get_orientation(best_pp2x, best_pp1x, best_pp0x, best_pp2y, best_pp1y, best_pp0y);
            double angle = angle_of_point_relative_to_origin(best_center_x, best_center_y);
            printf("%f,%f,%f,%d,%f:", best_center_x, best_center_y, best_r, orientation, angle);
        }
    }
}

// distances from center
// 1994.739508, 1846.524154, 1369.673020, 869.950641, 515.229198, 198.991633, 1994.739508, 1997.309197, 77.138871, 2212.443208, 288.418278, 160.810305, 339.408508, 2036.430916, 394.350918, 1196.517797, 453.035015, 394.350918, 151.268840, 595.565454, 1104.089715, 2093.411048, 1997.080040, 198.991633, 2002.234658, 1997.080040, 198.991633, 288.418278, 2002.234658, 2093.411048, 2010.419416, 198.991633, 1196.517797, 241.563171, 339.408508, 869.950641, 1326.011554, 339.408508, 1008.159180, 1449.831114, 1997.309197, 241.563171, 909.032821, 1192.061599, 1369.673020, 1449.831114, 2021.787078, 288.418278, 1997.309197, 1449.831114, 394.350918, 595.565454, 1997.309197, 288.418278, 394.350918, 1994.739508, 376.214695, 160.810305, 37.371984, 2036.430916, 241.563171, 1027.390207, 1369.673020, 1196.517797, 160.810305, 1994.931570, 1525.356929, 1458.726073, 1994.739508, 2075.653870, 1449.831114, 264.303414, 1369.673020, 160.810305, 394.350918, 486.729861, 794.045430, 1369.673020, 1719.281994, 1994.739508, 794.045430, 241.563171, 2075.653870, 2036.430916, 453.035015, 869.950641, 1449.831114, 1104.089715, 1997.080040, 1994.739508, 37.371984, 1369.673020, 1285.142654, 241.563171, 1994.931570, 1369.673020, 2021.787078, 198.991633, 595.565454, 198.991633, 198.991633, 288.418278, 2158.584755, 1525.356929, 339.408508, 580.686604, 1525.356929, 580.686604, 807.019796, 339.408508, 453.035015, 595.565454, 1525.356929, 909.032821, 198.991633, 151.268840, 339.408508, 453.035015,
