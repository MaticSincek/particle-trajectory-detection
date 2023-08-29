#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics:enable

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
    //printf("%f,  %f\n", a_min, ((double)(irnd % 100000) / 100000) * a_delta);
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

__kernel void trajectory_calculation
                       (__global double *det_x,
                        __global double *det_y,
                        __global int    *arr_data,
                        __local  double *ldet_x,
                        __local  double *ldet_y,
                        __local  int    *larr_data,
                                 int     npoints,
                                 int     nlayers,
                                 int     slice_angle,	
                        __global double *traj_x,
                        __global double *traj_y,
                        __global double *traj_r,
                        __global int    *ntrajectories)
{

    double realW = 20000;
    double realH = 20000;
    double SENSOR_DENSITY = 3600;
    int    N_SEED_CORRECTIONS = 30*30;
    double TOLERANCE = 50 * 50;
    double CENTER_TOLERANCE = 10;
    double TRAJECTORY_ANGLE_TOLERANCE = PI / 4;
    double GPU_SLICE_ANGLE = PI / 6;
    double SEED_ANGLE_TOLERANCE = PI / 18;
    double MIN_PERC_COVERAGE_FOR_TRAJ = 0.9;
    double DETECTION_FAIL_RATE = 0;
    bool   WITH_SENSORS = true;
    int    NUM_GRPS = 12;
    int    NTHREADS = 256;

    int lid = get_local_id(0);
    int lsize = get_local_size(0);
    int gid = get_global_id(0);
    int grp = get_group_id(0);
    int grplo = (grp - 1 + NUM_GRPS) % NUM_GRPS;
    int grphi = (grp + 1 + NUM_GRPS) % NUM_GRPS;

    int r1 = 12 + gid;
    int r2 = 27 + gid;

    int nslices = PI / slice_angle;

    if(lid == 0)
    {
        int i;
        for (i = 0; i < nlayers; i++) 
        {
            larr_data[i] = 0;
        }
    }

    if(lid == 0) 
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

    int points_needed = (int) ((nlayers - 1) * MIN_PERC_COVERAGE_FOR_TRAJ);

    double rmin = realW * 2 / 3 / 2;
    double rmax = realW * 2 / 3;

    double sensor_segment_angle = 2 * PI / SENSOR_DENSITY;

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
    int npasses = combinations % NTHREADS == 0 ? combinations / NTHREADS : combinations / NTHREADS + 1;
    
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

        // if (grp == 11)
        //     printf("(%f, %f), (%f, %f), (%f, %f)\n\n", p0x, p0y, p1x, p1y, p2x, p2y);

        bool found_trajectory = false;

        double best_r;
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
                random_point_on_sensor(p0x, p0y, sensor_segment_angle, &pp0x, &pp0y, &r1, &r2);
                random_point_on_sensor(p1x, p1y, sensor_segment_angle, &pp1x, &pp1y, &r1, &r2);
                random_point_on_sensor(p2x, p2y, sensor_segment_angle, &pp2x, &pp2y, &r1, &r2);

                double r, center_x, center_y;
                int success = circle_from_points(pp0x, pp0y, pp1x, pp1y, pp2x, pp2y, &r, &center_x, &center_y);
                if (!success)
                    continue;
            
                double distance_center_origin = sqrt(pow(center_x, 2) + pow(center_y, 2));
                double center_error = fabs(distance_center_origin - r);

                // if (center_error < 10)
                // printf("%d  %f\n", gid, center_error);

                if (center_error > CENTER_TOLERANCE)
                    continue;

                int points_on_seed_trajectory = 0;
                double cumul_error = 0;

                if (r < rmax && r > rmin)
                {
                    int layer, detection;
                    for (layer = nlayers-1; layer > 0; layer--)
                    {
                        double min_error = DBL_MAX;
                
                        for (detection = larr_data[layer-1] + 1; detection <= larr_data[layer]; detection++)  
                        {
                            double det_x = ldet_x[detection];
                            double det_y = ldet_y[detection];

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
                    double avg_err = cumul_error / points_on_seed_trajectory;
                    if (avg_err < min_avg_error)
                    {
                        best_r = r;
                        best_center_x = center_x;
                        best_center_y = center_y;

                        found_trajectory = true;
                        min_avg_error = avg_err;
                    }
                
                }
        
            }
        }
        if(found_trajectory) 
        {
            // int prev = atom_add(ntrajectories, 1);
            // traj_x[*ntrajectories] = best_center_x;
            // traj_y[*ntrajectories] = best_center_y;
            // traj_r[*ntrajectories] = best_r;
            printf("GID %d found trajectory (%f, %f), r = %f\n\n", gid, best_center_x, best_center_y, best_r);
        }
    }
}
