//
// 2026 Helios CV enter examination
//
/*
 * ██   ██ ███████ ██      ██  ██████  ███████
 * ██   ██ ██      ██      ██ ██    ██ ██
 * ███████ █████   ██      ██ ██    ██ ███████
 * ██   ██ ██      ██      ██ ██    ██      ██
 * ██   ██ ███████ ███████ ██  ██████  ███████
 */

#ifndef CHASSIS_H
#define CHASSIS_H

/*
 * set the mode of the target
 * if mode = 0, the target will stay still
 * if mode = 1, the target will move at a definate speed
 * if mode = 2, the target will move at a chaning speed(speed = 1.57 * sin(3.688 * t) + 2.61) 
*/
void SET_MODE(const int &mode);

/*
 * get your current position
*/
void GET_CURRENT_POSITION(float &result);

/*
 * get how far you are from the target
*/
void GET_SENSOR_DISTANCE(float &result);

/*
 * set where is the target
 * after you set the target position, you will move to the target position
*/
void SET_TARGET_POSITION(const float &target);

/*
 * is log enable is set true, log "Chassis Control" will be enabled
 * otherwise, you will not see the log
*/
void ENABLE_LOG(const bool &log_enable);

/*
 * if you compile this project with OpenCV and this function is called,
 * the debug window(the image show where you are and where the target is) will be shown.
*/
void SHOW_DEBUG_IMAGE();

#endif // CHASSIS_H