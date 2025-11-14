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
#include <iostream>
#include <chrono>
#include <cmath>
#include <deque>
#include "./chassis.h"

#include "kalman/kalman.h"
#include "kalman/AdaptiveEKF.h"

using namespace std::chrono;
/*
这个make之后一堆错误

大体上顺序：传感器数据读取->卡尔曼滤波进行状态估计->机器人控制->误差评估与重置
模式零和模式一卡尔曼滤波（对象初始化->预测更新->模式适配）
模式二用拓展卡尔曼处理非线性运动

*/

Kalman<1,2> kf;
AdaptiveEKF<2,1> aekf;

Eigen::Vector2d state;
float previous_robot_position=0.0f;
float predicted_position = 0.0f;

steady_clock::time_point program_start_time;
steady_clock::time_point last_time;
bool first_run =true;

float current_error =0.0f;
float average_error =0.0f;
float exponential_error =0.0f;
std::deque<float> error_history;
const size_t ERROR_WINDOW_SIZE =100;
float error_sum=0.0f;

steady_clock::time_point last_reset_time;
steady_clock::time_point poor_performance_start_time;
bool in_poor_performance=false;
const float EXPONENTIAL_ERROR_THRESHOLD=0.1f;
const float RESET_TIME_THRESHOLD=5.0f;

auto mode2_state_transition = [](const std::vector<ceres::Jet<double, 2>>& x, 
                                 std::vector<ceres::Jet<double, 2>>& x_next) {
    double t = duration<double>(steady_clock::now().time_since_epoch()).count();
    double speed = 1.57 * sin(3.688 * t) + 2.61;
    x_next[0].a = x[0].a + x[1].a * 0.001; // 位置 = 原位置 + 速度 * 时间步长(0.001s)
    x_next[0].v = x[0].v + x[1].v * 0.001;
    x_next[1].a = speed;                   // 速度由三角函数更新
    x_next[1].v = 0.0;                     // 简化雅克比，实际需计算三角函数导数
};

void updateError(float target_position, float current_position) {
    current_error = std::abs(target_position - current_position);
    error_history.push_back(current_error);
    error_sum += current_error;

    if (error_history.size() > ERROR_WINDOW_SIZE) {
        error_sum -= error_history.front();
        error_history.pop_front();
    }

    average_error = error_sum / error_history.size();
    exponential_error = (1.0f - 0.001f) * exponential_error + 0.001f * current_error;
}

bool checkReset() {
    auto current_time =steady_clock::now();
    if (exponential_error > EXPONENTIAL_ERROR_THRESHOLD) {
        if (!in_poor_performance) {
            in_poor_performance = true;
            poor_performance_start_time = current_time;
        } else {
            float poor_performance_duration = duration<float>(current_time - poor_performance_start_time).count();
            if (poor_performance_duration > RESET_TIME_THRESHOLD) {
                return true;
            }
        }
    } else {
        in_poor_performance = false;
    }
    return false;
}



void performReset(int mode) {
    SET_MODE(mode);
    Eigen::MatrixXd A;
    Eigen::MatrixXd R;
    Eigen::MatrixXd Q;
    Eigen::MatrixXd H;
    Eigen::VectorXd init;
    double t = duration<double>(steady_clock::now().time_since_epoch()).count();

    switch(mode) {
        case 0: { // 模式零：静止
            A = Eigen::MatrixXd::Identity(2, 2); // 状态转移矩阵：位置、速度（速度恒为0）
            H = Eigen::MatrixXd::Zero(2); H(0) = 1; // 测量矩阵：仅观测位置
            R = Eigen::MatrixXd::Identity(1, 1) * 0.01; // 测量噪声（传感器随机误差）
            Q = Eigen::MatrixXd::Identity(2, 2) * 0.001; // 过程噪声
            init = Eigen::VectorXd::Zero(2); // 初始状态：位置0、速度0
            kf.reset(A, H, Q, R,init,t);
            break;
        }
        case 1: { // 模式一：匀速
            A = Eigen::MatrixXd(2, 2);
            A << 1, 0.001,  // 时间步长0.001s（因控制频率1000Hz）
                 0, 1;
            H = Eigen::MatrixXd::Zero(2); H(0) = 1;
            R = Eigen::MatrixXd::Identity(1, 1) * 0.05;
            Q = Eigen::MatrixXd::Identity(2, 2) * 0.01;
            init = Eigen::VectorXd::Zero(2); init(1) = 1.5; // 初始速度假设为1.5m/s（模式一速度在1~2m/s之间）
            kf.reset(A, H, Q, R,init,t);
            break;
        }
        case 2: { // 模式二：三角函数变速（需EKF）
            A = Eigen::MatrixXd::Identity(2, 2); // 状态转移矩阵（位置、速度）
            H = Eigen::MatrixXd::Zero(2); H(0) = 1;
            R = Eigen::MatrixXd::Identity(1, 1) * 0.1;
            Q = Eigen::MatrixXd::Identity(2, 2) * 0.1;
            init = Eigen::VectorXd::Zero(2); init(1) = 2.61; // 初始速度为三角函数均值2.61
            aekf =AdaptiveEKF<2,1>(init);
            break;
        }
    }
}

int main() {
    const int mode = 2;
    SET_MODE(mode);
    ENABLE_LOG(true);

    program_start_time = steady_clock::now();
    last_time = program_start_time;
    last_reset_time = program_start_time;

    // 初始化KF/EKF
    performReset(mode);

    while (true) {
        float sensor_data;
        float current_position;

        // 读取传感器数据（含10ms延迟，需考虑时序）
        GET_SENSOR_DISTANCE(sensor_data);
        GET_CURRENT_POSITION(current_position);

       // 卡尔曼滤波处理
        if (mode == 2) {
            // 模式二：AdaptiveEKF
            std::vector<ceres::Jet<double, 2>> jet_state, jet_predict;
            // 初始化
            jet_state.resize(2);
            for (int i = 0; i < 2; ++i) {
                jet_state[i].a = aekf.getX()(i);
                jet_state[i].v = 0.0;
                jet_state[i].v[i] = 1.0; // 雅克比初始化为单位矩阵
            }
            // 预测步骤
            jet_predict=aekf.predict(mode2_state_transition, jet_state);
            // 构造测量值（目标位置 = 自身位置 + 传感器距离）
            Eigen::VectorXd measurement(1);
            measurement(0) = current_position + sensor_data;
            // 更新步骤
            state = aekf.update(mode2_state_transition, measurement, jet_predict);
        } else {
            // 模式零、模式一
            Eigen::VectorXd measurement(1);
            measurement(0) = current_position + sensor_data;
            state = kf.update(measurement, duration<double>(steady_clock::now().time_since_epoch()).count());
        }

        // 计算目标位置并控制机器人移动
        float target_position = current_position + (state(0) - current_position);
        SET_TARGET_POSITION(target_position);

        // 更新误差并检查是否需要重置
        updateError(state(0), current_position);
        if (checkReset()) {
            performReset(mode);
            last_reset_time = steady_clock::now();
        }

        SHOW_DEBUG_IMAGE();
    }

    return 0;
}