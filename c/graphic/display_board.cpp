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

#include "./display_board.h"

#include <cmath>

#ifdef CMAKE_DEF_WITH_OPENCV
    #include <opencv2/opencv.hpp>
#endif

DisplayBoard::DisplayBoard() :
    object_pos(0.), target_pos(0.), current_pos(0.)
{

}

void DisplayBoard::update(const float &object_pos, const float &target_pos, const float &current_pos)
{
    this->object_pos = object_pos;
    this->target_pos = target_pos;
    this->current_pos = current_pos;
}

void DisplayBoard::show()
{
#ifdef CMAKE_DEF_WITH_OPENCV
    cv::Mat im2show = cv::Mat::zeros(384, 640, CV_8UC3);
    im2show = ~im2show;

    int obj_low = (int) std::floor(object_pos);
    for (int i = obj_low - 4; i < obj_low + 4; i++)
    {
        float x = (object_pos - i) * 100 + 320;
        cv::line(im2show, {(int)x, 200}, {(int)x, 210}, cv::Scalar(20, 20, 20), 2);
    }
    cv::line(im2show, {0, 210}, {640, 210}, cv::Scalar(20, 20, 20), 2);

    cv::circle(im2show, {320, 200}, 10, cv::Scalar(20, 200, 20), -1);

    cv::circle(im2show, {(int)((object_pos - current_pos) * 100) + 320, 200}, 9, cv::Scalar(20, 200, 200), -1);

    cv::circle(im2show, {(int)((object_pos - target_pos) * 100) + 320, 200}, 9, cv::Scalar(50, 50, 200), 2);

    cv::imshow("status", im2show);
    cv::waitKey(1);
#endif
}