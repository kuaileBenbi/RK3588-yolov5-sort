#ifndef COMMON_H
#define COMMON_H

#include <opencv2/core.hpp>

typedef struct _DetectionBox
{
    float score;
    std::string det_name;
    cv::Rect_<int> box;
}DetectionBox;

typedef struct _BOX_RECT
{
    int left;
    int right;
    int top;
    int bottom;
} BOX_RECT;

typedef struct _DetectResultsGroup
{
    cv::Mat cur_img;
    int cur_frame_id;
    int count;
    std::vector<DetectionBox> dets; // 修改为vector
} DetectResultsGroup;

typedef struct _TrackingBox
{
    int id;
    std::string det_name;
    cv::Rect_<float> box;
}TrackingBox;

#endif // COMMON_H