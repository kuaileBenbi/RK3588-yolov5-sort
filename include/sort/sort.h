#pragma once
#include <opencv2/core.hpp>
#include "common.h"

class TrackingSession {
    public:
        virtual ~TrackingSession() {};
        virtual std::vector<TrackingBox> Update(const std::vector<DetectionBox> &dets) = 0;
};

#ifdef __cplusplus
extern "C" {
#endif

TrackingSession *CreateSession(int max_age, int min_hits, float iou_threshold);
void  ReleaseSession(TrackingSession **session_ptr);

#ifdef __cplusplus
}
#endif


