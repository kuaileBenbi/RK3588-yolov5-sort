#include <stdio.h>
#include <memory>
#include <sys/time.h>

#include "rkYolov5s.hpp"
#include "rknnPool.hpp"
#include "sort.h"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "postprocess.h"


int main(int argc, char **argv)
{
    if (argc < 4)
    {
        printf("Usage: %s <model path> <video_source> <save mode>\n", argv[0]);
        printf("mode: 1 - display, 2 - save as images, 3 - save as video, 4 - debug mode\n");
        return -1;
    }

    char *model_name = NULL;
    model_name = (char *)argv[1];
    const char *vedio_name = argv[2];
    int mode = std::atoi(argv[3]);

    // 初始化rknn线程池 /Initialize the rknn thread pool
    int threadNum = 3;

    rknnPool<rkYolov5s, cv::Mat, detect_result_group_t> detectPool(model_name, threadNum);
    if (detectPool.init() != 0)
    {
        printf("rknnPool init fail!\n");
        return -1;
    }

    TrackingSession *sess = CreateSession(2, 3, 0.3);

    cv::VideoCapture capture;
    capture.open(vedio_name);

    if (!capture.isOpened())
    {
        printf("Error: Could not open video or camera\n");
        return -1;
    }

    struct timeval time;
    gettimeofday(&time, nullptr);
    auto startTime = time.tv_sec * 1000 + time.tv_usec / 1000;

    int cnum = 20;
    cv::RNG rng(0xFFFFFFFF); // 生成随机颜色
    cv::Scalar_<int> randColor[cnum];
    for (int i = 0; i < cnum; i++)
        rng.fill(randColor[i], cv::RNG::UNIFORM, 0, 256);

    int frames = 0;
    auto beforeTime = startTime;

    while (capture.isOpened())
    {
        cv::Mat img;
        if (capture.read(img) == false)
        {
            printf("read original images failed!\n");
            break;
        }

        if (detectPool.put(img, frames) != 0)
        {
            printf("put original images failed!\n");
            break;
        }

        detect_result_group_t results_group = {cv::Mat(), 0, 0, {}};
        if (frames >= threadNum && detectPool.get(results_group) != 0)
        {
            printf("frames > 3 but get processed images failed!\n");
            break;
        }

        if (mode == 4)
        {
            // printf("frames: %d\n", frames);
            printf("Frame ID: %d\n", results_group.cur_frame_id);

            std::vector<DetectionBox> dets;
            cv::Mat current_tracking_img;
            for (int i = 0; i < results_group.count; i++)
            {
                DetectionBox detection;
                detection.det_name = results_group.results[i].name; 
                detection.score = results_group.results[i].prop;
                detection.box = cv::Rect_<float>(results_group.results[i].box.left,
                                                 results_group.results[i].box.top,
                                                 results_group.results[i].box.right - results_group.results[i].box.left,
                                                 results_group.results[i].box.bottom - results_group.results[i].box.top);
                dets.push_back(detection);
            }
            current_tracking_img = results_group.cur_img;
            auto trks = sess->Update(dets);
            char text[256];
            for (auto &trk : trks)
            {   
                // std::cout<< trk.det_name << std::endl;
                sprintf(text, "%s", trk.det_name.c_str());
                cv::rectangle(current_tracking_img, trk.box, randColor[trk.id % cnum], 2, 8, 0);
                cv::putText(current_tracking_img, text, cv::Point(trk.box.x, trk.box.y + 12), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255));
                std::string filename = "frame_" + std::to_string(results_group.cur_frame_id) + ".jpg";
                cv::imwrite(filename, current_tracking_img);
            }
        }

        frames++;

        if (frames % 100 == 0)
        {
            gettimeofday(&time, nullptr);
            auto currentTime = time.tv_sec * 1000 + time.tv_usec / 1000;
            printf("100帧内平均帧率:\t %f fps/s\n", 120.0 / float(currentTime - beforeTime) * 1000.0);
            beforeTime = currentTime;
        }
    }
    // 清空rknn线程池/Clear the thread pool
    while (true)
    {
        detect_result_group_t results_group;
        if (detectPool.get(results_group) != 0)
            break;
        
        if (mode == 4)
        {
            std::vector<DetectionBox> dets;
            cv::Mat current_tracking_img;
            for (int i = 0; i < results_group.count; i++)
            {
                DetectionBox detection;
                detection.det_name = results_group.results[i].name;
                detection.score = results_group.results[i].prop;
                detection.box = cv::Rect_<float>(results_group.results[i].box.left,
                                                 results_group.results[i].box.top,
                                                 results_group.results[i].box.right - results_group.results[i].box.left,
                                                 results_group.results[i].box.bottom - results_group.results[i].box.top);
                dets.push_back(detection);
            }

            current_tracking_img = results_group.cur_img;
            auto trks = sess->Update(dets);
            char text[256];
            for (auto &trk : trks)
            {
                sprintf(text, "%s", trk.det_name.c_str());
                cv::rectangle(current_tracking_img, trk.box, randColor[trk.id % cnum], 2, 8, 0);
                cv::putText(current_tracking_img, text, cv::Point(trk.box.x, trk.box.y + 12), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255));
                std::string filename = "frame_" + std::to_string(results_group.cur_frame_id) + ".jpg";
                cv::imwrite(filename, current_tracking_img);
            }
        }
        frames++;
    }

    ReleaseSession(&sess);
    capture.release();

    gettimeofday(&time, nullptr);
    auto endTime = time.tv_sec * 1000 + time.tv_usec / 1000;

    printf("Average:\t %f fps/s\n", float(frames) / float(endTime - startTime) * 1000.0);

    return 0;
}
