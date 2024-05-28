#include <stdio.h>
#include <memory>
#include <sys/time.h>

#include "rkYolov5s.hpp"
#include "rknnPool.hpp"
#include "sort.h"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

int main(int argc, char **argv)
{
    if (argc < 4)
    {
        printf("Usage: %s <model path> <video_source> <save mode>\n", argv[0]);
        printf("mode: 1 - display, 2 - save tracking results as images, 3 - save detection results as images, 4 - debug mode\n");
        return -1;
    }

    char *model_name = NULL;
    model_name = (char *)argv[1];
    const char *vedio_name = argv[2];
    int mode = std::atoi(argv[3]);

    // 初始化rknn线程池 /Initialize the rknn thread pool
    int threadNum = 3;

    rknnPool<rkYolov5s, cv::Mat, DetectResultsGroup> detectPool(model_name, threadNum);
    // rknnPool<rkYolov5s, cv::Mat, cv::Mat> detectPool(model_name, threadNum);

    if (detectPool.init() != 0)
    {
        printf("rknnPool init fail!\n");
        return -1;
    }

    TrackingSession *sess = CreateSession(3, 3, 0.25);
    if (sess == nullptr)
    {
        printf("CreateSession failed!\n");
        return -1;
    }

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

    int frames = 0;
    auto beforeTime = startTime;
    // cv::Mat img;
    // DetectResultsGroup results_group; /*必须放到循环里每次重新生成注销 否则会图片会框不对应*/

    while (capture.isOpened())
    {
        cv::Mat img;
        DetectResultsGroup results_group;
        if (capture.read(img) == false)
        {
            printf("read original images failed or work done!\n");
            break;
        }

        if (detectPool.put(img, frames) != 0)
        // if (detectPool.put(img) != 0)
        {
            printf("put original images failed or work done!\n");
            break;
        }

        // if (frames >= threadNum && detectPool.get(img) != 0)
        if (frames >= threadNum && detectPool.get(results_group) != 0)

        {
            printf("frames > 3 but get processed images failed! or work done\n");
            break;
        }
        if (mode == 1 && !results_group.cur_img.empty())
        {   
            auto trks = sess->Update(results_group.dets);
            show_draw_results(results_group.cur_img, trks);
            cv::imshow("src", results_group.cur_img);
            if (cv::waitKey(1) == 'q') // 延时1毫秒,按q键退出/Press q to exit
                return 0;
        }

        if (mode == 2 && !results_group.cur_img.empty())
        {
            auto trks = sess->Update(results_group.dets);
            if (draw_image_track(results_group.cur_img, trks, results_group.cur_frame_id) < 0)
            {
                printf("save tracking results failed!\n");
                break;
            }
        }

        if (mode == 3 && !results_group.cur_img.empty())
        {
            if (draw_image_detect(results_group.cur_img, results_group.dets, results_group.cur_frame_id) < 0)
            {
                printf("save tracking results failed!\n");
                break;
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

    // 清空线程池
    while (true)
    {
        DetectResultsGroup results_group;
        if (detectPool.get(results_group) != 0)
            break;

        if (mode == 1 && !results_group.cur_img.empty())
        {
            cv::imshow("src", results_group.cur_img);
            if (cv::waitKey(1) == 'q') // 延时1毫秒,按q键退出/Press q to exit
                return 0;
        }

        if (mode == 2 && !results_group.cur_img.empty())
        {
            auto trks = sess->Update(results_group.dets);
            if (draw_image_track(results_group.cur_img, trks, results_group.cur_frame_id) < 0)
            {
                printf("save tracking results failed!\n");
                break;
            }
        }

        if (mode == 3 && !results_group.cur_img.empty())
        {
            if (draw_image_detect(results_group.cur_img, results_group.dets, results_group.cur_frame_id) < 0)
            {
                printf("save tracking results failed!\n");
                break;
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
