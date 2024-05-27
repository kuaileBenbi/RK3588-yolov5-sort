#include <stdio.h>
#include <memory>
#include <sys/time.h>

#include <iomanip>
#include <sstream>

#include "rkYolov5s.hpp"
#include "rknnPool.hpp"
// #include "sort.h"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

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
    int threadNum = 1;

    rknnPool<rkYolov5s, cv::Mat, DetectResultsGroup> detectPool(model_name, threadNum);
    if (detectPool.init() != 0)
    {
        printf("rknnPool init fail!\n");
        return -1;
    }

    // TrackingSession *sess = CreateSession(10, 3, 0.3);
    // if (sess == nullptr)
    // {
    //     printf("CreateSession failed!\n");
    //     return -1;
    // }

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
    cv::Mat img;
    DetectResultsGroup results_group;

    while (capture.isOpened())
    {
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

        if (frames >= threadNum && detectPool.get(results_group) != 0)
        {
            printf("frames > 3 but get processed images failed!\n");
            break;
        }
        if (mode == 2 && !results_group.cur_img.empty())
        {
            std::ostringstream oss;
            oss << "detect_" << std::setfill('0') << std::setw(4) << results_group.cur_frame_id << ".jpg";
            std::string filename = oss.str();
            char text[256];
            for (auto &det : results_group.dets)
            {
                // std::cout << "cur_frame_id: " << detect_result_group.cur_frame_id << " " << draw_times << std::endl;
                sprintf(text, "%s %.1f%%", det.det_name.c_str(), det.score * 100);
                // cv::rectangle(new_img, det.box, cv::Scalar(256, 0, 0, 256), 3);
                // cv::putText(new_img, text, cv::Point(det.box.x, det.box.y + 12), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255),1.5);
                // draw_times ++ ;
                // cv::rectangle(results_group.cur_img, det.box, cv::Scalar(256, 0, 0, 256), 3);
                // cv::putText(results_group.cur_img, text, cv::Point(det.box.x, det.box.y + 12), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255));
                rectangle(results_group.cur_img, cv::Point(det.box.left, det.box.top), cv::Point(det.box.right, det.box.bottom), cv::Scalar(256, 0, 0, 256), 3);
                putText(results_group.cur_img, text, cv::Point(det.box.left, det.box.top + 12), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255));
            }
            cv::imwrite(filename, results_group.cur_img);
            // cv::imshow("test", results_group.cur_img);
            // if (cv::waitKey(1) == 'q')
            //     break;
        }
        // if (mode == 3 && !results_group.cur_img.empty())
        // {
        //     auto trks = sess->Update(results_group.dets);
        //     char text[256];
        //     for (auto &trk : trks)
        //     {
        //         sprintf(text, "%s", trk.det_name.c_str());
        //         cv::rectangle(results_group.cur_img, trk.box, randColor[trk.id % cnum], 2, 8, 0);
        //         cv::putText(results_group.cur_img, text, cv::Point(trk.box.x, trk.box.y + 12), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255));
        //         std::ostringstream oss;
        //         oss << "track_" << std::setfill('0') << std::setw(4) << results_group.cur_frame_id << ".jpg";
        //         std::string filename = oss.str();
        //         cv::imwrite(filename, results_group.cur_img);
        //     }
        // }

        // auto trks = sess->Update(results_group.dets);
        // char text[256];
        // for (auto &trk : trks)
        // {
        //     sprintf(text, "%s", trk.det_name.c_str());
        //     // cv::rectangle(results_group.cur_img, trk.box, randColor[trk.id % cnum], 2, 8, 0);
        //     // cv::putText(results_group.cur_img, text, cv::Point(trk.box.x, trk.box.y + 12), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255));

        //     if (mode == 4)
        //     {
        //         // std::string filename = "track_" + std::to_string(results_group.cur_frame_id) + ".jpg";
        //         std::ostringstream oss;
        //         oss << "track_" << std::setfill('0') << std::setw(4) << results_group.cur_frame_id << ".jpg";
        //         std::string filename = oss.str();
        //         cv::imwrite(filename, results_group.cur_img);
        //     }
        //     else if (mode == 1)
        //     {
        //         // 定义缩小后的尺寸 (例如：原始尺寸的一半)
        //         cv::Size new_size(results_group.cur_img.cols / 2, results_group.cur_img.rows / 2);
        //         // 创建用于保存缩小图像的Mat对象
        //         cv::Mat save_resized_img;
        //         // 缩小图像
        //         cv::resize(results_group.cur_img, save_resized_img, new_size);
        //         cv::imshow("test", save_resized_img);
        //         if (cv::waitKey(1) == 'q') // 延时1毫秒,按q键退出/Press q to exit
        //             return 0;
        //     }
        // }
        // }

        frames++;

        // if (frames == 10)
        // {
        //     break;
        // }

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
        if (detectPool.get(results_group) != 0)
            break;

        if (mode == 2 && !results_group.cur_img.empty())
        {
            std::ostringstream oss;
            oss << "detect_" << std::setfill('0') << std::setw(4) << results_group.cur_frame_id << ".jpg";
            std::string filename = oss.str();
            cv::imwrite(filename, results_group.cur_img);
        }

        // if (mode == 3 && !results_group.cur_img.empty())
        // {
        //     auto trks = sess->Update(results_group.dets);
        //     char text[256];
        //     for (auto &trk : trks)
        //     {
        //         sprintf(text, "%s", trk.det_name.c_str());
        //         cv::rectangle(results_group.cur_img, trk.box, randColor[trk.id % cnum], 2, 8, 0);
        //         cv::putText(results_group.cur_img, text, cv::Point(trk.box.x, trk.box.y + 12), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255));
        //         std::ostringstream oss;
        //         oss << "track_" << std::setfill('0') << std::setw(4) << results_group.cur_frame_id << ".jpg";
        //         std::string filename = oss.str();
        //         cv::imwrite(filename, results_group.cur_img);
        //     }
        // }

        // if (!results_group.dets.empty())
        // {
        //     if (mode == 2)
        //     {
        //         std::ostringstream oss;
        //         oss << "detect_" << std::setfill('0') << std::setw(4) << results_group.cur_frame_id << ".jpg";
        //         std::string filename = oss.str();
        //         cv::imwrite(filename, results_group.cur_img);
        //     }

        // auto trks = sess->Update(results_group.dets);
        // char text[256];
        // for (auto &trk : trks)
        // {
        //     // std::cout<< trk.det_name << std::endl;
        //     sprintf(text, "%s", trk.det_name.c_str());
        //     // cv::rectangle(results_group.cur_img, trk.box, randColor[trk.id % cnum], 2, 8, 0);
        //     // cv::putText(results_group.cur_img, text, cv::Point(trk.box.x, trk.box.y + 12), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255));

        //     if (mode == 4)
        //     {
        //         // std::string filename = "track_" + std::to_string(results_group.cur_frame_id) + ".jpg";
        //         std::ostringstream oss;
        //         oss << "track_" << std::setfill('0') << std::setw(4) << results_group.cur_frame_id << ".jpg";
        //         std::string filename = oss.str();
        //         cv::imwrite(filename, results_group.cur_img);
        //     }
        //     else if (mode == 1)
        //     {
        //         // 定义缩小后的尺寸 (例如：原始尺寸的一半)
        //         cv::Size new_size(results_group.cur_img.cols / 2, results_group.cur_img.rows / 2);
        //         // 创建用于保存缩小图像的Mat对象
        //         cv::Mat save_resized_img;
        //         // 缩小图像
        //         cv::resize(results_group.cur_img, save_resized_img, new_size);
        //         cv::imshow("test", save_resized_img);
        //         if (cv::waitKey(1) == 'q') // 延时1毫秒,按q键退出/Press q to exit
        //             return 0;
        //     }
        // }
        // }

        frames++;
    }

    // ReleaseSession(&sess);
    capture.release();

    gettimeofday(&time, nullptr);
    auto endTime = time.tv_sec * 1000 + time.tv_usec / 1000;

    printf("Average:\t %f fps/s\n", float(frames) / float(endTime - startTime) * 1000.0);

    return 0;
}
