#include <stdio.h>
#include <memory>
#include <sys/time.h>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "rkYolov5s.hpp"
#include "rknnPool.hpp"

int main(int argc, char **argv)
{
    if (argc < 4)
    {
        printf("Usage: %s <model path> <video_source> <save mode>\n", argv[0]);
        printf("mode: 1 - display, 2 - save as images, 3 - save as video\n");
        return -1;
    }
    char *model_name = NULL;
    model_name = (char *)argv[1];
    const char *vedio_name = argv[2];
    int mode = std::atoi(argv[3]);

    // 初始化rknn线程池 /Initialize the rknn thread pool
    int threadNum = 3;
    rknnPool<rkYolov5s, cv::Mat, cv::Mat> testPool(model_name, threadNum);
    if (testPool.init() != 0)
    {
        printf("rknnPool init fail!\n");
        return -1;
    }

    cv::VideoCapture capture;
    capture.open(vedio_name);

    if (!capture.isOpened())
    {
        printf("Error: Could not open video or camera\n");
        return -1;
    }

    cv::VideoWriter save_video;
    if (mode == 3)
    {
        double fps = capture.get(cv::CAP_PROP_FPS);
        int frame_width = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH));
        int frame_height = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT));
        save_video.open("output.mp4", cv::VideoWriter::fourcc('H', '2', '6', '4'), fps, cv::Size(frame_width, frame_height));
    }

    struct timeval time;
    gettimeofday(&time, nullptr);
    auto startTime = time.tv_sec * 1000 + time.tv_usec / 1000;

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
            
        if (testPool.put(img) != 0)
        {
            printf("put original images failed!\n");
            break;
        }
        if (frames >= threadNum && testPool.get(img) != 0)
        {
            printf("frames > 3 but get processed images failed!\n");
            break;
        }
            
        if (mode == 1)
        {
            cv::imshow("Camera FPS", img);
            if (cv::waitKey(1) == 'q') // 延时1毫秒,按q键退出/Press q to exit
                break;
        }
        else if (mode == 2)
        {
            std::string filename = "frame_" + std::to_string(frames) + ".jpg";
            cv::imwrite(filename, img);
        }
        else if (mode == 3)
        {
            save_video.write(img);
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
        cv::Mat img;
        if (testPool.get(img) != 0)
            break;
        if (mode == 1)
        {
            cv::imshow("Camera FPS", img);
            if (cv::waitKey(1) == 'q') // 延时1毫秒,按q键退出/Press q to exit
                break;
        }
        else if (mode == 2)
        {
            std::string filename = "frame_" + std::to_string(frames) + ".jpg";
            cv::imwrite(filename, img);
        }
        else if (mode == 3)
        {
            save_video.write(img);
        }
        frames++;
    }

    capture.release();
    if (mode == 3)
    {
        save_video.release();
    }

    gettimeofday(&time, nullptr);
    auto endTime = time.tv_sec * 1000 + time.tv_usec / 1000;

    printf("Average:\t %f fps/s\n", float(frames) / float(endTime - startTime) * 1000.0);


    return 0;
}
