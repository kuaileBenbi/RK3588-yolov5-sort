#include "postprocess.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <iostream>

#include <set>
#include <vector>

static const char *labels[OBJ_CLASS_NUM] = {"person", "bicycle", "car", "motorcycle",
                                            "airplane", "bus", "train", "truck", "boat",
                                            "traffic light", "fire hydrant", "stop sign",
                                            "parking meter", "bench", "bird", "cat", "dog",
                                            "horse", "sheep", "cow", "elephant", "bear",
                                            "zebra", "giraffe", "backpack", "umbrella",
                                            "handbag", "tie", "suitcase", "frisbee",
                                            "skis", "snowboard", "sports ball",
                                            "kite", "baseball bat", "baseball glove",
                                            "skateboard", "surfboard", "tennis racket",
                                            "bottle", "wine glass", "cup", "fork", "knife",
                                            "spoon", "bowl", "banana", "apple", "sandwich",
                                            "orange", "broccoli", "carrot", "hot dog", "pizza",
                                            "donut", "cake", "chair", "couch", "potted plant",
                                            "bed", "dining table", "toilet", "tv", "laptop",
                                            "mouse", "remote", "keyboard", "cell phone",
                                            "microwave", "oven", "toaster", "sink", "refrigerator",
                                            "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"};

const int anchor0[6] = {10, 13, 16, 30, 33, 23};
const int anchor1[6] = {30, 61, 62, 45, 59, 119};
const int anchor2[6] = {116, 90, 156, 198, 373, 326};

// 声明静态变量
static const int cnum = 20;
static cv::Scalar_<int> randColor[cnum];
static bool init_track = false;

inline static int clamp(float val, int min, int max) { return val > min ? (val < max ? val : max) : min; }

static float CalculateOverlap(float xmin0, float ymin0, float xmax0, float ymax0, float xmin1, float ymin1, float xmax1,
                              float ymax1)
{
  float w = fmax(0.f, fmin(xmax0, xmax1) - fmax(xmin0, xmin1) + 1.0);
  float h = fmax(0.f, fmin(ymax0, ymax1) - fmax(ymin0, ymin1) + 1.0);
  float i = w * h;
  float u = (xmax0 - xmin0 + 1.0) * (ymax0 - ymin0 + 1.0) + (xmax1 - xmin1 + 1.0) * (ymax1 - ymin1 + 1.0) - i;
  return u <= 0.f ? 0.f : (i / u);
}

static int nms(int validCount, std::vector<float> &outputLocations, std::vector<int> classIds, std::vector<int> &order,
               int filterId, float threshold)
{
  for (int i = 0; i < validCount; ++i)
  {
    if (order[i] == -1 || classIds[i] != filterId)
    {
      continue;
    }
    int n = order[i];
    for (int j = i + 1; j < validCount; ++j)
    {
      int m = order[j];
      if (m == -1 || classIds[i] != filterId)
      {
        continue;
      }
      float xmin0 = outputLocations[n * 4 + 0];
      float ymin0 = outputLocations[n * 4 + 1];
      float xmax0 = outputLocations[n * 4 + 0] + outputLocations[n * 4 + 2];
      float ymax0 = outputLocations[n * 4 + 1] + outputLocations[n * 4 + 3];

      float xmin1 = outputLocations[m * 4 + 0];
      float ymin1 = outputLocations[m * 4 + 1];
      float xmax1 = outputLocations[m * 4 + 0] + outputLocations[m * 4 + 2];
      float ymax1 = outputLocations[m * 4 + 1] + outputLocations[m * 4 + 3];

      float iou = CalculateOverlap(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1);

      if (iou > threshold)
      {
        order[j] = -1;
      }
    }
  }
  return 0;
}

static int quick_sort_indice_inverse(std::vector<float> &input, int left, int right, std::vector<int> &indices)
{
  float key;
  int key_index;
  int low = left;
  int high = right;
  if (left < right)
  {
    key_index = indices[left];
    key = input[left];
    while (low < high)
    {
      while (low < high && input[high] <= key)
      {
        high--;
      }
      input[low] = input[high];
      indices[low] = indices[high];
      while (low < high && input[low] >= key)
      {
        low++;
      }
      input[high] = input[low];
      indices[high] = indices[low];
    }
    input[low] = key;
    indices[low] = key_index;
    quick_sort_indice_inverse(input, left, low - 1, indices);
    quick_sort_indice_inverse(input, low + 1, right, indices);
  }
  return low;
}

inline static int32_t __clip(float val, float min, float max)
{
  float f = val <= min ? min : (val >= max ? max : val);
  return f;
}

static int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale)
{
  float dst_val = (f32 / scale) + zp;
  int8_t res = (int8_t)__clip(dst_val, -128, 127);
  return res;
}

static float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) { return ((float)qnt - (float)zp) * scale; }

static int process(int8_t *input, int *anchor, int grid_h, int grid_w, int height, int width, int stride,
                   std::vector<float> &boxes, std::vector<float> &objProbs, std::vector<int> &classId, float threshold,
                   int32_t zp, float scale)
{
  int validCount = 0;
  int grid_len = grid_h * grid_w;
  int8_t thres_i8 = qnt_f32_to_affine(threshold, zp, scale);
  for (int a = 0; a < 3; a++)
  {
    for (int i = 0; i < grid_h; i++)
    {
      for (int j = 0; j < grid_w; j++)
      {
        int8_t box_confidence = input[(PROP_BOX_SIZE * a + 4) * grid_len + i * grid_w + j];
        if (box_confidence >= thres_i8)
        {
          int offset = (PROP_BOX_SIZE * a) * grid_len + i * grid_w + j;
          int8_t *in_ptr = input + offset;
          float box_x = (deqnt_affine_to_f32(*in_ptr, zp, scale)) * 2.0 - 0.5;
          float box_y = (deqnt_affine_to_f32(in_ptr[grid_len], zp, scale)) * 2.0 - 0.5;
          float box_w = (deqnt_affine_to_f32(in_ptr[2 * grid_len], zp, scale)) * 2.0;
          float box_h = (deqnt_affine_to_f32(in_ptr[3 * grid_len], zp, scale)) * 2.0;
          box_x = (box_x + j) * (float)stride;
          box_y = (box_y + i) * (float)stride;
          box_w = box_w * box_w * (float)anchor[a * 2];
          box_h = box_h * box_h * (float)anchor[a * 2 + 1];
          box_x -= (box_w / 2.0);
          box_y -= (box_h / 2.0);

          int8_t maxClassProbs = in_ptr[5 * grid_len];
          int maxClassId = 0;
          for (int k = 1; k < OBJ_CLASS_NUM; ++k)
          {
            int8_t prob = in_ptr[(5 + k) * grid_len];
            if (prob > maxClassProbs)
            {
              maxClassId = k;
              maxClassProbs = prob;
            }
          }
          if (maxClassProbs > thres_i8)
          {
            objProbs.push_back((deqnt_affine_to_f32(maxClassProbs, zp, scale)) * (deqnt_affine_to_f32(box_confidence, zp, scale)));
            classId.push_back(maxClassId);
            validCount++;
            boxes.push_back(box_x);
            boxes.push_back(box_y);
            boxes.push_back(box_w);
            boxes.push_back(box_h);
          }
        }
      }
    }
  }
  return validCount;
}

int post_process(int8_t *input0, int8_t *input1, int8_t *input2, int model_in_h, int model_in_w, float conf_threshold,
                 float nms_threshold, BOX_RECT pads, float scale_w, float scale_h, std::vector<int32_t> &qnt_zps,
                 std::vector<float> &qnt_scales, DetectResultsGroup *group)
{
  memset(group, 0, sizeof(DetectResultsGroup));

  std::vector<float> filterBoxes;
  std::vector<float> objProbs;
  std::vector<int> classId;

  // stride 8
  int stride0 = 8;
  int grid_h0 = model_in_h / stride0;
  int grid_w0 = model_in_w / stride0;
  int validCount0 = 0;
  validCount0 = process(input0, (int *)anchor0, grid_h0, grid_w0, model_in_h, model_in_w, stride0, filterBoxes, objProbs,
                        classId, conf_threshold, qnt_zps[0], qnt_scales[0]);

  // stride 16
  int stride1 = 16;
  int grid_h1 = model_in_h / stride1;
  int grid_w1 = model_in_w / stride1;
  int validCount1 = 0;
  validCount1 = process(input1, (int *)anchor1, grid_h1, grid_w1, model_in_h, model_in_w, stride1, filterBoxes, objProbs,
                        classId, conf_threshold, qnt_zps[1], qnt_scales[1]);

  // stride 32
  int stride2 = 32;
  int grid_h2 = model_in_h / stride2;
  int grid_w2 = model_in_w / stride2;
  int validCount2 = 0;
  validCount2 = process(input2, (int *)anchor2, grid_h2, grid_w2, model_in_h, model_in_w, stride2, filterBoxes, objProbs,
                        classId, conf_threshold, qnt_zps[2], qnt_scales[2]);

  int validCount = validCount0 + validCount1 + validCount2;
  // no object detect
  if (validCount <= 0)
  {
    return 0;
  }

  std::vector<int> indexArray;
  for (int i = 0; i < validCount; ++i)
  {
    indexArray.push_back(i);
  }

  quick_sort_indice_inverse(objProbs, 0, validCount - 1, indexArray);

  std::set<int> class_set(std::begin(classId), std::end(classId));

  for (auto c : class_set)
  {
    nms(validCount, filterBoxes, classId, indexArray, c, nms_threshold);
  }

  /* box valid detect target */
  for (int i = 0; i < validCount; ++i)
  {
    // if (indexArray[i] == -1 || last_count >= OBJ_NUMB_MAX_SIZE)
    if (indexArray[i] == -1)
    {
      continue;
    }
    int n = indexArray[i];

    float x1 = filterBoxes[n * 4 + 0] - pads.left;
    float y1 = filterBoxes[n * 4 + 1] - pads.top;
    float x2 = x1 + filterBoxes[n * 4 + 2];
    float y2 = y1 + filterBoxes[n * 4 + 3];
    int id = classId[n];
    float obj_conf = objProbs[i];

    DetectionBox new_box;
    int _x1 = (int)(clamp(x1, 0, model_in_w) / scale_w);
    int _y1 = (int)(clamp(y1, 0, model_in_h) / scale_h);
    int _x2 = (int)(clamp(x2, 0, model_in_w) / scale_w);
    int _y2 = (int)(clamp(y2, 0, model_in_h) / scale_h);
    new_box.box = cv::Rect_<int>(_x1, _y1, _x2 - _x1, _y2 - _y1);
    new_box.score = objProbs[i];
    new_box.det_name = labels[id];
    group->dets.push_back(new_box);
  }

  return 0;
}

int draw_image_detect(cv::Mat &cur_img, std::vector<DetectionBox> &results, int cur_frame_id)
{
  char text[256];
  for (const auto& res : results)
  {
    sprintf(text, "%s", res.det_name.c_str());
    cv::rectangle(cur_img, res.box, cv::Scalar(256, 0, 0, 256), 3);
    cv::putText(cur_img, text, cv::Point(res.box.x, res.box.y + 12), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255));
  }
  std::ostringstream oss;

  oss << "detect_" << std::setfill('0') << std::setw(4) << cur_frame_id << ".jpg";

  std::string filename = oss.str();

  if (!cv::imwrite(filename, cur_img))
  {
    return -1;
  }
  return 0;
}

// 初始化随机颜色
static void initializeRandColors()
{
  cv::RNG rng(0xFFFFFFFF); // 生成随机颜色
  for (int i = 0; i < cnum; i++)
  {
    rng.fill(randColor[i], cv::RNG::UNIFORM, 0, 256);
  }
  init_track = true;
}

int draw_image_track(cv::Mat &cur_img, std::vector<TrackingBox> &results, int cur_frame_id)
{
  if (!init_track)
  {
    initializeRandColors();
  }
  char text[256];
  for (const auto& res : results)
  {
    sprintf(text, "%s", res.det_name.c_str());
    cv::rectangle(cur_img, res.box, randColor[res.id % cnum], 2, 8, 0);
    cv::putText(cur_img, text, cv::Point(res.box.x, res.box.y + 12), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255));
  }
  std::ostringstream oss;

  oss << "track_" << std::setfill('0') << std::setw(4) << cur_frame_id << ".jpg";

  std::string filename = oss.str();
  if (!cv::imwrite(filename, cur_img))
  {
    return -1;
  }
  return 0;
}


void show_draw_results(cv::Mat &cur_img, std::vector<TrackingBox> &results)
{
  if (!init_track)
  {
    initializeRandColors();
  }
  char text[256];
  for (const auto& res : results)
  {
    sprintf(text, "%s", res.det_name.c_str());
    cv::rectangle(cur_img, res.box, randColor[res.id % cnum], 2, 8, 0);
    cv::putText(cur_img, text, cv::Point(res.box.x, res.box.y + 12), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255));
  }
}