# RK3588+sort 目标检测与跟踪

## 使用说明
* "data"文件夹存放测试数据
* "model"文件夹存放模型（.rknn）
* 使用说明
```
mkdir build && cd build
cmake ..
make -j8
./rknn_yolov5sort_demo <rknn模型> <测试视频> <mode>
注：mode: 
1 - display
2 - save tracking results as images
3 - save detection results as images 
4 - debug mode
```

## 程序简介
* step-1: 使用线程池来实现yolov5的NPU多线程推理
* step-2: 将推理结果传给sort类实现多目标跟踪
所用模型为[官方](https://github.com/airockchip/rknn_model_zoo)提供onnx自己convert的，若修改为其他模型可能需要修改postprocess部分。

## 下一步更新计划
* 受限sort跟踪原理，其实无法解决长时间丢失跟踪及RE-ID，尽管存在m_max_age参数但是当目标丢失大于两帧时，m_hit_streak会立即置为0，移除相应跟踪器，所以仅仅能在目标短暂丢失（1帧）时候保持稳定跟踪，适合几乎每帧都能检测到同时需要对应前后帧物体的快速跟踪场景中。
* sort算法很快，ms级别所以没有特意评估，目标检测耗时在100fps/s左右。
* 有想过修改sort.CC函数Update中下面这部分的判断条件，使其缓解长时跟踪，但思考片刻及拜读各路大佬真知灼见发现m_hit_streak就应该在丢失到第二帧时立刻置为0，这符合它"连续检测"的定义。
```c++
if (((*it).m_time_since_update < m_max_age) && ((*it).m_hit_streak >= m_min_hits || m_frame_count <= m_min_hits))
{
    TrackingBox trk;
    trk.box = it->GetState();
    trk.id = it->m_id + 1;
    trk.det_name = it->m_det_name;
    trks.push_back(trk);
}

// remove dead tracklet
if ((*it).m_time_since_update > m_max_age)
{
    it = m_trackers.erase(it);
    it--;
}
```

* 基于此，为解决长时遮挡问题，决定转入deep-sort

## Acknowledgements
 在此特别鸣谢各位大佬，拼接侠拼拼凑凑
* https://github.com/abewley/sort.git
* https://github.com/YunYang1994/OpenWork/tree/master/sort
* https://github.com/mcximing/sort-cpp.git
* https://github.com/Zhou-sx/yolov5_Deepsort_rknn
* https://github.com/leafqycc/rknn-cpp-Multithreading