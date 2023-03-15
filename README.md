

## mot API

## 一、接口说明

本仓库主要目的是对视频中的行人进行跟踪+行人特征提取

行人跟踪使用了deepsort算法，行人特征提取使用了经过蒸馏的resnet34

本仓库代码由官方代码重构而得，

官方代码见：https://github.com/zengwb-lx/Yolov5-Deepsort-Fastreid.git

## 二、环境说明

相关依赖库见：requirements.txt

## 三、API参数说明

##### **mot.src.deep_reid.DeepReid**

行人多目标跟踪类

类构建参数：

extractor_config: str, 行人特征提取器的参数路径，这里可使用44服务器，“./mot/src/configs/config-test.yaml”参数<br>
extractor_weights: str, 行人特征提取器的权重路径，这里可使用44服务器，”./mot/src/weights/model_final.pth”权重<br>
tracker_config: str, 跟踪器参数路径，这里可使用44服务器，“./mot/src/configs/deep_sort.yaml”参数<br>
device: torch.device object, 推理时的device<br>

###### ****mot.src.deep_reid.DeepReid**.update**

行人跟踪方法

1. 输入参数：<br>

   三个输入参数

   bbox_xyxy: ndarray, (N, 4), bboxes, left,top,right,bottom

   confidences: list, len(confidences)=N, bbox对应置信度

   ori_img: ndarray, shape: (H, W, 3), 通道顺序：BGR<br>

2. 输出结果：<br>

   两个输出

   outputs: dict, 

   ​	key: track_id, 跟踪的人物id, int

   ​	value: dict, 具体包含以下字段，

   ​		detection_id，输入图像中的bbox id

   ​		bbox，输入图像的bbox

   ​		confidence，对应bbox的置信度

   ​		feature，对应bbox中行人512维度特征

   ​		det_id_raw，对应yolov5输出的bbox_id

   ```
   outputs[track_id] = {"detection_id": det_i,
                         "bbox": bbox_xyxy_i,
                         "confidence": confidence_i,
                         "feature": feature_i,
                         "det_id_raw": detections[det_i].det_id}
   ```

   added_track_ids: list, 本张图像中新增的人物id

### 四、API使用样例

**you can use it as submodule**

在自己的项目目录下，git submodule add  https://gitlab.ictidei.cn/band-intel-center/Algorithm-platform/mot.git

便会在项目目录下下载到mot相关代码

下载完成后，便可在自己项目中使用mot API，**使用样例**如下：

```python
import torch
from mot.src.deep_reid import DeepReid
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
reid_track = DeepReid(extractor_config="../mot/src/configs/config-test.yaml",
                      extractor_weights="./mot/src/weights/model_final.pth",
                      tracker_config="./mot/src/configs/deep_sort.yaml",
                      device=device)
# track
# API inputs    								  
    # bbox_xyxy: results of bbox from human detector, xyxy, numpy.ndarray, (N, 4)
    # confidences: results of confidences from human detector, list, len为N
    # ori_img: bgr image corresponding to detection results
outputs, added_track_ids = reid_track.update(
    bbox_xyxy=detection_results_bboxs, 
    confidences=detection_results_confidences, 
    ori_img=frame_bgr)
```