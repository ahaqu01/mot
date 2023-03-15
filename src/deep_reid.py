import numpy as np
import collections
import yaml
import cv2
import torch
from .models.feature_extracter import FeatureExtraction
from .models.sort.nn_matching import NearestNeighborDistanceMetric
from .models.sort.preprocessing import non_max_suppression
from .models.sort.detection import Detection
from .models.sort.tracker import Tracker


class DeepReid(object):
    def __init__(self,
                 extractor_config="",
                 extractor_weights="",
                 tracker_config="",
                 device=None):

        with open(extractor_config, 'r', encoding='utf-8') as ex_f:
            cont = ex_f.read()
            ex_cfg = yaml.load(cont)
            self.ex_cfg = ex_cfg
        with open(tracker_config, 'r', encoding='utf-8') as tk_f:
            cont = tk_f.read()
            tk_cfg = yaml.load(cont)
            self.tk_cfg = tk_cfg

        # create feature extractor and load weights
        print("creating person feature extactor")
        self.extractor = FeatureExtraction(ex_cfg, device=device)
        print("loading person feature extactor weights")
        extractor_weights_state_dict = \
        torch.load(extractor_weights, map_location=lambda storage, loc: storage.cuda(device))["model"]
        incompatible = self.extractor.load_state_dict(extractor_weights_state_dict, strict=False)
        if incompatible.missing_keys:
            print("missing_keys:", incompatible.missing_keys)
        if incompatible.unexpected_keys:
            print("unexpected_keys:", incompatible.unexpected_keys)
        print("person feature extactor weights loaded")

        # create tracker
        print("creating person tracker")
        self.min_confidence = self.tk_cfg["DEEPSORT"]["MIN_CONFIDENCE"]
        self.nms_max_overlap = self.tk_cfg["DEEPSORT"]["NMS_MAX_OVERLAP"]
        max_cosine_distance = self.tk_cfg["DEEPSORT"]["MAX_DIST"]
        nn_budget = self.tk_cfg["DEEPSORT"]["NN_BUDGET"]
        max_iou_distance = self.tk_cfg["DEEPSORT"]["MAX_IOU_DISTANCE"]
        max_age = self.tk_cfg["DEEPSORT"]["MAX_AGE"]
        n_init = self.tk_cfg["DEEPSORT"]["N_INIT"]
        metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(metric, max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init)
        print("person tracker created")

    @torch.no_grad()
    def update(self, bbox_xyxy, confidences, ori_img):
        # bbox_xyxy: ndarray, (N, 4), left,top,right,bottom
        # confidences: list, len(confidences)=N
        # ori_img; ndarray; BGR, (H, W, 3)
        outputs = {}
        added_track_ids = []
        if not len(confidences):
            return outputs, added_track_ids
        else:
            self.height, self.width = ori_img.shape[:2]

            features = self._get_features(bbox_xyxy, ori_img)

            bbox_tlwh = self._xyxy_to_tlwh(bbox_xyxy)
            detections = [Detection(bbox_tlwh[i], conf, features[i], det_id=i) for i, conf in enumerate(confidences) if
                          conf > self.min_confidence]

            # run on non-maximum supression
            boxes = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            indices = non_max_suppression(boxes, self.nms_max_overlap, scores)
            detections = [detections[i] for i in indices]

            # update tracker
            self.tracker.predict()
            # update tracker
            added_track_ids, \
            deleted_track_ids, \
            remain_track_ids, \
            matched_detection_track_map, \
            unmatched_detection_track_map = self.tracker.update(detections)
            print("add track id", added_track_ids)
            print("delete track id", deleted_track_ids)
            print("remain track id", remain_track_ids)
            print("matched_detection", matched_detection_track_map)
            print("unmatched_detection", unmatched_detection_track_map)

            # output
            # {"track_id":
            #     {
            #         "detection_id": "int",
            #         "bbox": "list", # yxyx
            #         "confidence": "float", # score
            #         "feature": "ndarray", # dim=512
            #         # Tentative = 1
            #         # Confirmed = 2
            #         # Deleted = 3
            #     }
            # }
            for det_i in matched_detection_track_map:
                track_id = matched_detection_track_map[det_i]
                bbox_xyxy_i = detections[det_i].to_tlbr()
                confidence_i = detections[det_i].confidence
                feature_i = detections[det_i].feature
                outputs[track_id] = {"detection_id": det_i,
                                     "bbox": bbox_xyxy_i,
                                     "confidence": confidence_i,
                                     "feature": feature_i,
                                     "det_id_raw": detections[det_i].det_id}
            for det_i in unmatched_detection_track_map:
                track_id = unmatched_detection_track_map[det_i]
                bbox_xyxy_i = detections[det_i].to_tlbr()
                confidence_i = detections[det_i].confidence
                feature_i = detections[det_i].feature
                outputs[track_id] = {"detection_id": det_i,
                                     "bbox": bbox_xyxy_i,
                                     "confidence": confidence_i,
                                     "feature": feature_i,
                                     "det_id_raw": detections[det_i].det_id}
            return outputs, added_track_ids

    """
    TODO:
        Convert bbox from xc_yc_w_h to xtl_ytl_w_h
    Thanks JieChen91@github.com for reporting this bug!
    """

    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()
        bbox_tlwh[:, 0] = bbox_xywh[:, 0] - bbox_xywh[:, 2] / 2.
        bbox_tlwh[:, 1] = bbox_xywh[:, 1] - bbox_xywh[:, 3] / 2.
        return bbox_tlwh

    def _xywh_to_xyxy(self, bbox_xywh):
        x, y, w, h = bbox_xywh
        x1 = max(int(x - w / 2), 0)
        x2 = min(int(x + w / 2), self.width - 1)
        y1 = max(int(y - h / 2), 0)
        y2 = min(int(y + h / 2), self.height - 1)
        return x1, y1, x2, y2

    def _tlwh_to_xyxy(self, bbox_tlwh):
        """
        TODO:
            Convert bbox from xtl_ytl_w_h to xc_yc_w_h
        Thanks JieChen91@github.com for reporting this bug!
        """
        x, y, w, h = bbox_tlwh
        x1 = max(int(x), 0)
        x2 = min(int(x + w), self.width - 1)
        y1 = max(int(y), 0)
        y2 = min(int(y + h), self.height - 1)
        return x1, y1, x2, y2

    def _xyxy_to_tlwh(self, bbox_xyxy):
        if isinstance(bbox_xyxy, np.ndarray):
            bbox_tlwh = bbox_xyxy.copy()
        elif isinstance(bbox_xyxy, torch.Tensor):
            bbox_tlwh = bbox_xyxy.clone()
        bbox_tlwh[:, 2] = bbox_xyxy[:, 2] - bbox_xyxy[:, 0]
        bbox_tlwh[:, 3] = bbox_xyxy[:, 3] - bbox_xyxy[:, 1]
        return bbox_tlwh

    def _get_features(self, bbox_xyxy, ori_img):
        im_crops = []
        for box in bbox_xyxy:
            x1, y1, x2, y2 = box
            im = ori_img[int(y1):int(y2), int(x1):int(x2)]
            im = im[:, :, ::-1]  # reid 前处理
            im = cv2.resize(im, (128, 256), interpolation=cv2.INTER_CUBIC)
            im_crops.append(torch.as_tensor(im.astype("float32").transpose(2, 0, 1))[None])
        if im_crops:
            batch_image = torch.cat(im_crops, dim=0)
            features = self.extractor(batch_image)
        else:
            features = np.array([])
        return features
