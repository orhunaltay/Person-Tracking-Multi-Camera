import cv2
import time
import threading
import queue
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import supervision as sv
from collections import deque, defaultdict


CAMERA_SOURCES = [0, 1]           
REQUESTED_RES = (1920, 1080)      
REQUESTED_FPS = 30

MODEL_PATH = "yolov8m.pt"         
IMG_SIZE = 960
CONF_THRESH = 0.35
IOU_THRESH  = 0.6
FORCE_MPS   = True                

CLASSES_TO_TRACK = {"person"}     
GLOBAL_ASSOC_TIME = 3.0           
EMB_SIM_THRESHOLD = 0.55          
MAX_GALLERY = 30                 


ROI_POLYGONS = {
    0: [],  
    1: []
}


BOX_SMOOTH_ALPHA = 0.6           

model = YOLO(MODEL_PATH)


try:
    if FORCE_MPS:
        import torch
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            model.to("mps")
        
except Exception:
    pass

names = model.model.names  


trackers = {
    cam: DeepSort(
        max_age=40,
        n_init=2,
        nms_max_overlap=1.0,
        max_cosine_distance=0.25,
        nn_budget=200,
        embedder="mobilenet",
        half=True
    ) for cam in CAMERA_SOURCES
}


frame_queues = {cam: queue.Queue(maxsize=1) for cam in CAMERA_SOURCES}


class GlobalCoordinator:
    """
    Merge per-camera tracks into global identities using appearance embeddings.
    Keeps a time-limited gallery per identity and matches by cosine similarity.
    """
    def __init__(self, sim_thresh=EMB_SIM_THRESHOLD, time_window=GLOBAL_ASSOC_TIME):
        self.sim_thresh = sim_thresh
        self.time_window = time_window
        self.next_global_id = 1
        self.registry = {}  
        self.lock = threading.Lock()

    @staticmethod
    def cos_sim(a, B):
        a = a / (np.linalg.norm(a) + 1e-8)
        B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
        return B @ a

    def _match_existing(self, label, emb, now):
        best_gid, best_sim = None, 0.0
        for gid, info in self.registry.items():
            if info["label"] != label:
                continue
            if now - info["last_seen"] > self.time_window:
                continue
            gal = np.array(info["gallery"]) if len(info["gallery"]) else None
            if gal is None:
                continue
            sims = self.cos_sim(emb, gal)
            sim = float(np.max(sims))
            if sim > best_sim:
                best_sim, best_gid = sim, gid
        if best_gid is not None and best_sim >= self.sim_thresh:
            return best_gid, best_sim
        return None, best_sim

    def assign_global_id(self, label, emb, now, cam):
        with self.lock:
            gid, sim = self._match_existing(label, emb, now)
            if gid is None:
                gid = self.next_global_id
                self.next_global_id += 1
                self.registry[gid] = {
                    "last_seen": now,
                    "last_cam": cam,
                    "label": label,
                    "gallery": deque(maxlen=MAX_GALLERY)
                }
            self.registry[gid]["gallery"].append(emb.astype(np.float32))
            self.registry[gid]["last_seen"] = now
            self.registry[gid]["last_cam"] = cam
            return gid, sim

coordinator = GlobalCoordinator()


box_annotator   = sv.BoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator()

def last_embedding(track):
    """
    Robustly fetch the latest appearance embedding from a DeepSort Track.
    """
    if hasattr(track, "features") and track.features:
        return np.asarray(track.features[-1], dtype=np.float32).ravel()
    getf = getattr(track, "get_feature", None)
    if callable(getf):
        out = getf()
        if out is not None:
            return np.asarray(out, dtype=np.float32).ravel()
    return None

def build_roi_mask(frame_shape, polygon):
    if not polygon:
        return None
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    pts = np.array(polygon, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask

prev_boxes = defaultdict(lambda: None)

def smooth_box(gid, x1, y1, x2, y2, alpha=BOX_SMOOTH_ALPHA):
    prev = prev_boxes[gid]
    if prev is None:
        prev_boxes[gid] = (x1, y1, x2, y2)
        return x1, y1, x2, y2
    px1, py1, px2, py2 = prev
    sx1 = int(alpha * x1 + (1 - alpha) * px1)
    sy1 = int(alpha * y1 + (1 - alpha) * py1)
    sx2 = int(alpha * x2 + (1 - alpha) * px2)
    sy2 = int(alpha * y2 + (1 - alpha) * py2)
    prev_boxes[gid] = (sx1, sy1, sx2, sy2)
    return sx1, sy1, sx2, sy2


stop_flag = False

def open_camera(cam_index):
    cap = cv2.VideoCapture(cam_index, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        print(f"[!] Could not open camera {cam_index}")
        return None
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  REQUESTED_RES[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, REQUESTED_RES[1])
    cap.set(cv2.CAP_PROP_FPS, REQUESTED_FPS)
    return cap

def apply_roi(frame, roi_poly):
    if not roi_poly:
        return frame
    mask = build_roi_mask(frame.shape, roi_poly)
    return cv2.bitwise_and(frame, frame, mask=mask)

def producer(cam_index):
    global stop_flag
    cap = open_camera(cam_index)
    if cap is None:
        return

    roi_poly = ROI_POLYGONS.get(cam_index, [])
    while not stop_flag:
        ok, frame = cap.read()
        if not ok or frame is None:
            print(f("[!] Camera {cam_index} read failed."))
            break

        
        frame_proc = apply_roi(frame, roi_poly) if roi_poly else frame

        
        r = model.predict(
            frame_proc,
            imgsz=IMG_SIZE,
            conf=CONF_THRESH,
            iou=IOU_THRESH,
            classes=[0],      
            verbose=False
        )[0]

        dets = []
        if r.boxes is not None and r.boxes.xyxy is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            conf = r.boxes.conf.cpu().numpy()
            cls  = r.boxes.cls.cpu().numpy().astype(int)
            for i in range(len(xyxy)):
                cls_name = names.get(int(cls[i]), str(int(cls[i])))
                if cls_name not in CLASSES_TO_TRACK:
                    continue
                x1, y1, x2, y2 = xyxy[i]
                dets.append(([float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                             float(conf[i]), cls_name))

        tracks = trackers[cam_index].update_tracks(dets, frame=frame_proc)

        draw_boxes, draw_labels = [], []
        now = time.time()

        for trk in tracks:
            if not trk.is_confirmed():
                continue

            l, t, r_, b = trk.to_ltrb()
            x1, y1, x2, y2 = map(int, (l, t, r_, b))
            cls_name = trk.get_det_class() or "person"
            det_conf = trk.get_det_conf() or 0.0

            emb = last_embedding(trk)
            if emb is None:
                continue

            gid, sim = coordinator.assign_global_id(cls_name, emb, now, cam_index)

            
            x1, y1, x2, y2 = smooth_box(gid, x1, y1, x2, y2, BOX_SMOOTH_ALPHA)

            draw_boxes.append([x1, y1, x2, y2])
            draw_labels.append(f"{cls_name} G#{gid} s:{sim:.2f} c:{det_conf:.2f}")

        
        display_frame = frame.copy()
        if draw_boxes:
            det_for_draw = sv.Detections(xyxy=np.array(draw_boxes), class_id=np.zeros(len(draw_boxes), dtype=int))
            display_frame = box_annotator.annotate(scene=display_frame, detections=det_for_draw)
            display_frame = label_annotator.annotate(scene=display_frame, detections=det_for_draw, labels=draw_labels)

        
        q = frame_queues[cam_index]
        if not q.empty():
            try:
                q.get_nowait()
            except queue.Empty:
                pass
        q.put(display_frame)

    cap.release()


threads = []
for cam in CAMERA_SOURCES:
    t = threading.Thread(target=producer, args=(cam,), daemon=True)
    t.start()
    threads.append(t)


try:
    titles = [f"Camera {c} • Collective Tracker (pro)" for c in CAMERA_SOURCES]
    for title in titles:
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)

    while not stop_flag:
        for cam, title in zip(CAMERA_SOURCES, titles):
            q = frame_queues[cam]
            if not q.empty():
                frame = q.get()
                if frame is not None and frame.size:
                    cv2.imshow(title, frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            stop_flag = True
            break
        time.sleep(0.005)

finally:
    stop_flag = True
    for t in threads:
        t.join(timeout=1.0)
    cv2.destroyAllWindows()
