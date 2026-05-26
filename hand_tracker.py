import cv2, mediapipe as mp, urllib.request, os

MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = "hand_landmarker.task"

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

def _download_model():
    if os.path.exists(MODEL_PATH):
        return
    print("📥  Downloading hand model (~5MB, one-time)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("✅  Model ready!")

class HandTracker:
    def __init__(self):
        _download_model()
        Opt  = mp.tasks.vision.HandLandmarkerOptions
        Base = mp.tasks.BaseOptions
        Mode = mp.tasks.vision.RunningMode
        opts = Opt(
            base_options=Base(model_asset_path=MODEL_PATH),
            running_mode=Mode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self.detector = mp.tasks.vision.HandLandmarker.create_from_options(opts)
        self.ts       = 0

    def process(self, frame):
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self.ts += 1
        result = self.detector.detect_for_video(mp_img, self.ts)
        lms    = result.hand_landmarks
        if lms:
            self._draw(frame, lms[0])
        return frame, lms[0] if lms else None

    def _draw(self, frame, lm):
        H, W = frame.shape[:2]
        pts  = [(int(l.x*W), int(l.y*H)) for l in lm]
        for a,b in CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (140,140,200), 1, cv2.LINE_AA)
        for i,(x,y) in enumerate(pts):
            c = (0,220,255) if i in (4,8,12,16,20) else (80,180,255)
            cv2.circle(frame,(x,y),5,c,-1,cv2.LINE_AA)
            cv2.circle(frame,(x,y),5,(255,255,255),1,cv2.LINE_AA)

    def fingers_up(self, lm):
        """Returns [thumb,index,middle,ring,pinky]  1=raised"""
        f = []
        f.append(1 if lm[4].x < lm[2].x else 0)          # thumb
        for tip,pip in [(8,6),(12,10),(16,14),(20,18)]:
            f.append(1 if lm[tip].y < lm[pip].y else 0)
        return f

    def tip_px(self, lm, idx, W, H):
        return int(lm[idx].x*W), int(lm[idx].y*H)