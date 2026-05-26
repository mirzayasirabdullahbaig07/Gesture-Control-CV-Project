import cv2
import numpy as np
import mediapipe as mp
import urllib.request
import os, time, random, math
from collections import deque

# ================================================================
#  GESTURE DRAWING BOARD  —  gesture_board.py  v4
#  Run:  python gesture_board.py
#
#  GESTURES:
#    ☝  INDEX only              →  DRAW
#    ✌  INDEX + MIDDLE          →  MOVE / select color (pen up)
#    🖐  ALL fingers open HIGH  →  CLEAR (shake to confirm)
#    ✊  Hold FIST              →  ERASE
#
#  KEY FEATURE — PEN LIFT:
#    Switch to ✌ MOVE between letters.
#    Strokes ONLY draw while index is up and middle is DOWN.
#    The moment you raise your middle finger = pen lifts off.
#
#  KEYBOARD:
#    S=Save  C=Clear  B=Brush  [ ]=Size  Q=Quit
# ================================================================

MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = "hand_landmarker.task"

COLORS = [
    ("Purple", (220,  40, 200)),
    ("Blue",   (255,  60,   0)),
    ("Green",  (  0, 210,   0)),
    ("Yellow", (  0, 215, 255)),
    ("Red",    (  0,   0, 255)),
    ("Orange", (  0, 140, 255)),
    ("Cyan",   (255, 200,   0)),
    ("White",  (255, 255, 255)),
    ("Black",  ( 20,  20,  20)),
]
BRUSHES = ["pen", "neon", "spray", "glow", "marker"]

# ════════════════════════════════════════════════════════════════
#  MODEL DOWNLOAD
# ════════════════════════════════════════════════════════════════
def download_model():
    if os.path.exists(MODEL_PATH):
        return
    print("📥  Downloading hand model (~5MB, one-time)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("✅  Done!")

# ════════════════════════════════════════════════════════════════
#  HAND DETECTOR
# ════════════════════════════════════════════════════════════════
SKEL = [(0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (0,9),(9,10),(10,11),(11,12),
        (0,13),(13,14),(14,15),(15,16),
        (0,17),(17,18),(18,19),(19,20),
        (5,9),(9,13),(13,17)]

def make_detector():
    download_model()
    opts = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.55,
        min_hand_presence_confidence=0.55,
        min_tracking_confidence=0.55,
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(opts)

def get_landmarks(detector, frame, ts):
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    res    = detector.detect_for_video(mp_img, ts)
    return res.hand_landmarks[0] if res.hand_landmarks else None

def draw_hand(frame, lm, W, H):
    pts = [(int(l.x*W), int(l.y*H)) for l in lm]
    for a,b in SKEL:
        cv2.line(frame, pts[a], pts[b], (140,140,200), 1, cv2.LINE_AA)
    for i,(x,y) in enumerate(pts):
        col = (0,200,255) if i in (4,8,12,16,20) else (70,140,255)
        cv2.circle(frame,(x,y),4,col,-1,cv2.LINE_AA)
        cv2.circle(frame,(x,y),4,(255,255,255),1,cv2.LINE_AA)

# ════════════════════════════════════════════════════════════════
#  FINGER STATE
# ════════════════════════════════════════════════════════════════
def get_finger_state(lm, W, H):
    """
    Returns dict:
      index_up   — index finger extended
      middle_up  — middle finger extended
      fist       — all 4 fingers curled
      all_open   — all 4 fingers extended
      wrist_y    — wrist pixel Y (to detect hand raised high)
    """
    palm_x = sum(lm[i].x for i in [0,5,9,13,17]) / 5
    palm_y = sum(lm[i].y for i in [0,5,9,13,17]) / 5

    def tdist(i):
        return math.hypot(lm[i].x - palm_x, lm[i].y - palm_y)

    R   = 1.3
    idx = tdist(8)  > tdist(5)*R  and lm[8].y  < lm[6].y
    mid = tdist(12) > tdist(9)*R  and lm[12].y < lm[10].y
    rng = tdist(16) > tdist(13)*R and lm[16].y < lm[14].y
    pnk = tdist(20) > tdist(17)*R and lm[20].y < lm[18].y

    fist     = not idx and not mid and not rng and not pnk
    all_open = idx and mid and rng and pnk

    wrist_y = int(lm[0].y * H)   # wrist pixel Y position

    return {
        "index_up":  idx,
        "middle_up": mid,
        "fist":      fist,
        "all_open":  all_open,
        "wrist_y":   wrist_y,
    }

# ════════════════════════════════════════════════════════════════
#  SHAKE DETECTOR  — detects rapid left-right wrist movement
# ════════════════════════════════════════════════════════════════
class ShakeDetector:
    """
    Detects a shake gesture: rapid back-and-forth X movement.
    Used to confirm CLEAR when hand is open and raised high.
    """
    def __init__(self):
        self.x_history  = deque(maxlen=20)
        self.dir_changes = 0
        self.last_dir    = 0
        self.cooldown    = 0

    def update(self, x, all_open, wrist_y, H):
        """Returns True if shake + open hand + raised high detected."""
        if self.cooldown > 0:
            self.cooldown -= 1

        # Only track when hand is open AND raised above 40% of frame
        raised = wrist_y < H * 0.45
        if not all_open or not raised:
            self.x_history.clear()
            self.dir_changes = 0
            self.last_dir    = 0
            return False

        self.x_history.append(x)
        if len(self.x_history) < 4:
            return False

        # Count direction changes in recent history
        changes = 0
        for i in range(2, len(self.x_history)):
            dx_prev = self.x_history[i-1] - self.x_history[i-2]
            dx_curr = self.x_history[i]   - self.x_history[i-1]
            # Direction changed and movement is significant (>15px)
            if abs(dx_curr) > 15 and abs(dx_prev) > 15:
                if (dx_curr > 0) != (dx_prev > 0):
                    changes += 1

        # 3+ direction changes in 20 frames = shake detected
        if changes >= 3 and self.cooldown == 0:
            self.x_history.clear()
            self.dir_changes = 0
            self.cooldown    = 45   # prevent double-trigger
            return True

        return False

# ════════════════════════════════════════════════════════════════
#  SMOOTH CURSOR  — double exponential (position + velocity)
# ════════════════════════════════════════════════════════════════
class SmoothCursor:
    def __init__(self, alpha=0.60, beta=0.18):
        self.alpha = alpha
        self.beta  = beta
        self.s     = None
        self.b     = None

    def update(self, x, y):
        if self.s is None:
            self.s = np.array([x,y], dtype=float)
            self.b = np.zeros(2)
            return int(x), int(y)
        prev_s = self.s.copy()
        self.s = self.alpha*np.array([x,y]) + (1-self.alpha)*(self.s+self.b)
        self.b = self.beta*(self.s-prev_s) + (1-self.beta)*self.b
        return int(self.s[0]), int(self.s[1])

    def reset(self):
        self.s = None
        self.b = None

# ════════════════════════════════════════════════════════════════
#  CATMULL-ROM SPLINE  — smooth curved strokes
# ════════════════════════════════════════════════════════════════
def catmull_rom(pts, steps=6):
    if len(pts) < 4:
        return list(pts)
    result = []
    for i in range(1, len(pts)-2):
        p0,p1,p2,p3 = pts[i-1],pts[i],pts[i+1],pts[i+2]
        for j in range(steps):
            t  = j/steps
            t2 = t*t; t3 = t2*t
            x  = 0.5*((2*p1[0])+(-p0[0]+p2[0])*t+
                      (2*p0[0]-5*p1[0]+4*p2[0]-p3[0])*t2+
                      (-p0[0]+3*p1[0]-3*p2[0]+p3[0])*t3)
            y  = 0.5*((2*p1[1])+(-p0[1]+p2[1])*t+
                      (2*p0[1]-5*p1[1]+4*p2[1]-p3[1])*t2+
                      (-p0[1]+3*p1[1]-3*p2[1]+p3[1])*t3)
            result.append((int(x),int(y)))
    return result

# ════════════════════════════════════════════════════════════════
#  STROKE MANAGER  — buffers points, draws spline curves
# ════════════════════════════════════════════════════════════════
class StrokeManager:
    """
    Buffers cursor points and draws smooth Catmull-Rom splines.
    KEY FIX: minimum movement threshold — only adds a point if
    the finger moved enough. This stops jitter from drawing
    blobs when the hand is steady between letters.
    """
    def __init__(self, buf_size=8):
        self.buf     = deque(maxlen=buf_size)
        self.active  = False
        self.last_x  = 0
        self.last_y  = 0
        self.MIN_MOVE = 4    # pixels — must move this much to register

    def begin(self):
        self.active  = True
        self.last_x  = 0
        self.last_y  = 0
        self.buf.clear()

    def end(self):
        """PEN UP — completely reset, no bleed into next stroke."""
        self.active  = False
        self.last_x  = 0
        self.last_y  = 0
        self.buf.clear()

    def add(self, x, y, canvas):
        if not self.active:
            return

        # Only register point if finger moved enough (kills jitter blobs)
        moved = math.hypot(x-self.last_x, y-self.last_y)
        if self.last_x != 0 and moved < self.MIN_MOVE:
            return

        # If jump is too large (>150px) — hand repositioned between letters
        # Treat as pen-up/pen-down, don't connect
        if self.last_x != 0 and moved > 150:
            self.buf.clear()   # break the stroke — don't connect letters

        self.last_x, self.last_y = x, y
        self.buf.append((x,y))

        if len(self.buf) >= 4:
            curve = catmull_rom(list(self.buf), steps=5)
            for i in range(1, len(curve)):
                ax,ay = curve[i-1]
                bx,by = curve[i]
                spd   = math.hypot(bx-ax, by-ay)
                press = max(0.80, 1.0 - spd/100.0)
                sz    = max(1, int(canvas.brush_size * press))
                canvas.stroke(ax,ay,bx,by,sz)

# ════════════════════════════════════════════════════════════════
#  DRAWING CANVAS
# ════════════════════════════════════════════════════════════════
class Canvas:
    def __init__(self, W, H):
        self.W, self.H  = W, H
        self.img        = np.zeros((H,W,3), np.uint8)
        self.color_idx  = 0
        self.brush_idx  = 0
        self.brush_size = 5       # thin default — like a real pen

    @property
    def color(self): return COLORS[self.color_idx][1]
    @property
    def color_name(self): return COLORS[self.color_idx][0]
    @property
    def brush(self): return BRUSHES[self.brush_idx]

    def stroke(self, x1,y1,x2,y2, sz=None):
        c = self.color
        s = sz if sz else self.brush_size

        if self.brush == "pen":
            # Single clean anti-aliased line — thin and precise
            cv2.line(self.img,(x1,y1),(x2,y2),c,s,cv2.LINE_AA)

        elif self.brush == "marker":
            ov = self.img.copy()
            cv2.line(ov,(x1,y1),(x2,y2),c,s*2,cv2.LINE_AA)
            cv2.addWeighted(ov,0.5,self.img,0.5,0,self.img)

        elif self.brush == "neon":
            for w,a in [(s*5,0.04),(s*3,0.12),(s*2,0.40),(s,1.0)]:
                col=tuple(min(255,int(v*a)) for v in c)
                cv2.line(self.img,(x1,y1),(x2,y2),col,max(1,w),cv2.LINE_AA)
            cv2.line(self.img,(x1,y1),(x2,y2),(255,255,255),max(1,s//3),cv2.LINE_AA)

        elif self.brush == "spray":
            density=max(20,s*6)
            for _ in range(density):
                ang=random.uniform(0,2*math.pi)
                r  =min(random.expovariate(1/(s*1.5)),s*4)
                px2=int(x2+math.cos(ang)*r); py2=int(y2+math.sin(ang)*r)
                if 0<=px2<self.W and 0<=py2<self.H:
                    a=random.uniform(0.25,1.0)
                    cv2.circle(self.img,(px2,py2),
                               random.randint(1,max(2,s//4)),
                               tuple(min(255,int(v*a)) for v in c),-1)

        elif self.brush == "glow":
            for i in range(6,0,-1):
                a=0.06*(7-i)
                cv2.line(self.img,(x1,y1),(x2,y2),
                         tuple(min(255,int(v*a)) for v in c),s*i,cv2.LINE_AA)
            cv2.line(self.img,(x1,y1),(x2,y2),c,s,cv2.LINE_AA)
            cv2.line(self.img,(x1,y1),(x2,y2),(255,255,255),max(1,s//4),cv2.LINE_AA)

    def erase(self, x, y):
        r = max(25, self.brush_size*5)
        cv2.circle(self.img,(x,y),r,(0,0,0),-1)

    def clear(self):
        self.img[:] = 0

    def blend(self, frame):
        gray=cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)
        _,mask=cv2.threshold(gray,6,255,cv2.THRESH_BINARY)
        bg=cv2.bitwise_and(frame,frame,mask=cv2.bitwise_not(mask))
        fg=cv2.bitwise_and(self.img,self.img,mask=mask)
        return cv2.add(bg,fg)

# ════════════════════════════════════════════════════════════════
#  TOOLBAR
# ════════════════════════════════════════════════════════════════
TB = 80

class Toolbar:
    def __init__(self, W, H):
        self.W,self.H = W,H
        self.r        = 28
        cy            = TB//2
        n             = len(COLORS)
        gap           = max(65,(W-160)//(n+1))
        self.circles  = [(gap+i*gap,cy) for i in range(n)]
        self.slx      = W-50
        self.sl_top   = TB+80
        self.sl_bot   = H-80
        self.sl_h     = self.sl_bot-self.sl_top
        self._cc      = 0

    def in_bar(self,y): return y<TB

    def check(self,ix,iy,fs,canvas):
        if self._cc>0: self._cc-=1
        # Color: index up, middle down, in toolbar
        if fs["index_up"] and not fs["middle_up"] and iy<TB:
            for i,(cx,cy) in enumerate(self.circles):
                if math.hypot(ix-cx,iy-cy)<self.r+18 and self._cc==0:
                    canvas.color_idx=i; self._cc=30
                    print(f"🎨  {COLORS[i][0]}")
                    break
        # Slider
        if abs(ix-self.slx)<55 and fs["index_up"]:
            raw=max(0.0,min(1.0,(iy-self.sl_top)/max(1,self.sl_h)))
            canvas.brush_size=max(2,int(40-raw*38))   # range 2–40

    def draw(self,frame,canvas,fps,fs,mode,shake_prog):
        out=frame.copy()
        ov=out.copy()
        cv2.rectangle(ov,(0,0),(self.W,TB),(12,12,22),-1)
        cv2.addWeighted(ov,0.85,out,0.15,0,out)
        cv2.line(out,(0,TB),(self.W,TB),(55,55,80),1)

        # Help text
        cv2.putText(out,"S=Save  Q=Quit  B=Brush  C=Clear  [=smaller  ]=bigger",
                    (8,22),cv2.FONT_HERSHEY_SIMPLEX,0.38,(130,130,150),1,cv2.LINE_AA)
        cv2.putText(out,f"Brush:{canvas.brush.upper()}  Size:{canvas.brush_size}",
                    (8,44),cv2.FONT_HERSHEY_SIMPLEX,0.40,(150,150,170),1,cv2.LINE_AA)
        cv2.putText(out,"LIFT middle finger to stop drawing | MOVE=pen up",
                    (8,64),cv2.FONT_HERSHEY_SIMPLEX,0.36,(100,100,130),1,cv2.LINE_AA)

        # Color circles
        for i,(cx,cy) in enumerate(self.circles):
            bgr=COLORS[i][1]
            if i==canvas.color_idx:
                cv2.circle(out,(cx,cy),self.r+8,(255,255,255),2)
                cv2.circle(out,(cx,cy),self.r+4,bgr,2)
            cv2.circle(out,(cx,cy),self.r,bgr,-1)
            cv2.circle(out,(cx,cy),self.r,(180,180,200),1)

        # Slider
        cv2.line(out,(self.slx,self.sl_top),(self.slx,self.sl_bot),(50,50,75),2)
        for pct in (0,25,50,75,100):
            ty=self.sl_top+int(pct/100*self.sl_h)
            cv2.line(out,(self.slx-6,ty),(self.slx+6,ty),(70,70,95),1)
        ratio=(40-canvas.brush_size)/38.0
        ty2=int(self.sl_top+ratio*self.sl_h)
        cv2.circle(out,(self.slx,ty2),14,canvas.color,-1)
        cv2.circle(out,(self.slx,ty2),14,(255,255,255),2)
        cv2.putText(out,str(canvas.brush_size),(self.slx-11,ty2+5),
                    cv2.FONT_HERSHEY_SIMPLEX,0.38,(0,0,0),1,cv2.LINE_AA)
        cv2.putText(out,"SIZE",(self.slx-22,self.sl_bot+20),
                    cv2.FONT_HERSHEY_SIMPLEX,0.36,(90,90,120),1,cv2.LINE_AA)

        # Mode badge
        mc={"DRAW":(0,220,80),"MOVE":(200,200,0),"ERASE":(50,60,220),
            "CLEAR!":(0,200,200)}.get(mode,(180,180,180))
        cv2.putText(out,mode,(10,TB+32),
                    cv2.FONT_HERSHEY_SIMPLEX,0.9,mc,2,cv2.LINE_AA)

        # Finger state
        ic=(0,220,80) if fs.get("index_up") else (110,110,120)
        mc2=(0,220,80) if fs.get("middle_up") else (110,110,120)
        cv2.putText(out,f"Index:{'UP' if fs.get('index_up') else 'down'}",(10,TB+55),
                    cv2.FONT_HERSHEY_SIMPLEX,0.38,ic,1,cv2.LINE_AA)
        cv2.putText(out,f"Middle:{'UP' if fs.get('middle_up') else 'down'}",(10,TB+70),
                    cv2.FONT_HERSHEY_SIMPLEX,0.38,mc2,1,cv2.LINE_AA)

        # Color swatch
        cv2.rectangle(out,(10,TB+80),(65,TB+98),canvas.color,-1)
        cv2.rectangle(out,(10,TB+80),(65,TB+98),(180,180,180),1)
        cv2.putText(out,canvas.color_name,(70,TB+94),
                    cv2.FONT_HERSHEY_SIMPLEX,0.38,canvas.color,1,cv2.LINE_AA)

        # Shake progress bar (shows when hand open + raised)
        if shake_prog > 0:
            bar_w=200; bar_x=W//2-bar_w//2; bar_y=TB+10
            cv2.rectangle(out,(bar_x,bar_y),(bar_x+bar_w,bar_y+16),(30,30,40),-1)
            filled=int(shake_prog/3*bar_w)
            cv2.rectangle(out,(bar_x,bar_y),(bar_x+filled,bar_y+16),(0,200,200),-1)
            cv2.rectangle(out,(bar_x,bar_y),(bar_x+bar_w,bar_y+16),(80,80,100),1)
            cv2.putText(out,"SHAKE to CLEAR!",(bar_x+10,bar_y+13),
                        cv2.FONT_HERSHEY_SIMPLEX,0.40,(255,255,255),1,cv2.LINE_AA)

        # FPS
        cv2.putText(out,f"FPS:{fps}",(self.W-75,self.H-8),
                    cv2.FONT_HERSHEY_SIMPLEX,0.40,(60,60,90),1,cv2.LINE_AA)

        return out

# ════════════════════════════════════════════════════════════════
#  EFFECTS
# ════════════════════════════════════════════════════════════════
class Sparks:
    def __init__(self,W,H):
        self.W,self.H,self.pool=W,H,[]
    def emit(self,x,y,col,n=8,spd=4):
        for _ in range(n):
            a=random.uniform(0,math.pi*2); s=random.uniform(0.5,spd)
            self.pool.append({'x':float(x),'y':float(y),
                'vx':math.cos(a)*s,'vy':math.sin(a)*s-random.uniform(0.5,2),
                'life':1.0,'decay':random.uniform(0.04,0.10),
                'sz':random.uniform(2,5),'col':col})
    def draw(self,frame):
        alive=[]
        for p in self.pool:
            p['x']+=p['vx']; p['y']+=p['vy']
            p['vy']+=0.1;    p['life']-=p['decay']
            if p['life']<=0: continue
            alive.append(p)
            cx,cy=int(p['x']),int(p['y'])
            if 0<=cx<self.W and 0<=cy<self.H:
                c=tuple(min(255,int(v*p['life'])) for v in p['col'])
                cv2.circle(frame,(cx,cy),max(1,int(p['sz']*p['life'])),c,-1)
        self.pool=alive; return frame

class Flash:
    def __init__(self): self.a=0.0
    def trigger(self,a=0.45): self.a=a
    def draw(self,frame):
        if self.a<=0: return frame
        ov=frame.copy()
        cv2.rectangle(ov,(0,0),(frame.shape[1],frame.shape[0]),(255,255,255),-1)
        out=cv2.addWeighted(ov,self.a,frame,1-self.a,0)
        self.a=max(0.0,self.a-0.07); return out

# ════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════
def main():
    cap=cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
    cap.set(cv2.CAP_PROP_FPS,30)
    W=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"📷  Camera: {W}x{H}")

    detector = make_detector()
    canvas   = Canvas(W,H)
    toolbar  = Toolbar(W,H)
    sparks   = Sparks(W,H)
    flash    = Flash()
    cursor   = SmoothCursor(alpha=0.62, beta=0.18)
    stroke   = StrokeManager(buf_size=8)
    shaker   = ShakeDetector()

    ts          = 0
    fps,fc,ft   = 0,0,time.time()
    save_banner = 0
    erase_hold  = 0
    was_drawing = False
    fs          = {"index_up":False,"middle_up":False,"fist":False,
                   "all_open":False,"wrist_y":H}
    mode_text   = "MOVE"
    shake_prog  = 0
    draw_confirm = 0   # must hold index-up for N frames before drawing starts

    print("="*55)
    print("  🎨  GESTURE DRAWING BOARD v4 — READY!")
    print("="*55)
    print("  ☝  INDEX only       →  DRAW")
    print("  ✌  INDEX + MIDDLE   →  MOVE (PEN LIFTED)")
    print("  🖐  Open hand HIGH  →  shake to CLEAR")
    print("  ✊  Hold FIST       →  ERASE")
    print()
    print("  HOW TO WRITE SEPARATE LETTERS:")
    print("  Write '1' with index only → raise middle finger")
    print("  (pen lifts) → move hand → lower middle → write '0'")
    print("="*55)

    while True:
        ret,frame=cap.read()
        if not ret: break
        frame=cv2.flip(frame,1)

        fc+=1
        if fc>=20:
            fps=int(fc/(time.time()-ft+1e-6)); fc=0; ft=time.time()

        ts+=1
        lm=get_landmarks(detector,frame,ts)

        cx,cy=W//2,H//2
        if lm is not None:
            draw_hand(frame,lm,W,H)

            raw_x=int(lm[8].x*W); raw_y=int(lm[8].y*H)
            cx,cy=cursor.update(raw_x,raw_y)
            fs=get_finger_state(lm,W,H)

            # ── SHAKE TO CLEAR ───────────────────────────────
            # Count direction changes for progress bar display
            cleared=shaker.update(cx,fs["all_open"],fs["wrist_y"],H)
            if fs["all_open"] and fs["wrist_y"]<H*0.45:
                # Count approx progress
                shake_prog=min(3,len(shaker.x_history)//6)
            else:
                shake_prog=0

            if cleared:
                canvas.clear()
                flash.trigger(0.5)
                sparks.emit(W//2,H//2,(0,220,200),n=80,spd=10)
                mode_text="CLEAR!"
                print("🗑  Canvas cleared by shake!")

            # ── TOOLBAR ──────────────────────────────────────
            toolbar.check(cx,cy,fs,canvas)
            in_tb=toolbar.in_bar(cy)

            # ── ERASE ─────────────────────────────────────────
            if fs["fist"]:
                erase_hold=min(erase_hold+1,10)
            else:
                erase_hold=0

            if erase_hold>=6 and not in_tb:
                mode_text="ERASE"
                canvas.erase(cx,cy)
                if was_drawing:
                    stroke.end(); was_drawing=False
                cursor.reset()

            # ── DRAW — index UP, middle DOWN ──────────────────
            elif fs["index_up"] and not fs["middle_up"] and not in_tb and not cleared:
                draw_confirm = min(draw_confirm+1, 5)
                # Only start drawing after index held steady for 3 frames
                # This prevents blobs from accidental brief index raises
                if draw_confirm >= 3:
                    mode_text="DRAW"
                    if not was_drawing:
                        stroke.begin()
                        was_drawing=True
                    stroke.add(cx,cy,canvas)
                    if canvas.brush in ("neon","glow"):
                        sparks.emit(cx,cy,canvas.color,n=2,spd=1.5)
                else:
                    mode_text="MOVE"

            # ── MOVE — pen LIFTED (middle up OR in toolbar) ───
            else:
                draw_confirm = 0   # reset confirmation counter
                if was_drawing:
                    stroke.end()
                    was_drawing=False
                if mode_text != "CLEAR!":
                    mode_text="MOVE"

            # ── CURSOR RING ───────────────────────────────────
            if mode_text=="DRAW":
                rc=canvas.color; rr=canvas.brush_size+8
            elif mode_text=="ERASE":
                rc=(50,50,220); rr=max(25,canvas.brush_size*5)
            elif mode_text=="CLEAR!":
                rc=(0,200,200); rr=30
            else:
                rc=(180,180,180); rr=16
            cv2.circle(frame,(cx,cy),rr+5,tuple(max(0,v//5) for v in rc),1,cv2.LINE_AA)
            cv2.circle(frame,(cx,cy),rr+1,tuple(max(0,v//2) for v in rc),1,cv2.LINE_AA)
            cv2.circle(frame,(cx,cy),rr,rc,2,cv2.LINE_AA)
            cv2.circle(frame,(cx,cy),4,(255,255,255),-1,cv2.LINE_AA)
            cv2.putText(frame,mode_text,(cx+rr+8,cy+6),
                        cv2.FONT_HERSHEY_SIMPLEX,0.50,rc,2,cv2.LINE_AA)

        else:
            if was_drawing: stroke.end(); was_drawing=False
            fs={"index_up":False,"middle_up":False,"fist":False,
                "all_open":False,"wrist_y":H}
            mode_text="MOVE"; shake_prog=0
            cursor.reset()
            cv2.putText(frame,"Show your hand to camera",
                        (W//2-190,H//2),cv2.FONT_HERSHEY_SIMPLEX,
                        0.85,(80,80,220),2,cv2.LINE_AA)

        # ── COMPOSE ───────────────────────────────────────────
        if canvas.brush in ("neon","glow"):
            blur    = cv2.GaussianBlur(canvas.img,(23,23),0)
            bloomed = cv2.addWeighted(canvas.img,1.0,blur,0.40,0)
            gray    = cv2.cvtColor(bloomed,cv2.COLOR_BGR2GRAY)
            _,mask  = cv2.threshold(gray,6,255,cv2.THRESH_BINARY)
            bg      = cv2.bitwise_and(frame,frame,mask=cv2.bitwise_not(mask))
            fg      = cv2.bitwise_and(bloomed,bloomed,mask=mask)
            output  = cv2.add(bg,fg)
        else:
            output=canvas.blend(frame)

        output=sparks.draw(output)
        output=flash.draw(output)
        output=toolbar.draw(output,canvas,fps,fs,mode_text,shake_prog)

        if save_banner>0:
            txt="  IMAGE SAVED!  "
            (tw,th),_=cv2.getTextSize(txt,cv2.FONT_HERSHEY_SIMPLEX,1.1,3)
            bx,by=(W-tw)//2,H//2
            cv2.rectangle(output,(bx-14,by-th-14),(bx+tw+14,by+14),(0,0,0),-1)
            cv2.rectangle(output,(bx-14,by-th-14),(bx+tw+14,by+14),(0,200,80),2)
            cv2.putText(output,txt,(bx,by),cv2.FONT_HERSHEY_SIMPLEX,
                        1.1,(0,230,100),3,cv2.LINE_AA)
            save_banner-=1

        cv2.imshow("Gesture Drawing Board",output)
        key=cv2.waitKey(1)&0xFF

        if key in (ord('q'),27):    break
        elif key==ord('s'):
            os.makedirs("saves",exist_ok=True)
            fn=f"saves/drawing_{int(time.time())}.png"
            cv2.imwrite(fn,canvas.img)
            print(f"✅  Saved → {fn}"); save_banner=60
        elif key==ord('c'):
            canvas.clear(); flash.trigger(0.35)
            sparks.emit(W//2,H//2,(200,200,255),n=60,spd=8)
        elif key==ord('b'):
            canvas.brush_idx=(canvas.brush_idx+1)%len(BRUSHES)
            print(f"🖌  Brush → {canvas.brush}")
        elif key==ord(']') or key==ord('='):
            canvas.brush_size=min(40,canvas.brush_size+1)
        elif key==ord('[') or key==ord('-'):
            canvas.brush_size=max(2,canvas.brush_size-1)

    cap.release()
    cv2.destroyAllWindows()
    print("👋  Goodbye!")

if __name__=="__main__":
    main()