import cv2, numpy as np

PALETTE = [
    ("Red",     (  0,   0, 255)),
    ("Orange",  (  0, 140, 255)),
    ("Yellow",  (  0, 220, 255)),
    ("Green",   (  0, 220,   0)),
    ("Cyan",    (255, 220,   0)),
    ("Blue",    (255,  60,   0)),
    ("Purple",  (220,  40, 200)),
    ("White",   (255, 255, 255)),
]
BRUSHES = ["pen","neon","spray","glow","marker"]

class DrawingEngine:
    def __init__(self, W, H):
        self.W, self.H   = W, H
        self.canvas      = np.zeros((H,W,3), dtype=np.uint8)
        self.color_idx   = 0
        self.brush_idx   = 0
        self.brush_size  = 8

    @property
    def color(self):      return PALETTE[self.color_idx][1]
    @property
    def color_name(self): return PALETTE[self.color_idx][0]
    @property
    def brush_name(self): return BRUSHES[self.brush_idx]
    @property
    def palette(self):    return PALETTE

    def draw_stroke(self, x1,y1, x2,y2):
        c,s,bt = self.color, self.brush_size, self.brush_name

        if bt == "pen":
            cv2.line(self.canvas,(x1,y1),(x2,y2),c,s,cv2.LINE_AA)

        elif bt == "marker":
            # Semi-transparent wide marker
            tmp = self.canvas.copy()
            cv2.line(tmp,(x1,y1),(x2,y2),c,s*3,cv2.LINE_AA)
            cv2.addWeighted(tmp,0.35,self.canvas,0.65,0,self.canvas)

        elif bt == "neon":
            for w,a in [(s*5,0.06),(s*3,0.15),(s*2,0.35),(s,1.0)]:
                col = tuple(min(255,int(v*a)) for v in c)
                cv2.line(self.canvas,(x1,y1),(x2,y2),col,max(1,w),cv2.LINE_AA)
            cv2.line(self.canvas,(x1,y1),(x2,y2),(255,255,255),max(1,s//3),cv2.LINE_AA)

        elif bt == "spray":
            density = max(15, s*7)
            for _ in range(density):
                angle = np.random.uniform(0,2*np.pi)
                r     = np.random.exponential(scale=s*1.3)
                r     = min(r, s*3.5)
                px    = int(x2 + np.cos(angle)*r)
                py    = int(y2 + np.sin(angle)*r)
                if 0<=px<self.W and 0<=py<self.H:
                    a   = np.random.uniform(0.25,1.0)
                    dot = tuple(min(255,int(v*a)) for v in c)
                    dr  = np.random.randint(1,max(2,s//4))
                    cv2.circle(self.canvas,(px,py),dr,dot,-1)

        elif bt == "glow":
            for i in range(6,0,-1):
                a   = 0.06*(7-i)
                col = tuple(min(255,int(v*a)) for v in c)
                cv2.line(self.canvas,(x1,y1),(x2,y2),col,s*i,cv2.LINE_AA)
            cv2.line(self.canvas,(x1,y1),(x2,y2),c,s,cv2.LINE_AA)
            cv2.line(self.canvas,(x1,y1),(x2,y2),(255,255,255),max(1,s//4),cv2.LINE_AA)

    def erase(self, x, y):
        cv2.circle(self.canvas,(x,y),self.brush_size*6,(0,0,0),-1)

    def clear(self):
        self.canvas[:] = 0

    def set_color(self, idx):   self.color_idx  = int(idx) % len(PALETTE)
    def next_brush(self):       self.brush_idx  = (self.brush_idx+1) % len(BRUSHES)
    def set_brush_size(self,s): self.brush_size = max(2,min(40,int(s)))