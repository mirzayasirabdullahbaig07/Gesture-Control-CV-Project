import cv2, numpy as np, random, math

class Particles:
    def __init__(self, W, H):
        self.W, self.H = W, H
        self.pool = []

    def emit(self, x, y, color, count=6, speed=3.0):
        for _ in range(count):
            ang = random.uniform(0, math.pi*2)
            spd = random.uniform(0.5, speed)
            self.pool.append(dict(
                x=x, y=y,
                vx=math.cos(ang)*spd,
                vy=math.sin(ang)*spd - random.uniform(0.5,2.0),
                life=1.0, decay=random.uniform(0.03,0.09),
                size=random.uniform(1.5,4.0), color=color
            ))

    def draw(self, frame):
        alive = []
        for p in self.pool:
            p['x'] += p['vx']; p['y'] += p['vy']
            p['vy'] += 0.1;    p['life'] -= p['decay']
            if p['life'] <= 0: continue
            alive.append(p)
            cx,cy = int(p['x']), int(p['y'])
            if not(0<=cx<self.W and 0<=cy<self.H): continue
            a   = p['life']
            col = tuple(min(255,int(c*a)) for c in p['color'])
            r   = max(1, int(p['size']*a))
            cv2.circle(frame,(cx,cy),r,col,-1)
        self.pool = alive
        return frame

class Flash:
    def __init__(self): self.alpha=0.0
    def trigger(self, a=0.45): self.alpha=a
    def draw(self, frame, color=(255,255,255)):
        if self.alpha<=0: return frame
        ov = frame.copy()
        cv2.rectangle(ov,(0,0),(frame.shape[1],frame.shape[0]),color,-1)
        out = cv2.addWeighted(ov,self.alpha,frame,1-self.alpha,0)
        self.alpha = max(0.0, self.alpha-0.07)
        return out

def bloom(canvas):
    """Add soft glow bloom to the canvas before blending."""
    blur = cv2.GaussianBlur(canvas,(25,25),0)
    return cv2.addWeighted(canvas,1.0,blur,0.4,0)