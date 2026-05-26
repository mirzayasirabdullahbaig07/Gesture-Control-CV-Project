import cv2, numpy as np, time, os
from hand_tracker  import HandTracker
from drawing_engine import DrawingEngine
from gestures      import GestureEngine, DRAW, MOVE, ERASE, CLEAR
from effects       import Particles, Flash, bloom
from ui            import UI, TB

# ================================================================
#  GESTURE DRAWING BOARD  ·  main.py
#
#  HAND GESTURES:
#    ☝  INDEX finger only          →  DRAW on screen
#    ✌  INDEX + MIDDLE fingers     →  MOVE cursor / select color
#    ✊  FIST  (all fingers down)  →  ERASE
#    🖐  ALL 5 fingers open        →  CLEAR canvas
#
#  KEYBOARD SHORTCUTS:
#    S  →  Save PNG to /saves folder
#    C  →  Clear canvas
#    B  →  Cycle brush  (pen → neon → spray → glow → marker)
#    +  →  Brush size up
#    -  →  Brush size down
#    Q  →  Quit
# ================================================================

def blend(frame, canvas):
    """Overlay drawing canvas — black pixels are transparent."""
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask  = cv2.threshold(gray,8,255,cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
    fg = cv2.bitwise_and(canvas,canvas,mask=mask)
    return cv2.add(bg,fg)

def smooth(px,py,nx,ny,a=0.55):
    if px==0 and py==0: return float(nx),float(ny)
    return a*nx+(1-a)*px, a*ny+(1-a)*py

# ── Camera ───────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"\n📷  Camera: {W}x{H}")

# ── Init modules ─────────────────────────────────────────────
tracker  = HandTracker()
engine   = DrawingEngine(W,H)
gesture  = GestureEngine(cooldown=15)
ui       = UI(W,H)
sparks   = Particles(W,H)
flash    = Flash()

# ── State ────────────────────────────────────────────────────
sx,sy       = 0.0,0.0
px,py       = 0,0
mode        = MOVE
fingers     = [0]*5
fps,fc,ft   = 0,0,time.time()
save_banner = 0

print("=" * 50)
print("  🎨  GESTURE DRAWING BOARD  —  READY!")
print("=" * 50)
print()
print("  HOW TO DRAW:")
print("  👉  Raise ONLY your INDEX finger → draws!")
print("  ✌   INDEX + MIDDLE → move / pick color")
print("  ✊  Close fist → erase")
print("  🖐   Open all 5 fingers → clear screen")
print()
print("  TIP: Use INDEX+MIDDLE to hover over a")
print("       color circle in the toolbar to select it.")
print()

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame,1)

    # FPS
    fc += 1
    if fc >= 20:
        fps = int(fc/(time.time()-ft+1e-6))
        fc=0; ft=time.time()

    # Hand detection
    frame, lm = tracker.process(frame)

    if lm is not None:
        # Index fingertip pixel  (landmark 8)
        raw_x = int(lm[8].x*W)
        raw_y = int(lm[8].y*H)
        sx,sy = smooth(sx,sy,raw_x,raw_y,0.60)
        cx,cy = int(sx),int(sy)

        fingers = tracker.fingers_up(lm)
        mode    = gesture.update(fingers)

        # CLEAR gesture
        if mode == CLEAR:
            engine.clear()
            flash.trigger(0.45)
            sparks.emit(W//2,H//2,(200,200,255),count=80,speed=9)
            gesture.force_cooldown(45)

        # UI interaction (colors, brush, size slider)
        ui.interact(cx,cy,fingers,engine)

        in_tb = ui.in_toolbar(cy)

        # DRAW
        if mode==DRAW and not in_tb:
            if px and py:
                engine.draw_stroke(px,py,cx,cy)
                if engine.brush_name in ("neon","glow"):
                    sparks.emit(cx,cy,engine.color,count=3,speed=2)
            px,py = cx,cy

        # ERASE
        elif mode==ERASE and not in_tb:
            engine.erase(cx,cy)
            px,py = 0,0

        else:
            px,py = 0,0

        # Cursor
        if mode==DRAW:
            rc,rr = engine.color, engine.brush_size+8
        elif mode==ERASE:
            rc,rr = (50,50,220), engine.brush_size*6
        else:
            rc,rr = (200,200,200), 16

        cv2.circle(frame,(cx,cy),rr+4,tuple(max(0,v//4) for v in rc),1,cv2.LINE_AA)
        cv2.circle(frame,(cx,cy),rr,rc,2,cv2.LINE_AA)
        cv2.circle(frame,(cx,cy),5,(255,255,255),-1,cv2.LINE_AA)

        # Mode label near cursor
        mlbl={"draw":"DRAW","move":"MOVE","erase":"ERASE","clear":"CLEAR!"}.get(mode,"")
        cv2.putText(frame,mlbl,(cx+rr+8,cy+6),
                    cv2.FONT_HERSHEY_SIMPLEX,0.55,rc,2,cv2.LINE_AA)
    else:
        fingers=[0]*5; px,py=0,0
        # No-hand prompt
        cv2.putText(frame,"Show your hand to the camera 👋",
                    (W//2-240,H//2),cv2.FONT_HERSHEY_SIMPLEX,0.9,(80,80,210),2,cv2.LINE_AA)

    # Compose frame
    glowed  = bloom(engine.canvas)
    output  = blend(frame,glowed)
    output  = sparks.draw(output)
    output  = flash.draw(output)
    output  = ui.draw(output,mode,engine,fps,fingers)

    # Save banner
    if save_banner>0:
        txt="  ✅  SAVED!  "
        (tw,th),_=cv2.getTextSize(txt,cv2.FONT_HERSHEY_SIMPLEX,1.1,3)
        bx,by=(W-tw)//2,H//2
        cv2.rectangle(output,(bx-14,by-th-14),(bx+tw+14,by+14),(0,0,0),-1)
        cv2.rectangle(output,(bx-14,by-th-14),(bx+tw+14,by+14),(0,200,80),2)
        cv2.putText(output,txt,(bx,by),cv2.FONT_HERSHEY_SIMPLEX,1.1,(0,230,100),3,cv2.LINE_AA)
        save_banner-=1

    cv2.imshow("Gesture Drawing Board",output)

    key=cv2.waitKey(1)&0xFF
    if key in (ord('q'),27):   break
    elif key==ord('s'):
        os.makedirs("saves",exist_ok=True)
        f=f"saves/drawing_{int(time.time())}.png"
        cv2.imwrite(f,engine.canvas)
        print(f"✅  Saved → {f}")
        save_banner=60
    elif key==ord('c'):
        engine.clear(); flash.trigger(0.35)
        sparks.emit(W//2,H//2,(200,200,255),count=50,speed=7)
    elif key==ord('b'):
        engine.next_brush()
        print(f"🖌  Brush → {engine.brush_name}")
    elif key in (ord('+'),ord('=')):
        engine.set_brush_size(engine.brush_size+1)
    elif key==ord('-'):
        engine.set_brush_size(engine.brush_size-1)

cap.release()
cv2.destroyAllWindows()
print("👋  Goodbye!")