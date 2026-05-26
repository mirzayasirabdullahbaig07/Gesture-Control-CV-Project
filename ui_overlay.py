import cv2, numpy as np

TB = 90      # toolbar height px

class UI:
    def __init__(self, W, H):
        self.W, self.H = W, H

        # Color circles
        self.cr = 24   # circle radius
        self.cy = 44   # circle center y
        n   = 8
        gap = 70
        sx  = (W - (n-1)*gap) // 2
        self.color_pos = [(sx + i*gap, self.cy) for i in range(n)]

        # Brush button  (x,y,w,h)
        self.bbtn = (W-170, 22, 150, 46)

        # Size slider (right edge)
        self.slx = W - 48
        self.sl_top = TB + 60
        self.sl_bot = H - 80
        self.sl_h   = self.sl_bot - self.sl_top

        # Cooldowns
        self._cc = 0   # color
        self._bc = 0   # brush btn

    def in_toolbar(self, y): return y < TB

    # ── Detect finger interactions ────────────────────────────
    def interact(self, sx, sy, fingers, engine):
        if len(fingers)<5: return
        idx_up = fingers[1]; mid_up = fingers[2]
        select = idx_up==1 and mid_up==0   # pointing gesture = select

        # Color circles (pointing into toolbar)
        if self._cc>0: self._cc-=1
        if self._cc==0 and select and sy<TB:
            for i,(cx,cy) in enumerate(self.color_pos):
                if abs(sx-cx)<=self.cr+12 and abs(sy-cy)<=self.cr+12:
                    engine.set_color(i); self._cc=25; break

        # Brush button
        if self._bc>0: self._bc-=1
        bx,by,bw,bh = self.bbtn
        if self._bc==0 and select and bx<=sx<=bx+bw and by<=sy<=by+bh:
            engine.next_brush(); self._bc=30
            print(f"🖌  Brush → {engine.brush_name}")

        # Size slider (index finger near right edge)
        if abs(sx-self.slx)<45 and idx_up:
            raw  = (sy-self.sl_top)/max(1,self.sl_h)
            raw  = max(0.0,min(1.0,raw))
            size = int(40-raw*38)   # top=40 bottom=2
            engine.set_brush_size(size)

    # ── Draw HUD ──────────────────────────────────────────────
    def draw(self, frame, gesture, engine, fps, fingers):
        out = frame.copy()

        # Toolbar background (semi-transparent dark)
        ov = out.copy()
        cv2.rectangle(ov,(0,0),(self.W,TB),(10,10,20),-1)
        cv2.addWeighted(ov,0.80,out,0.20,0,out)
        cv2.line(out,(0,TB),(self.W,TB),(60,60,90),1)

        # ── Color circles ─────────────────────────────────────
        for i,(cx,cy) in enumerate(self.color_pos):
            bgr  = engine.palette[i][1]
            name = engine.palette[i][0]
            sel  = (i==engine.color_idx)
            if sel:
                cv2.circle(out,(cx,cy),self.cr+8,bgr,2)
                cv2.circle(out,(cx,cy),self.cr+5,(255,255,255),1)
            cv2.circle(out,(cx,cy),self.cr,bgr,-1)
            cv2.circle(out,(cx,cy),self.cr,(200,200,220),1)
            tx = cx - len(name)*3
            cv2.putText(out,name,(tx,cy+self.cr+14),
                        cv2.FONT_HERSHEY_SIMPLEX,0.3,(110,110,140),1,cv2.LINE_AA)

        # ── Brush button ──────────────────────────────────────
        bx,by,bw,bh = self.bbtn
        cv2.rectangle(out,(bx,by),(bx+bw,by+bh),(25,18,42),-1)
        cv2.rectangle(out,(bx,by),(bx+bw,by+bh),(120,70,200),1)
        icons={"pen":"PEN","neon":"NEON GLOW","spray":"SPRAY","glow":"GLOW","marker":"MARKER"}
        lbl = icons.get(engine.brush_name, engine.brush_name.upper())
        cv2.putText(out,lbl,(bx+10,by+30),
                    cv2.FONT_HERSHEY_SIMPLEX,0.55,(180,140,255),1,cv2.LINE_AA)

        # ── Size slider ───────────────────────────────────────
        cv2.line(out,(self.slx,self.sl_top),(self.slx,self.sl_bot),(45,45,70),2)
        for pct in (0,25,50,75,100):
            ty = self.sl_top+int(pct/100*self.sl_h)
            cv2.line(out,(self.slx-6,ty),(self.slx+6,ty),(70,70,95),1)
        ratio  = (40-engine.brush_size)/38.0
        thumb_y= int(self.sl_top+ratio*self.sl_h)
        cv2.circle(out,(self.slx,thumb_y),14,engine.color,-1)
        cv2.circle(out,(self.slx,thumb_y),14,(255,255,255),1)
        cv2.putText(out,str(engine.brush_size),
                    (self.slx-11,thumb_y+5),
                    cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,0),1,cv2.LINE_AA)
        cv2.putText(out,"SIZE",(self.slx-22,self.sl_bot+22),
                    cv2.FONT_HERSHEY_SIMPLEX,0.38,(90,90,120),1,cv2.LINE_AA)

        # ── Mode badge ────────────────────────────────────────
        gcol = {"draw":(0,210,80),"move":(200,200,0),
                "erase":(40,60,220),"clear":(0,200,200)}.get(gesture,(180,180,180))
        glbl = {"draw":"✏ DRAW","move":"MOVE","erase":"ERASE","clear":"CLEAR"}.get(gesture,gesture.upper())
        (tw,th),_ = cv2.getTextSize(glbl,cv2.FONT_HERSHEY_SIMPLEX,0.7,2)
        px0,py0 = 12, TB+10
        cv2.rectangle(out,(px0-6,py0-4),(px0+tw+10,py0+th+8),(10,10,20),-1)
        cv2.rectangle(out,(px0-6,py0-4),(px0+tw+10,py0+th+8),gcol,1)
        cv2.putText(out,glbl,(px0,py0+th),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,gcol,2,cv2.LINE_AA)

        # ── Color swatch ──────────────────────────────────────
        sw_x,sw_y = 12,TB+62
        cv2.rectangle(out,(sw_x,sw_y),(sw_x+55,sw_y+20),engine.color,-1)
        cv2.rectangle(out,(sw_x,sw_y),(sw_x+55,sw_y+20),(200,200,200),1)
        cv2.putText(out,engine.color_name,(sw_x+60,sw_y+15),
                    cv2.FONT_HERSHEY_SIMPLEX,0.42,engine.color,1,cv2.LINE_AA)

        # ── Finger LEDs ───────────────────────────────────────
        for i,lbl in enumerate("TIMRP"):
            up  = fingers[i] if i<len(fingers) else 0
            col = (0,255,120) if up else (40,40,60)
            fx  = self.W-150+i*26; fy=self.H-18
            cv2.circle(out,(fx,fy-7),8,col,-1)
            cv2.putText(out,lbl,(fx-4,fy-3),
                        cv2.FONT_HERSHEY_SIMPLEX,0.3,(0,0,0),1,cv2.LINE_AA)

        # ── FPS ───────────────────────────────────────────────
        cv2.putText(out,f"FPS:{fps}",(self.W-90,self.H-8),
                    cv2.FONT_HERSHEY_SIMPLEX,0.42,(60,60,90),1,cv2.LINE_AA)

        # ── Bottom legend ─────────────────────────────────────
        hints = ["☝=Draw","✌=Move","✊=Erase","🖐=Clear","S=Save","B=Brush","Q=Quit"]
        hx,hy = 10, self.H-8
        for h in hints:
            cv2.putText(out,h,(hx,hy),cv2.FONT_HERSHEY_SIMPLEX,0.33,(75,75,110),1,cv2.LINE_AA)
            tw2,_ = cv2.getTextSize(h,cv2.FONT_HERSHEY_SIMPLEX,0.33,1)
            hx += tw2[0]+14

        return out