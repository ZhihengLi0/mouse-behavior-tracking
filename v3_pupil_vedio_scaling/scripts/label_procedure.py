"""Reference implementation of the v3 labelling procedure (AI part). All numbers are the ones quoted in LABELING_GUIDE.md.
label_frame(video_path, frame, prev=None, prior=None) -> dict with the 8 points, ellipse, confidence, flags.
"""
import numpy as np, cv2
from scipy.ndimage import gaussian_filter, label as cclabel, binary_fill_holes, binary_dilation, maximum_filter, distance_transform_edt, binary_opening
EYE_BOX=(150,900,200,720)          # x0,x1,y0,y1 search region (whole eye)
SIG=3.0                            # blur for segmentation
T_CORE=4.5                         # core mask threshold above the plateau level
T_PLATEAU=3.0                      # tighter threshold used only for the plateau tips in the fixed-aspect fallback
WEAK_OUT=8.0                       # contour point is 'weak' (bordered by a dark shoulder, not by iris/lid) if g(25 px outward) < core+8
GLINT_HI,GLINT_LO=45,12
LID_OUT_MIN=20; LID_SKIN_FRAC=0.35 # 'outside is lid/skin' if g(10 px outward) > core + max(20, 0.35*(skin-core)); skin = 80th percentile of g in the eye box
EYEREG_LEVEL=22                    # dark eye region (pupil+iris) = g < core+22 ; used for lids / corners
DEFAULT_PRIOR=dict(aspect=1.45,tilt=0.0)   # per-mouse: A 1.15, Pluto 1.45, Terra 1.45 (median W/H of good open-eye fits)   # W/H and tilt (deg, ccw from x axis) used when top+bottom are hidden

def ellipse_pts(e,n=721):
    (ex,ey),(a1,a2),ang=e; t=np.linspace(0,2*np.pi,n); a=np.deg2rad(ang)
    return ex+(a1/2)*np.cos(t)*np.cos(a)-(a2/2)*np.sin(t)*np.sin(a), ey+(a1/2)*np.cos(t)*np.sin(a)+(a2/2)*np.sin(t)*np.cos(a)
def extremes(e):
    X,Y=ellipse_pts(e)
    return dict(pupil_left=(X.min(),Y[np.argmin(X)]),pupil_right=(X.max(),Y[np.argmax(X)]),pupil_top=(X[np.argmin(Y)],Y.min()),pupil_bottom=(X[np.argmax(Y)],Y.max()),W=X.max()-X.min(),H=Y.max()-Y.min())
def read_avg(cap,f,avg=1):
    """mean of frames f-avg..f+avg; falls back to the single frame if the eye moved (mean |f+1 - f-1| > 6 grey in the eye box)"""
    f0=max(0,f-avg); cap.set(cv2.CAP_PROP_POS_FRAMES,f0); ims=[]
    for k in range(f0,f+avg+1):
        ok,im=cap.read()
        if not ok: break
        ims.append(im[...,0].astype(np.float32))
    if not ims: return None
    if len(ims)>=3:
        x0,x1,y0,y1=EYE_BOX; d=np.abs(ims[-1][y0:y1,x0:x1]-ims[0][y0:y1,x0:x1]).mean()
        if d>6: return ims[min(avg,len(ims)-1)]
    return np.mean(ims,axis=0)
def glint_mask(g1,core):
    b=g1>core+GLINT_HI; lab,n=cclabel(b); big=np.zeros_like(b)
    if n:
        sizes=np.bincount(lab.ravel()); ok=(sizes<3000); ok[0]=False; big=ok[lab]
    big=binary_dilation(big,iterations=6)
    g8=gaussian_filter(g1,8); loc=(g1==maximum_filter(g1,size=15))&(g1>core+GLINT_LO)&(g8<core+20)
    return big|binary_dilation(loc,iterations=12)
def find_seed(g):
    x0,x1,y0,y1=EYE_BOX; g8=gaussian_filter(g,8); sub=g8[y0:y1,x0:x1]; iy,ix=np.unravel_index(np.argmin(sub),sub.shape); return x0+ix,y0+iy
def label_frame(im,seed=None,prior=None,ap_median=None):
    prior=prior or DEFAULT_PRIOR; H,W=im.shape; yy,xx=np.mgrid[0:H,0:W]
    g=gaussian_filter(im,SIG); g1=gaussian_filter(im,1.5)
    # 1. seed and core level
    if seed is None: seed=find_seed(g)
    sx,sy=int(seed[0]),int(seed[1]); g8=gaussian_filter(g,5)
    win=g8[max(0,sy-40):sy+41,max(0,sx-40):sx+41]; iy,ix=np.unravel_index(np.argmin(win),win.shape); sx=max(0,sx-40)+ix; sy=max(0,sy-40)+iy
    disc=(xx-sx)**2+(yy-sy)**2<25*25; core=float(np.percentile(g[disc],5))
    gm=glint_mask(g1,core)
    w=(~gm).astype(np.float32); num=gaussian_filter(np.where(gm,0,im).astype(np.float32),SIG); den=gaussian_filter(w,SIG)
    gs=np.where(den>0.3,num/np.maximum(den,1e-6),np.nan)
    x0,x1,y0,y1=EYE_BOX; box=(xx>=x0)&(xx<=x1)&(yy>=y0)&(yy<=y1)
    out=dict(core=core,seed=(sx,sy),confidence='none',flags=[])
    def segment(thr):
        m=(gs<core+thr)|gm; m[np.isnan(gs)&~gm]=False; m&=box; m=binary_fill_holes(m)
        lab,n=cclabel(m); l=lab[sy,sx]
        if l==0: return None
        return lab==l
    # 2. provisional mask -> iris level -> final threshold
    mm=segment(4.5)
    if mm is None: out['flags'].append('no dark region at seed'); return out
    ys,xs=np.nonzero(mm); ex,ey=xs.mean(),ys.mean(); rx=max(20,(xs.max()-xs.min())/2); ry=max(15,(ys.max()-ys.min())/2)
    rr=np.sqrt(((xx-ex)/rx)**2+((yy-ey)/ry)**2); ann=(rr>1.15)&(rr<1.4)&~gm&box
    iris=float(np.nanmedian(g[ann])) if ann.any() else core+10
    thr=T_CORE; skin=float(np.percentile(g[box],80)); out.update(iris=iris,thr=thr,skin=skin)
    mm=segment(thr)
    if mm is None: out['flags'].append('no dark region at seed'); return out
    # leak guard for dark/low-contrast videos: if most of the boundary is bordered by dark shadow (not iris), re-segment tighter
    def weak_fraction(m):
        cs,_=cv2.findContours(m.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE); c=max(cs,key=cv2.contourArea).reshape(-1,2).astype(float)
        mcx,mcy=c[:,0].mean(),c[:,1].mean(); vx=c[:,0]-mcx; vy=c[:,1]-mcy; nr=np.hypot(vx,vy)+1e-6
        ox=np.clip((c[:,0]+25*vx/nr).astype(int),0,W-1); oy=np.clip((c[:,1]+25*vy/nr).astype(int),0,H-1); return float((g[oy,ox]<core+WEAK_OUT).mean())
    if weak_fraction(mm)>0.5:
        m2=segment(T_PLATEAU)
        if m2 is not None and m2.sum()>200: mm=m2; thr=T_PLATEAU; out['thr']=thr; out['flags'].append('re-segmented at core+3 (dark surround)')
    ys,xs=np.nonzero(mm)
    if xs.min()<=x0 or xs.max()>=x1 or ys.min()<=y0 or ys.max()>=y1: out['flags'].append('mask touches search box (leak)')
    # 3. eye region (pupil+iris) for lids and corners
    er=(g<core+EYEREG_LEVEL)|gm; er&=box; er=binary_opening(binary_fill_holes(er),iterations=4); lab,n=cclabel(er); l=lab[sy,sx]; er=(lab==l) if l else er
    # corners: the eye region is a lens shape; take its extreme x within +-150 rows of the pupil centre, but stop where the region becomes thinner than 12 px (shadow leaks)
    band=np.zeros_like(er); band[max(0,sy-150):sy+150,:]=True; er2=er&band
    colh=er2.sum(axis=0); cols=np.nonzero(colh>=12)[0]
    if len(cols):
        # keep the contiguous run of columns containing sx
        runs=np.split(cols,np.nonzero(np.diff(cols)>1)[0]+1); run=[r for r in runs if r.min()<=sx<=r.max()]
        cols=run[0] if run else cols
        er2[:, :cols.min()]=False; er2[:, cols.max()+1:]=False
    eys,exs=np.nonzero(er2); er=er2
    # 4. contour, exclusions, ellipse
    cs,_=cv2.findContours(mm.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE); c=max(cs,key=cv2.contourArea).reshape(-1,2).astype(float)
    dg=distance_transform_edt(~gm); mcx,mcy=c[:,0].mean(),c[:,1].mean()
    vx=c[:,0]-mcx; vy=c[:,1]-mcy; nrm=np.hypot(vx,vy)+1e-6; vx/=nrm; vy/=nrm
    ox=np.clip((c[:,0]+10*vx).astype(int),0,W-1); oy=np.clip((c[:,1]+10*vy).astype(int),0,H-1)
    lidpt=g[oy,ox]>core+max(LID_OUT_MIN,LID_SKIN_FRAC*(skin-core)); glpt=dg[c[:,1].astype(int),c[:,0].astype(int)]<=6
    ox2=np.clip((c[:,0]+25*vx).astype(int),0,W-1); oy2=np.clip((c[:,1]+25*vy).astype(int),0,H-1)
    weak=(g[oy2,ox2]<core+WEAK_OUT)&~lidpt&~glpt
    keep=~lidpt&~glpt&~weak; pts=c[keep]
    out.update(mask=mm,contour=c,lidpt=lidpt,glpt=glpt,weak=weak,gm=gm,g=g,gs=gs,eyereg=er,weak_frac=float(weak.mean()))
    top_half=c[:,1]<mcy; lid_top=lidpt[top_half].mean() if top_half.any() else 0; lid_bot=lidpt[~top_half].mean() if (~top_half).any() else 0
    out.update(lid_top_frac=float(lid_top),lid_bot_frac=float(lid_bot))
    # eye closed?
    ap=ys.max()-ys.min(); wd=xs.max()-xs.min()
    col_er=np.nonzero(er[:,max(0,sx-3):sx+4].any(axis=1))[0]; ap_er=(col_er.max()-col_er.min()) if len(col_er) else 0   # lid aperture (eye region) at the pupil column
    out['aperture']=float(ap_er)
    if (ap_median and ap_er<0.40*ap_median) or wd/max(ap,1)>2.8 or (ap<25 and wd/max(ap,1)>2.0): out['flags'].append('eye closed / lid slit (visible core too flat to extrapolate)'); out['confidence']='none'
    e=None
    if len(pts)>=40:
        e=cv2.fitEllipse(pts.astype(np.float32))
        for it in range(2):
            X,Y=ellipse_pts(e); d=np.sqrt((pts[:,0,None]-X[None])**2+(pts[:,1,None]-Y[None])**2).min(axis=1)
            p2=pts[d<max(6,np.percentile(d,80))]
            if len(p2)<40: break
            e=cv2.fitEllipse(p2.astype(np.float32)); pts=p2
        X,Y=ellipse_pts(e); d=np.sqrt((pts[:,0,None]-X[None])**2+(pts[:,1,None]-Y[None])**2).min(axis=1); res=float(np.median(d))
        ang=np.rad2deg(np.arctan2(-(pts[:,1]-e[0][1]),pts[:,0]-e[0][0]))%360; cover=len(np.unique((ang//15).astype(int)))/24
        out.update(res=res,cover=cover,e_free=e)
    # 5. accept or fall back to the fixed-aspect ellipse
    mode='free'
    if e is not None:
        ext=extremes(e); asp=ext['W']/max(ext['H'],1)
        bad=(asp>1.9 or asp<0.75) or (lid_top>0.4 and lid_bot>0.4) or cover<0.45 or out['weak_frac']>0.5
        if bad: mode='fixed'
    else: mode='fixed'
    if mode=='fixed':
        # width from the horizontal extent of the core mask (left tip to right tip; right tip may be under the glint: use the free fit's right if it exists and lies inside a glint)
        mp=segment(T_PLATEAU); mp=mp if mp is not None else mm
        pys,pxs=np.nonzero(mp); L=float(pxs.min()); R=float(pxs.max()); ys=pys
        if e is not None:
            ext=extremes(e); rxp,ryp=ext['pupil_right']
            if 0<=int(ryp)<H and 0<=int(rxp)<W and gm[int(ryp),int(rxp)]: R=max(R,rxp)
        Wd=R-L; Hd=Wd/prior['aspect']; cy=(ys.min()+ys.max())/2; cxm=(L+R)/2
        e=((cxm,cy),(Wd,Hd),prior.get('tilt',0.0)); out['flags'].append('fixed-aspect ellipse (top/bottom hidden or shape implausible)')
    out['e']=e; out['mode']=mode; out.update(extremes(e))
    # 6. lids and corners from the eye region on the column through the pupil centre / extreme x
    cxp=int(round(e[0][0])); col=er[:,max(0,cxp-3):cxp+4].any(axis=1); ycol=np.nonzero(col)[0]
    if len(ycol): out['eyelid_top']=(float(cxp),float(ycol.min())); out['eyelid_bottom']=(float(cxp),float(ycol.max()))
    if len(exs):
        i=np.argmin(exs); j=np.argmax(exs); out['eye_nasal_corner']=(float(exs[i]),float(np.median(eys[exs==exs[i]]))); out['eye_temporal_corner']=(float(exs[j]),float(np.median(eys[exs==exs[j]])))
    # 7. confidence
    conf='high'
    if mode=='fixed': conf='low'
    elif out.get('res',9)>4 or out.get('cover',0)<0.6: conf='medium'
    if 'mask touches search box (leak)' in out['flags']: conf='low'
    if any(f.startswith('eye closed') for f in out['flags']): conf='none'
    out['confidence']=conf; return out
