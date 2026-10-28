from pathlib import Path
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(r'C:\Users\Siddharth Gupta\Desktop\CODES-ALL\3dg')
OUT=ROOT/'output'; OUT.mkdir(exist_ok=True)
W,H=2000,1150
im=Image.new('RGB',(W,H),'#F7F9FC'); d=ImageDraw.Draw(im)
font=r'C:\Windows\Fonts\arial.ttf'; bold=r'C:\Windows\Fonts\arialbd.ttf'
def F(size,strong=False): return ImageFont.truetype(bold if strong else font,size)
navy='#183653'; blue='#E8F0FA'; teal='#E8F5F3'; gold='#FFF4DD'; text='#243447'; muted='#536579'; border='#7290AE'
d.rounded_rectangle((32,28,W-32,H-28),radius=28,fill='white',outline='#C8D5E3',width=3)
d.text((W//2,66),'PanoGS  |  System Architecture',font=F(43,True),fill=navy,anchor='mm')
d.text((W//2,116),'Local panorama-to-Gaussian pipeline and optional hosted generation path',font=F(23),fill=muted,anchor='mm')

def wrap(text,font_obj,maxw):
    words=text.split(); lines=[]; line=''
    for word in words:
        candidate=(line+' '+word).strip()
        if d.textbbox((0,0),candidate,font=font_obj)[2] > maxw and line:
            lines.append(line); line=word
        else: line=candidate
    if line: lines.append(line)
    return lines
def card(x,y,w,h,title,details,fill=blue,tag=None):
    d.rounded_rectangle((x,y,x+w,y+h),radius=22,fill=fill,outline=border,width=3)
    if tag:
        d.rounded_rectangle((x+15,y+14,x+w-15,y+49),radius=12,fill=navy)
        d.text((x+w/2,y+31),tag,font=F(18,True),fill='white',anchor='mm')
        ty=y+67
    else: ty=y+26
    titlef=F(25,True)
    title_lines=wrap(title,titlef,w-34)
    yy=ty
    for line in title_lines:
        d.text((x+w/2,yy),line,font=titlef,fill=navy,anchor='mt'); yy+=32
    yy+=11
    bodyf=F(19)
    for item in details:
        lines=wrap('• '+item,bodyf,w-34)
        for line in lines:
            d.text((x+18,yy),line,font=bodyf,fill=text); yy+=25
        yy+=6
def arrow(x1,y1,x2,y2,color=navy):
    d.line((x1,y1,x2,y2),fill=color,width=6)
    import math
    a=math.atan2(y2-y1,x2-x1); length=19; wing=12
    p1=(x2-length*math.cos(a)+wing*math.sin(a),y2-length*math.sin(a)-wing*math.cos(a))
    p2=(x2-length*math.cos(a)-wing*math.sin(a),y2-length*math.sin(a)+wing*math.cos(a))
    d.polygon([(x2,y2),p1,p2],fill=color)

# Six-stage local pipeline.
y=205; h=435; w=282; gap=28; start=52
xs=[start+i*(w+gap) for i in range(6)]
cards=[
 ('Input panorama',['Equirectangular RGB or HDR image','Single capture viewpoint']),
 ('Interface and orchestration',['CLI or local Studio','Load config, select model and run reconstruction']),
 ('Depth inference',['Six overlapping perspective faces','Pretrained Depth Anything V2 Metric Indoor or MiDaS','Inference runs for this input']),
 ('Geometry and repair',['Align face depths and blend seams','Spherical rays and back-projection','Room priors, cleanup and optional inpainting']),
 ('Gaussian scene',['Initialize 3D Gaussian primitives','Optional per-scene Adam optimization','Does not retrain the depth model']),
 ('Export and inspect',['PLY, SPLAT or HDRSPLAT','PanoGS Three.js viewer','SuperSplat editor']),
]
fills=[blue,blue,teal,teal,gold,blue]
for x,(title,details),fill in zip(xs,cards,fills): card(x,y,w,h,title,details,fill)
for i in range(5): arrow(xs[i]+w+4,y+h//2,xs[i+1]-9,y+h//2)
d.rounded_rectangle((50,668,W-50,728),radius=15,fill='#EEF3F8',outline='#CBD7E2',width=2)
d.text((W//2,698),'Model weights load lazily and may download on first use. Local reconstruction does not require an API key.',font=F(21),fill=navy,anchor='mm')

# Separate hosted provider path, shown dashed to distinguish it from local inference.
d.text((65,780),'OPTIONAL HOSTED WORKFLOW',font=F(21,True),fill='#8A5B0A')
hy=818; hh=190
host_cards=[
 (70,370,'Input + credential',['Image preview','WLT_API_KEY / WORLDLABS_API_KEY']),
 (625,430,'World Labs Marble API',['Upload, submit generation job','Poll status and export scene']),
 (1260,650,'Hosted result',['Provider-generated Gaussian scene','PLY download and viewer handoff']),
]
for x,cw,title,details in host_cards: card(x,hy,cw,hh,title,details,fill='#FFF8EA')
arrow(455,hy+hh//2,610,hy+hh//2,color='#A77828'); arrow(1067,hy+hh//2,1245,hy+hh//2,color='#A77828')
d.text((W//2,1055),'The hosted path is separate from the local depth-and-geometry pipeline.',font=F(20),fill=muted,anchor='mm')
path=OUT/'PanoGS_Architecture_Diagram.png'; im.save(path,dpi=(180,180)); print(path)
