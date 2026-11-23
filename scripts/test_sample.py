import json, os, urllib.request, sys

outdir = "/workspace/edge-project/data"
os.makedirs(outdir, exist_ok=True)
img_path = os.path.join(outdir, "sample.jpg")

try:
    api = ("https://commons.wikimedia.org/w/api.php?action=query&generator=search"
           "&gsrsearch=cement%20bags%20construction&gsrlimit=3&prop=imageinfo"
           "&iiprop=url&iiurlwidth=1024&format=json")
    req = urllib.request.Request(api, headers={"User-Agent": "edge-project-test/1.0"})
    d = json.loads(urllib.request.urlopen(req, timeout=30).read())
    pages = list(d["query"]["pages"].values())
    url = pages[0]["imageinfo"][0]["thumburl"]
    req2 = urllib.request.Request(url, headers={"User-Agent": "edge-project-test/1.0"})
    with open(img_path, "wb") as f:
        f.write(urllib.request.urlopen(req2, timeout=30).read())
    print("이미지 다운로드 OK")
except Exception as e:
    print("다운로드 실패, 합성 이미지 생성:", e)
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (800, 600), (205, 195, 185))
    dr = ImageDraw.Draw(im)
    for x, y, w, h, c in [(80,120,180,140,(180,150,110)), (300,150,170,130,(150,150,150)), (520,100,200,160,(170,160,140))]:
        dr.rectangle([x, y, x+w, y+h], fill=c, outline=(60,50,40), width=6)
    im.save(img_path)

from ultralytics import YOLO
m = YOLO("yolov8n.pt")
r = m.predict(img_path, conf=0.25, verbose=False)[0]
print("YOLOv8 검출:", [(m.names[int(c)], round(float(s), 2)) for c, s in zip(r.boxes.cls, r.boxes.conf)] or "없음")

import torch
from PIL import Image
from torchvision import transforms
from transformers import Dinov2Model
b = Dinov2Model.from_pretrained("facebook/dinov2-small", cache_dir="/workspace/.hf-cache").cuda().half().eval()
tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                         transforms.Normalize([0.485]*3, [0.229]*3)])
x = tf(Image.open(img_path).convert("RGB")).unsqueeze(0).cuda().half()
with torch.inference_mode():
    o = b(x)
print("DINOv2 임베딩:", tuple(o.last_hidden_state.shape), "(384차원, CLS+256패치)")
