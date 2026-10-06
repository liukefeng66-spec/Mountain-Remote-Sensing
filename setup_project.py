import os
from pathlib import Path

base = Path(r'E:\PycharmProjects\website')
base.mkdir(parents=True, exist_ok=True)

dirs = [
    'backend/app/api',
    'backend/app/model_adapters',
    'backend/app/services',
    'backend/weights',
    'backend/data',
    'frontend/src/views',
    'frontend/src/assets'
]

for d in dirs:
    (base / d).mkdir(parents=True, exist_ok=True)

files = {
    'backend/requirements.txt': """fastapi==0.115.6
uvicorn[standard]==0.34.0
python-multipart==0.0.20
pydantic==2.10.4
numpy==2.1.3
Pillow==11.0.0
opencv-python==4.10.0.84
rasterio==1.4.3
shapely==2.0.6
geopandas==1.0.1
pyproj==3.7.0
pandas==2.2.3
openpyxl==3.1.5
torch>=2.2
onnxruntime-gpu>=1.20""",

    'backend/app/config.py': """from pathlib import Path
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
RESULT_DIR = DATA_DIR / "results"
EXPORT_DIR = DATA_DIR / "exports"
DB_PATH = DATA_DIR / "mountain_eye.db"
MODEL_DIR = BASE_DIR / "weights"
for d in (DATA_DIR, UPLOAD_DIR, RESULT_DIR, EXPORT_DIR, MODEL_DIR):
    d.mkdir(parents=True, exist_ok=True)
TILE_SIZE = 512
TILE_OVERLAP = 64
DEFAULT_THRESHOLD = 0.5
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}""",

    'backend/app/db.py': """import sqlite3
from datetime import datetime
from .config import DB_PATH
SCHEMA = '''CREATE TABLE IF NOT EXISTS feedback (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  job_id TEXT NOT NULL,
  patch_id TEXT,
  feedback_type TEXT NOT NULL,
  note TEXT,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS inspection_tasks (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  job_id TEXT NOT NULL,
  patch_id TEXT NOT NULL,
  risk_level TEXT NOT NULL,
  priority TEXT NOT NULL,
  centroid_x REAL,
  centroid_y REAL,
  area REAL,
  change_type TEXT,
  status TEXT DEFAULT '待核查',
  result TEXT,
  created_at TEXT NOT NULL
);'''
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn
def init_db():
    with get_conn() as conn:
        conn.executescript(SCHEMA)
def add_feedback(job_id, patch_id, feedback_type, note=""):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO feedback(job_id,patch_id,feedback_type,note,created_at) VALUES(?,?,?,?,?)",
            (job_id, patch_id, feedback_type, note, datetime.now().isoformat(timespec="seconds")),
        )
def add_task(task):
    with get_conn() as conn:
        conn.execute(
            '''INSERT INTO inspection_tasks
            (job_id,patch_id,risk_level,priority,centroid_x,centroid_y,area,change_type,status,created_at)
            VALUES(?,?,?,?,?,?,?,?,?,?)''',
            (task["job_id"], task["patch_id"], task["risk_level"], task["priority"],
             task.get("centroid_x"), task.get("centroid_y"), task.get("area"),
             task.get("change_type", "未分类"), "待核查", datetime.now().isoformat(timespec="seconds")),
        )
def list_tasks(job_id=None):
    with get_conn() as conn:
        if job_id:
            rows = conn.execute("SELECT * FROM inspection_tasks WHERE job_id=? ORDER BY id DESC", (job_id,)).fetchall()
        else:
            rows = conn.execute("SELECT * FROM inspection_tasks ORDER BY id DESC").fetchall()
    return [dict(r) for r in rows]""",

    'backend/app/main.py': """from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from .api.routes import router
from .config import DATA_DIR
from .db import init_db
app=FastAPI(title="山地慧眼 MountainEye API",version="1.0.0")
app.add_middleware(CORSMiddleware,allow_origins=["http://localhost:5173","http://127.0.0.1:5173"],allow_credentials=True,allow_methods=["*"],allow_headers=["*"])
app.include_router(router)
app.mount("/files",StaticFiles(directory=DATA_DIR),name="files")
@app.on_event("startup")
def startup(): init_db()""",

    'backend/app/api/routes.py': """from pathlib import Path
from uuid import uuid4
import json, shutil
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from ..config import UPLOAD_DIR, RESULT_DIR, EXPORT_DIR, ALLOWED_EXTENSIONS
from ..services.raster import read_image
from ..services.inference import infer_large, save_prediction
from ..services.vectorize import vectorize_binary
from ..services.export import export_geojson, export_csv, export_shapefile_zip
from ..services.risk import evaluate_patch
from ..db import add_feedback, add_task, list_tasks
router = APIRouter(prefix="/api")
JOBS = {}
def _save_upload(f: UploadFile, job_dir: Path, label: str):
    ext = Path(f.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS: raise HTTPException(400, f"{label} 文件格式不支持")
    p = job_dir / f"{label}{ext}"
    with p.open("wb") as out: shutil.copyfileobj(f.file, out)
    return p
@router.get("/health")
def health(): return {"status":"ok"}
@router.post("/detect")
def detect(t1: UploadFile=File(...), t2: UploadFile=File(...), model: str=Form("tidal"), threshold: float=Form(0.5)):
    if model not in {"tidal","fsd","both"}: raise HTTPException(400,"model 必须为 tidal/fsd/both")
    job_id = uuid4().hex[:12]; upload_dir=UPLOAD_DIR/job_id; result_dir=RESULT_DIR/job_id
    upload_dir.mkdir(parents=True, exist_ok=True); result_dir.mkdir(parents=True, exist_ok=True)
    p1,p2=_save_upload(t1,upload_dir,"t1"),_save_upload(t2,upload_dir,"t2")
    im1,g1=read_image(str(p1)); im2,g2=read_image(str(p2))
    if im1.shape[:2] != im2.shape[:2]: raise HTTPException(400,"T1/T2 尺寸不一致，请先配准")
    models=["tidal","fsd"] if model=="both" else [model]
    output={}
    for m in models:
        prob,binary=infer_large(im1,im2,m,threshold)
        prob_p,bin_p=save_prediction(result_dir,m,prob,binary)
        fc,stats=vectorize_binary(binary,g1)
        geo_p=result_dir/f"{m}_patches.geojson"; csv_p=result_dir/f"{m}_stats.csv"
        export_geojson(fc,geo_p); export_csv(stats,csv_p)
        output[m]={"prob_url":f"/files/results/{job_id}/{prob_p.name}","binary_url":f"/files/results/{job_id}/{bin_p.name}",
                   "geojson_url":f"/files/results/{job_id}/{geo_p.name}","stats_url":f"/files/results/{job_id}/{csv_p.name}",
                   "patch_count":len(stats),"total_area_px":round(sum(x["area_px"] for x in stats),2),"stats":stats}
    JOBS[job_id]={"t1":str(p1),"t2":str(p2),"outputs":output}
    return {"job_id":job_id,"width":im1.shape[1],"height":im1.shape[0],"outputs":output}
class RiskRequest(BaseModel):
    job_id: str
    patch_id: str
    factor_scores: dict[str,float] = Field(default_factory=dict)
    current_mm: float = 0
    duration_h: float = 1
    history: list[float] = Field(default_factory=list)
    forecast_mm: float = 0
    area: float | None = None
    centroid_x: float | None = None
    centroid_y: float | None = None
    change_type: str = "未分类"
@router.post("/risk/evaluate")
def risk_eval(req: RiskRequest):
    result=evaluate_patch(req.factor_scores,req.current_mm,req.duration_h,req.history,req.forecast_mm)
    if result["risk_level"] in {"高","极高"}:
        add_task({"job_id":req.job_id,"patch_id":req.patch_id,"risk_level":result["risk_level"],"priority":result["action"],
                  "centroid_x":req.centroid_x,"centroid_y":req.centroid_y,"area":req.area,"change_type":req.change_type})
    return result
class FeedbackRequest(BaseModel):
    job_id:str; patch_id:str|None=None; feedback_type:str; note:str=""
@router.post("/feedback")
def feedback(req:FeedbackRequest):
    if req.feedback_type not in {"误报","漏报","确认正确"}: raise HTTPException(400,"反馈类型非法")
    add_feedback(req.job_id,req.patch_id,req.feedback_type,req.note); return {"ok":True}
@router.get("/tasks")
def tasks(job_id:str|None=None): return {"items":list_tasks(job_id)}
@router.get("/tasks/export")
def tasks_export(job_id:str|None=None):
    import pandas as pd
    items=list_tasks(job_id); p=EXPORT_DIR/f"tasks_{job_id or 'all'}.xlsx"
    pd.DataFrame(items).to_excel(p,index=False); return FileResponse(p,filename=p.name)""",

    'backend/app/model_adapters/base.py': """from abc import ABC, abstractmethod
import numpy as np
class ChangeModelAdapter(ABC):
    name = "base"
    @abstractmethod
    def predict(self, t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
        '''Return HxW probability map in [0,1].'''""",

    'backend/app/model_adapters/tidal.py': """from pathlib import Path
import numpy as np
import torch
from .base import ChangeModelAdapter
class TIDALAdapter(ChangeModelAdapter):
    name = "TIDAL-Net"
    def __init__(self, weight_path: str | None = None, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.weight_path = weight_path
        self.model = None
    def _prep(self, x):
        x = torch.from_numpy(x).float().permute(2,0,1).unsqueeze(0) / 255.0
        return x.to(self.device)
    @torch.inference_mode()
    def predict(self, t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
        if self.model is None:
            d = np.abs(t1.astype(np.float32)-t2.astype(np.float32)).mean(axis=2)/255.0
            return np.clip(d, 0, 1)
        y = self.model(self._prep(t1), self._prep(t2))
        if isinstance(y, (tuple, list)): y = y[-1]
        if isinstance(y, dict): y = y.get("out", next(iter(y.values())))
        y = torch.sigmoid(y) if y.min() < 0 or y.max() > 1 else y
        return y.squeeze().float().cpu().numpy()""",

    'backend/app/model_adapters/fsd.py': """import numpy as np
import torch
from .base import ChangeModelAdapter
class FSDAdapter(ChangeModelAdapter):
    name = "FSD-Net"
    def __init__(self, weight_path: str | None = None, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.weight_path = weight_path
        self.model = None
    @torch.inference_mode()
    def predict(self, t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
        if self.model is None:
            a = t1.astype(np.float32).mean(axis=2)
            b = t2.astype(np.float32).mean(axis=2)
            diff = np.abs(a-b)/255.0
            import cv2
            low = cv2.GaussianBlur(diff, (0,0), 3)
            return np.clip(0.65*diff + 0.35*low, 0, 1)
        x1 = torch.from_numpy(t1).float().permute(2,0,1).unsqueeze(0).to(self.device)/255.0
        x2 = torch.from_numpy(t2).float().permute(2,0,1).unsqueeze(0).to(self.device)/255.0
        y = self.model(x1, x2)
        if isinstance(y, (tuple,list)): y = y[-1]
        if isinstance(y, dict): y = y.get("out", next(iter(y.values())))
        y = torch.sigmoid(y) if y.min() < 0 or y.max() > 1 else y
        return y.squeeze().float().cpu().numpy()""",

    'backend/app/services/raster.py': """from pathlib import Path
import numpy as np
from PIL import Image
import rasterio
def read_image(path: str):
    p = Path(path)
    georef = None
    if p.suffix.lower() in {".tif", ".tiff"}:
        with rasterio.open(p) as src:
            arr = src.read()
            georef = {"transform": src.transform, "crs": src.crs, "width": src.width, "height": src.height}
        if arr.shape[0] >= 3:
            arr = np.moveaxis(arr[:3], 0, -1)
        else:
            arr = np.repeat(arr[0][..., None], 3, axis=-1)
        arr = arr.astype(np.float32)
        lo, hi = np.percentile(arr, (2, 98))
        arr = np.clip((arr - lo) / max(hi-lo, 1e-6), 0, 1)
        arr = (arr * 255).astype(np.uint8)
    else:
        arr = np.array(Image.open(p).convert("RGB"))
    return arr, georef
def save_gray(arr, path):
    arr = np.asarray(arr)
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255 if arr.max() <= 1.0 else arr, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)
def iter_tiles(img, tile_size=512, overlap=64):
    h, w = img.shape[:2]
    step = tile_size - overlap
    ys = list(range(0, max(h - tile_size, 0) + 1, step))
    xs = list(range(0, max(w - tile_size, 0) + 1, step))
    if not ys or ys[-1] != max(h - tile_size, 0): ys.append(max(h - tile_size, 0))
    if not xs or xs[-1] != max(w - tile_size, 0): xs.append(max(w - tile_size, 0))
    for y in sorted(set(ys)):
        for x in sorted(set(xs)):
            tile = img[y:y+tile_size, x:x+tile_size]
            pad_h, pad_w = tile_size-tile.shape[0], tile_size-tile.shape[1]
            if pad_h or pad_w:
                tile = np.pad(tile, ((0,pad_h),(0,pad_w),(0,0)), mode="reflect")
            yield x, y, tile
def merge_tiles(tile_outputs, out_shape, tile_size=512):
    h, w = out_shape
    canvas = np.zeros((h, w), np.float32)
    weight = np.zeros((h, w), np.float32)
    for x, y, pred in tile_outputs:
        ph, pw = min(tile_size, h-y), min(tile_size, w-x)
        canvas[y:y+ph, x:x+pw] += pred[:ph, :pw]
        weight[y:y+ph, x:x+pw] += 1
    return canvas / np.maximum(weight, 1)""",

    'backend/app/services/inference.py': """from pathlib import Path
import cv2
import numpy as np
from ..config import TILE_SIZE, TILE_OVERLAP, DEFAULT_THRESHOLD, RESULT_DIR, MODEL_DIR
from ..model_adapters.tidal import TIDALAdapter
from ..model_adapters.fsd import FSDAdapter
from .raster import iter_tiles, merge_tiles, save_gray
_MODELS = {}
def get_model(name):
    name = name.lower()
    if name not in _MODELS:
        if name == "tidal": _MODELS[name] = TIDALAdapter(str(MODEL_DIR / "tidal.pth"))
        elif name == "fsd": _MODELS[name] = FSDAdapter(str(MODEL_DIR / "fsd.pth"))
        else: raise ValueError(f"unknown model: {name}")
    return _MODELS[name]
def infer_large(t1, t2, model_name, threshold=DEFAULT_THRESHOLD):
    if t1.shape[:2] != t2.shape[:2]:
        raise ValueError("T1 与 T2 尺寸必须一致；正式项目建议先做严格配准。")
    model = get_model(model_name)
    h, w = t1.shape[:2]
    outs = []
    tiles2 = {(x,y):tile for x,y,tile in iter_tiles(t2, TILE_SIZE, TILE_OVERLAP)}
    for x, y, tile1 in iter_tiles(t1, TILE_SIZE, TILE_OVERLAP):
        pred = model.predict(tile1, tiles2[(x,y)])
        if pred.shape != (TILE_SIZE, TILE_SIZE):
            pred = cv2.resize(pred.astype(np.float32), (TILE_SIZE,TILE_SIZE), interpolation=cv2.INTER_LINEAR)
        outs.append((x,y,pred.astype(np.float32)))
    prob = merge_tiles(outs, (h,w), TILE_SIZE)
    binary = (prob >= threshold).astype(np.uint8) * 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))
    return prob, binary
def save_prediction(job_dir: Path, model_name, prob, binary):
    job_dir.mkdir(parents=True, exist_ok=True)
    prob_path = job_dir / f"{model_name}_prob.png"
    bin_path = job_dir / f"{model_name}_binary.png"
    save_gray(prob, prob_path)
    save_gray(binary, bin_path)
    return prob_path, bin_path""",

    'backend/app/services/vectorize.py': """import cv2
import numpy as np
from shapely.geometry import Polygon, mapping
from rasterio.transform import xy
def pixel_to_world(transform, px, py):
    if transform is None: return float(px), float(py)
    x, y = xy(transform, py, px, offset="center")
    return float(x), float(y)
def vectorize_binary(binary, georef=None, min_area_px=20):
    contours, _ = cv2.findContours((binary>0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    features, stats = [], []
    transform = georef.get("transform") if georef else None
    for i, c in enumerate(contours, start=1):
        area_px = cv2.contourArea(c)
        if area_px < min_area_px: continue
        perimeter_px = cv2.arcLength(c, True)
        m = cv2.moments(c)
        cx = m["m10"]/m["m00"] if m["m00"] else 0
        cy = m["m01"]/m["m00"] if m["m00"] else 0
        pts = [pixel_to_world(transform, int(p[0][0]), int(p[0][1])) for p in c]
        if len(pts) < 3: continue
        poly = Polygon(pts)
        if not poly.is_valid: poly = poly.buffer(0)
        wx, wy = pixel_to_world(transform, cx, cy)
        props = {"patch_id": f"P{i:04d}", "area_px": float(area_px), "perimeter_px": float(perimeter_px),
                 "centroid_x": wx, "centroid_y": wy}
        features.append({"type":"Feature", "geometry":mapping(poly), "properties":props})
        stats.append(props)
    return {"type":"FeatureCollection", "features":features}, stats""",

    'backend/app/services/risk.py': """from dataclasses import dataclass
from math import log
DEFAULT_WEIGHTS = {
    "slope": 0.22, "relief": 0.12, "lithology": 0.16, "fault_distance": 0.10,
    "road_distance": 0.10, "landuse": 0.10, "vegetation": 0.08, "mean_rain": 0.12,
}
def information_value(ni, n, si, s, eps=1e-9):
    return log(((ni+eps)/(n+eps))/((si+eps)/(s+eps)))
def weighted_susceptibility(factor_scores: dict, weights=None):
    weights = weights or DEFAULT_WEIGHTS
    score = sum(float(factor_scores.get(k, 0.0))*w for k,w in weights.items()) / max(sum(weights.values()), 1e-9)
    if score >= 0.75: level = "高易发"
    elif score >= 0.50: level = "中易发"
    elif score >= 0.25: level = "低易发"
    else: level = "非易发"
    return round(score,4), level
def antecedent_rainfall(history, k=0.84, n=6):
    vals = list(history)[:n]
    return sum((k**i)*float(r) for i,r in enumerate(vals, start=1))
def rainfall_warning(current_mm, duration_h, history, forecast_mm=0, id_a=20.0, id_b=0.5):
    duration_h = max(float(duration_h), 1.0)
    intensity = float(current_mm) / duration_h
    rp = antecedent_rainfall(history)
    critical = id_a * (duration_h ** (-id_b))
    trigger_ratio = intensity / max(critical, 1e-6)
    wetness = min((rp + float(forecast_mm)) / 150.0, 1.5)
    index = 0.65*trigger_ratio + 0.35*wetness
    if index >= 1.20: level = "红色"
    elif index >= 0.85: level = "橙色"
    elif index >= 0.55: level = "黄色"
    else: level = "无"
    return {"rain_index":round(index,4), "warning":level, "antecedent_rain":round(rp,2),
            "intensity":round(intensity,2), "critical_intensity":round(critical,2)}
RISK_MATRIX = {
    ("高易发","红色"):("极高","立即核查"),
    ("高易发","橙色"):("高","优先核查"), ("高易发","黄色"):("高","优先核查"),
    ("中易发","红色"):("高","优先核查"),
}
def combine_risk(sus_level, rain_level):
    if (sus_level, rain_level) in RISK_MATRIX: return RISK_MATRIX[(sus_level,rain_level)]
    if sus_level == "中易发" and rain_level in {"黄色","橙色","无"}: return ("中","常规核查")
    if sus_level in {"低易发","非易发"}: return ("低","暂不核查")
    if sus_level == "高易发": return ("中","常规核查")
    return ("中","常规核查")
def evaluate_patch(factor_scores, current_mm, duration_h, history, forecast_mm=0):
    sus_score, sus_level = weighted_susceptibility(factor_scores)
    rain = rainfall_warning(current_mm, duration_h, history, forecast_mm)
    overall, action = combine_risk(sus_level, rain["warning"])
    return {"susceptibility_score":sus_score, "susceptibility_level":sus_level, **rain,
            "risk_level":overall, "action":action,
            "disclaimer":"风险结果用于竞赛原型与辅助研判；I-D 参数和因子权重在真实部署前必须用当地历史灾害数据标定。"}""",

    'backend/app/services/export.py': """import json, shutil
from pathlib import Path
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape
def export_geojson(fc, path):
    Path(path).write_text(json.dumps(fc, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
def export_csv(stats, path):
    pd.DataFrame(stats).to_csv(path, index=False, encoding="utf-8-sig")
    return path
def export_shapefile_zip(fc, out_dir, base="patches"):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    geoms = [shape(f["geometry"]) for f in fc["features"]]
    props = [f["properties"] for f in fc["features"]]
    gdf = gpd.GeoDataFrame(props, geometry=geoms)
    shp_dir = out_dir / base
    shp_dir.mkdir(exist_ok=True)
    shp = shp_dir / f"{base}.shp"
    gdf.to_file(shp, driver="ESRI Shapefile", encoding="utf-8")
    archive = shutil.make_archive(str(out_dir / base), "zip", shp_dir)
    return archive""",

    'frontend/package.json': """{
  "name":"mountain-eye-frontend",
  "version":"1.0.0",
  "type":"module",
  "scripts":{
    "dev":"vite --host 0.0.0.0",
    "build":"vite build",
    "preview":"vite preview"
  },
  "dependencies":{
    "@element-plus/icons-vue":"^2.3.1",
    "axios":"^1.7.9",
    "element-plus":"^2.9.1",
    "leaflet":"^1.9.4",
    "vue":"^3.5.13",
    "vue-router":"^4.5.0"
  },
  "devDependencies":{
    "@vitejs/plugin-vue":"^5.2.1",
    "vite":"^6.0.5"
  }
}""",

    'frontend/vite.config.js': """import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
export default defineConfig({
  plugins:[vue()],
  server:{
    port:5173,
    proxy:{
      '/api':'http://127.0.0.1:8000',
      '/files':'http://127.0.0.1:8000'
    }
  }
})""",

    'frontend/index.html': """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1.0"/>
  <title>山地慧眼 · 智慧水土</title>
</head>
<body>
  <div id="app"></div>
  <script type="module" src="/src/main.js"></script>
</body>
</html>""",

    'frontend/src/main.js': """import { createApp } from 'vue'
import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'
import 'leaflet/dist/leaflet.css'
import './assets/main.css'
import App from './App.vue'
import router from './router'
createApp(App).use(router).use(ElementPlus).mount('#app')""",

    'frontend/src/router.js': """import { createRouter, createWebHistory } from 'vue-router'
import Dashboard from './views/Dashboard.vue'
import Detect from './views/Detect.vue'
import Risk from './views/Risk.vue'
import Tasks from './views/Tasks.vue'
import Feedback from './views/Feedback.vue'
export default createRouter({
  history:createWebHistory(),
  routes:[
    {path:'/',component:Dashboard},
    {path:'/detect',component:Detect},
    {path:'/risk',component:Risk},
    {path:'/tasks',component:Tasks},
    {path:'/feedback',component:Feedback}
  ]
})""",

    'frontend/src/App.vue': """<template>
<div class="shell">
  <aside class="sidebar">
    <div class="brand">
      <div class="logo">ME</div>
      <div><b>山地慧眼</b><small>智慧水土 · MountainEye</small></div>
    </div>
    <nav>
      <RouterLink to="/">态势总览</RouterLink>
      <RouterLink to="/detect">双域智能检测</RouterLink>
      <RouterLink to="/risk">风险预估</RouterLink>
      <RouterLink to="/tasks">核查任务</RouterLink>
      <RouterLink to="/feedback">反馈样本</RouterLink>
    </nav>
    <div class="side-foot">Challenge Cup Demo<br/>TIDAL-Net × FSD-Net</div>
  </aside>
  <main>
    <header>
      <div><strong>西南山地山洪地灾与水保遥感智能监测系统</strong><span>空天地采集 · 双域检测 · 图斑解析 · 风险预估 · 核查反馈</span></div>
      <el-tag type="success" effect="dark">系统在线</el-tag>
    </header>
    <section class="page">
      <RouterView/>
    </section>
  </main>
</div>
</template>""",

    'frontend/src/assets/main.css': """:root{font-family:Inter,"Microsoft YaHei",sans-serif;color:#18313a;background:#f3f7f6}
*{box-sizing:border-box}
body{margin:0}
.shell{display:flex;min-height:100vh}
.sidebar{width:238px;background:linear-gradient(180deg,#123c3d,#0b292f);color:#fff;padding:24px 18px;position:fixed;inset:0 auto 0 0}
.brand{display:flex;gap:12px;align-items:center;margin-bottom:32px}
.brand small{display:block;opacity:.65;margin-top:4px}
.logo{width:42px;height:42px;border-radius:12px;background:#79d6b4;color:#0a2c2e;display:grid;place-items:center;font-weight:900}
.sidebar nav{display:flex;flex-direction:column;gap:8px}
.sidebar a{color:#d8ece7;text-decoration:none;padding:12px 14px;border-radius:10px}
.sidebar a.router-link-active,.sidebar a:hover{background:rgba(121,214,180,.16);color:#fff}
.side-foot{position:absolute;bottom:22px;font-size:12px;line-height:1.7;opacity:.55}
main{margin-left:238px;flex:1}
header{height:74px;background:#fff;border-bottom:1px solid #e3eeeb;padding:14px 26px;display:flex;align-items:center;justify-content:space-between;position:sticky;top:0;z-index:20}
header span{display:block;font-size:12px;color:#71858a;margin-top:5px}
.page{padding:24px}
.hero{background:linear-gradient(120deg,#174d50,#2e7d73);border-radius:18px;color:#fff;padding:30px;box-shadow:0 14px 40px rgba(19,72,70,.16)}
.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-top:18px}
.card{background:#fff;border:1px solid #e4eeeb;border-radius:16px;padding:18px;box-shadow:0 8px 28px rgba(22,62,60,.05)}
.metric b{font-size:28px;display:block}
.muted{color:#71858a}
.section-title{margin:24px 0 12px;font-size:18px;font-weight:800}
.upload-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.result-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:14px}
.result-img{width:100%;max-height:380px;object-fit:contain;background:#0b1719;border-radius:12px}
.toolbar{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
.risk-form{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}
@media(max-width:1000px){.grid{grid-template-columns:repeat(2,1fr)}.risk-form{grid-template-columns:repeat(2,1fr)}}""",

    'frontend/src/views/Dashboard.vue': """<template>
  <div class="hero">
    <h1>山地慧眼 · 智慧水土</h1>
    <p>面向西南山地乡村的山洪地灾与水保遥感智能监测平台。把论文模型能力转化为“检测—解析—风险—核查”的可演示业务闭环。</p>
    <el-button type="success" size="large" @click="$router.push('/detect')">开始双时相检测</el-button>
  </div>
  <div class="grid">
    <div class="card metric"><span class="muted">核心模型</span><b>2</b><small>TIDAL-Net / FSD-Net</small></div>
    <div class="card metric"><span class="muted">业务主链</span><b>5</b><small>上传·推理·可视化·统计·导出</small></div>
    <div class="card metric"><span class="muted">风险维度</span><b>2</b><small>静态易发性 + 动态降雨</small></div>
    <div class="card metric"><span class="muted">GIS 输出</span><b>3</b><small>GeoJSON / Shapefile / CSV</small></div>
  </div>
  <div class="section-title">技术闭环</div>
  <div class="card">空天地采集 → 双域智能检测 → 图斑业务解析 → 风险动态评估 → 预警决策支持 → 现场核查反馈 → 人工迭代优化</div>
</template>""",

    'frontend/src/views/Detect.vue': """<script setup>
import {ref} from 'vue'; 
import axios from 'axios'; 
import {ElMessage} from 'element-plus'
const t1=ref(null),t2=ref(null),model=ref('both'),threshold=ref(0.5),loading=ref(false),res=ref(null)
const pick=(e,k)=>{ if(k===1)t1.value=e.target.files[0]; else t2.value=e.target.files[0] }
async function run(){
  if(!t1.value||!t2.value)return ElMessage.warning('请先选择 T1 和 T2');
  loading.value=true;
  const fd=new FormData();
  fd.append('t1',t1.value); fd.append('t2',t2.value);
  fd.append('model',model.value); fd.append('threshold',threshold.value);
  try{
    res.value=(await axios.post('/api/detect',fd)).data;
    localStorage.setItem('lastJob',JSON.stringify(res.value));
    ElMessage.success('推理完成')
  }catch(e){
    ElMessage.error(e.response?.data?.detail||e.message)
  }finally{
    loading.value=false
  }
}
</script>
<template>
  <div class="card">
    <div class="toolbar">
      <el-select v-model="model" style="width:180px">
        <el-option label="双模型对比" value="both"/>
        <el-option label="TIDAL-Net" value="tidal"/>
        <el-option label="FSD-Net" value="fsd"/>
      </el-select>
      <span>阈值</span>
      <el-slider v-model="threshold" :min="0.1" :max="0.9" :step="0.05" style="width:220px"/>
      <el-button type="primary" :loading="loading" @click="run">开始推理</el-button>
    </div>
    <div class="upload-grid" style="margin-top:18px">
      <div><b>时相 T1</b><input type="file" @change="e=>pick(e,1)"/></div>
      <div><b>时相 T2</b><input type="file" @change="e=>pick(e,2)"/></div>
    </div>
  </div>
  <template v-if="res">
    <div v-for="(o,name) in res.outputs" :key="name">
      <div class="section-title">{{name.toUpperCase()}} 结果 · {{o.patch_count}} 个图斑</div>
      <div class="result-grid">
        <div class="card"><b>变化概率图</b><img class="result-img" :src="o.prob_url"/></div>
        <div class="card"><b>二值变化图</b><img class="result-img" :src="o.binary_url"/></div>
      </div>
      <div class="card" style="margin-top:14px">
        <a :href="o.geojson_url" target="_blank">下载 GeoJSON</a>　<a :href="o.stats_url" target="_blank">下载统计 CSV</a>
        <el-table :data="o.stats" max-height="320" style="margin-top:12px">
          <el-table-column prop="patch_id" label="图斑"/>
          <el-table-column prop="area_px" label="面积(px²)"/>
          <el-table-column prop="perimeter_px" label="周长(px)"/>
          <el-table-column prop="centroid_x" label="质心X"/>
          <el-table-column prop="centroid_y" label="质心Y"/>
        </el-table>
      </div>
    </div>
  </template>
</template>""",

    'frontend/src/views/Risk.vue': """<script setup>
import {reactive,ref} from 'vue';
import axios from 'axios';
import {ElMessage} from 'element-plus'
const last=JSON.parse(localStorage.getItem('lastJob')||'null');
const result=ref(null);
const f=reactive({
  job_id:last?.job_id||'', patch_id:'P0001',
  slope:.6, relief:.5, lithology:.6, fault_distance:.4, road_distance:.4, landuse:.5, vegetation:.4, mean_rain:.6,
  current_mm:35, duration_h:3, history:'18,12,10,8,4,2', forecast_mm:25
})
async function calc(){
  const factor_scores={
    slope:f.slope, relief:f.relief, lithology:f.lithology, fault_distance:f.fault_distance,
    road_distance:f.road_distance, landuse:f.landuse, vegetation:f.vegetation, mean_rain:f.mean_rain
  };
  try{
    result.value=(await axios.post('/api/risk/evaluate',{
      ...f, factor_scores, history:f.history.split(',').map(Number)
    })).data
  }catch(e){
    ElMessage.error(e.message)
  }
}
</script>
<template>
  <div class="card">
    <h2>风险预估</h2>
    <p class="muted">静态易发性（AHP/信息量思想）+ 动态降雨触发（I-D 阈值）+ 变化图斑叠加。竞赛演示参数可调；真实部署需本地历史灾害样本标定。</p>
    <div class="risk-form">
      <el-input v-model="f.job_id" placeholder="Job ID"/>
      <el-input v-model="f.patch_id" placeholder="图斑编号"/>
      <el-input-number v-model="f.slope" :min="0" :max="1" :step=".1" placeholder="坡度"/>
      <el-input-number v-model="f.relief" :min="0" :max="1" :step=".1" placeholder="起伏度"/>
      <el-input-number v-model="f.lithology" :min="0" :max="1" :step=".1" placeholder="岩性"/>
      <el-input-number v-model="f.fault_distance" :min="0" :max="1" :step=".1" placeholder="距断层"/>
      <el-input-number v-model="f.road_distance" :min="0" :max="1" :step=".1" placeholder="距道路"/>
      <el-input-number v-model="f.landuse" :min="0" :max="1" :step=".1" placeholder="土地利用"/>
      <el-input-number v-model="f.vegetation" :min="0" :max="1" :step=".1" placeholder="植被"/>
      <el-input-number v-model="f.mean_rain" :min="0" :max="1" :step=".1" placeholder="年均降雨"/>
      <el-input-number v-model="f.current_mm" :min="0" placeholder="当前雨量"/>
      <el-input-number v-model="f.duration_h" :min="1" placeholder="持续时间"/>
      <el-input v-model="f.history" placeholder="前6日雨量,逗号分隔"/>
      <el-input-number v-model="f.forecast_mm" :min="0" placeholder="预报雨量"/>
    </div>
    <el-button type="danger" style="margin-top:16px" @click="calc">计算综合风险</el-button>
  </div>
  <div v-if="result" class="grid" style="margin-top:16px;">
    <div class="card metric"><span>静态易发性</span><b>{{result.susceptibility_level}}</b><small>指数 {{result.susceptibility_score}}</small></div>
    <div class="card metric"><span>降雨预警</span><b>{{result.warning}}</b><small>前期有效雨量 {{result.antecedent_rain}} mm</small></div>
    <div class="card metric"><span>综合风险</span><b>{{result.risk_level}}</b><small>{{result.action}}</small></div>
    <div class="card"><b>说明</b><p style="font-size:12px;">{{result.disclaimer}}</p></div>
  </div>
</template>""",

    'frontend/src/views/Tasks.vue': """<script setup>
import {ref,onMounted} from 'vue';
import axios from 'axios';
const rows=ref([]);
async function load(){
  rows.value=(await axios.get('/api/tasks')).data.items
}
onMounted(load)
</script>
<template>
  <div class="card">
    <div class="toolbar">
      <h2 style="margin-right:auto">核查任务</h2>
      <el-button @click="load">刷新</el-button>
      <el-button type="primary" @click="window.open('/api/tasks/export')">导出 Excel</el-button>
    </div>
    <el-table :data="rows">
      <el-table-column prop="patch_id" label="图斑"/>
      <el-table-column prop="risk_level" label="风险"/>
      <el-table-column prop="priority" label="优先级"/>
      <el-table-column prop="area" label="面积"/>
      <el-table-column prop="change_type" label="类型"/>
      <el-table-column prop="status" label="状态"/>
    </el-table>
  </div>
</template>""",

    'frontend/src/views/Feedback.vue': """<script setup>
import {reactive} from 'vue';
import axios from 'axios';
import {ElMessage} from 'element-plus';
const last=JSON.parse(localStorage.getItem('lastJob')||'null');
const f=reactive({
  job_id:last?.job_id||'',
  patch_id:'',
  feedback_type:'误报',
  note:''
});
async function send(){
  await axios.post('/api/feedback',f);
  ElMessage.success('反馈已保存；不会自动触发模型训练')
}
</script>
<template>
  <div class="card">
    <h2>反馈样本收集</h2>
    <p class="muted">仅记录误报、漏报和人工确认结果，供后续人工导出并离线重新训练；不自动微调模型。</p>
    <el-form label-width="90px" style="max-width:650px">
      <el-form-item label="任务ID"><el-input v-model="f.job_id"/></el-form-item>
      <el-form-item label="图斑ID"><el-input v-model="f.patch_id"/></el-form-item>
      <el-form-item label="反馈类型">
        <el-radio-group v-model="f.feedback_type">
          <el-radio-button label="误报"/>
          <el-radio-button label="漏报"/>
          <el-radio-button label="确认正确"/>
        </el-radio-group>
      </el-form-item>
      <el-form-item label="备注"><el-input v-model="f.note" type="textarea"/></el-form-item>
      <el-button type="primary" @click="send">保存反馈</el-button>
    </el-form>
  </div>
</template>"""
}

for path, content in files.items():
    p = base / path
    p.write_text(content, encoding='utf-8')

print("All files have been successfully created in E:\\PycharmProjects\\website!")
