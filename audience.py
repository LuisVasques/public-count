from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.ops import nms
from ultralytics import YOLO

PERSON = 0
CHAIR = 56


@dataclass
class Result:
    people: np.ndarray
    confs: np.ndarray
    chairs: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))
    heads: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))

    @property
    def total(self):
        return len(self.people) + len(self.heads)

    @property
    def centers(self):
        boxes = np.column_stack([(self.people[:, 0] + self.people[:, 2]) / 2, (self.people[:, 1] + self.people[:, 3]) / 2]) if len(self.people) else np.zeros((0, 2))
        return np.concatenate([boxes, self.heads.reshape(-1, 2)])

    @property
    def free_chairs(self):
        if len(self.chairs) == 0 or self.total == 0:
            return self.chairs
        return self.chairs[~occupied(self.chairs, self.people, self.heads)]


def occupied(chairs, people, heads):
    result = np.zeros(len(chairs), dtype=bool)
    if len(people):
        cx = (chairs[:, 0] + chairs[:, 2]) / 2
        cy = (chairs[:, 1] + chairs[:, 3]) / 2
        inside_x = (cx[:, None] >= people[None, :, 0]) & (cx[:, None] <= people[None, :, 2])
        inside_y = (cy[:, None] >= people[None, :, 1]) & (cy[:, None] <= people[None, :, 3])
        result |= (inside_x & inside_y).any(axis=1)
    if len(heads):
        height = chairs[:, 3] - chairs[:, 1]
        top = chairs[:, 1] - height
        inside_x = (heads[None, :, 0] >= chairs[:, None, 0]) & (heads[None, :, 0] <= chairs[:, None, 2])
        inside_y = (heads[None, :, 1] >= top[:, None]) & (heads[None, :, 1] <= chairs[:, None, 3])
        result |= (inside_x & inside_y).any(axis=1)
    return result


def tiles(width, height, size, overlap):
    step = max(1, int(size * (1 - overlap)))

    def starts(total):
        if total <= size:
            return [0]
        positions = list(range(0, total - size, step))
        positions.append(total - size)
        return positions

    return [(x, y, min(x + size, width), min(y + size, height)) for y in starts(height) for x in starts(width)]


def contained_in_another(boxes, confs, limit=0.8, any_size=False):
    if len(boxes) < 2:
        return np.zeros(len(boxes), dtype=bool)
    x1 = np.maximum(boxes[:, None, 0], boxes[None, :, 0])
    y1 = np.maximum(boxes[:, None, 1], boxes[None, :, 1])
    x2 = np.minimum(boxes[:, None, 2], boxes[None, :, 2])
    y2 = np.minimum(boxes[:, None, 3], boxes[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    fraction = inter / np.maximum(area[:, None], 1e-6)
    np.fill_diagonal(fraction, 0)
    if any_size:
        return ((fraction > limit) & (area[:, None] <= area[None, :])).any(axis=1)
    same_size = area[None, :] <= 3 * area[:, None]
    worse = confs[:, None] <= confs[None, :]
    return ((fraction > limit) & same_size & worse).any(axis=1)


def cut_by_tile(boxes, crop, width, height, margin=2, min_span=0.6):
    x1, y1, x2, y2 = crop
    left = (x1 > 0) & (boxes[:, 0] <= x1 + margin)
    top = (y1 > 0) & (boxes[:, 1] <= y1 + margin)
    right = (x2 < width) & (boxes[:, 2] >= x2 - margin)
    bottom = (y2 < height) & (boxes[:, 3] >= y2 - margin)
    wide = (boxes[:, 2] - boxes[:, 0]) >= min_span * (x2 - x1)
    tall = (boxes[:, 3] - boxes[:, 1]) >= min_span * (y2 - y1)
    return (left | right | top | bottom) & (wide | tall)


def inside_large_box(boxes, from_full, tile, limit=0.8):
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    large = boxes[from_full & (area >= tile * tile)]
    if len(large) == 0:
        return np.zeros(len(boxes), dtype=bool)
    width = np.clip(np.minimum(boxes[:, None, 2], large[None, :, 2]) - np.maximum(boxes[:, None, 0], large[None, :, 0]), 0, None)
    height = np.clip(np.minimum(boxes[:, None, 3], large[None, :, 3]) - np.maximum(boxes[:, None, 1], large[None, :, 1]), 0, None)
    return ~from_full & ((width * height / np.maximum(area[:, None], 1e-6)) > limit).any(axis=1)


def unmatched_points(points, boxes, head_fraction=0.45):
    if len(points) == 0 or len(boxes) == 0:
        return np.ones(len(points), dtype=bool)
    free = np.ones(len(points), dtype=bool)
    width = boxes[:, 2] - boxes[:, 0]
    height = boxes[:, 3] - boxes[:, 1]
    target_x = (boxes[:, 0] + boxes[:, 2]) / 2
    target_y = boxes[:, 1] + 0.12 * height
    for i in np.argsort(-(width * height)):
        x1, y1, x2, _ = boxes[i]
        inside = free & (points[:, 0] >= x1) & (points[:, 0] <= x2) & (points[:, 1] >= y1) & (points[:, 1] <= y1 + head_fraction * height[i])
        if inside.any():
            candidates = np.flatnonzero(inside)
            distance = np.hypot(points[candidates, 0] - target_x[i], points[candidates, 1] - target_y[i])
            free[candidates[np.argmin(distance)]] = False
    return free


class TiledDetector:
    def __init__(self, model, conf=0.25, iou=0.5, tile_size=320, overlap=0.3, imgsz=640, batch_size=16, classes=None, no_nesting=False):
        self.model = YOLO(model)
        self.conf = conf
        self.iou = iou
        self.tile_size = tile_size
        self.overlap = overlap
        self.imgsz = imgsz
        self.batch_size = batch_size
        self.classes = classes
        self.no_nesting = no_nesting

    def _predict(self, images):
        output = []
        for i in range(0, len(images), self.batch_size):
            output.extend(self.model(images[i:i + self.batch_size], conf=self.conf, iou=self.iou, imgsz=self.imgsz, classes=self.classes, max_det=1000, verbose=False))
        return output

    def detect(self, image):
        height, width = image.shape[:2]
        crops = [(0, 0, width, height)]
        tile = max(self.tile_size, max(height, width) // 4) if self.tile_size else 0
        if tile and max(height, width) > tile:
            crops += tiles(width, height, tile, self.overlap)

        boxes, confs, classes, from_full = [], [], [], []
        for index, (crop, r) in enumerate(zip(crops, self._predict([image[y1:y2, x1:x2] for x1, y1, x2, y2 in crops]))):
            if len(r.boxes) == 0:
                continue
            found = r.boxes.xyxy.cpu().numpy() + [crop[0], crop[1], crop[0], crop[1]]
            keep = np.ones(len(found), dtype=bool) if index == 0 else ~cut_by_tile(found, crop, width, height)
            boxes.append(found[keep])
            confs.append(r.boxes.conf.cpu().numpy()[keep])
            classes.append(r.boxes.cls.cpu().numpy().astype(int)[keep])
            from_full.append(np.full(keep.sum(), index == 0))

        boxes = np.concatenate(boxes) if boxes else np.zeros((0, 4))
        confs = np.concatenate(confs) if confs else np.zeros(0)
        classes = np.concatenate(classes) if classes else np.zeros(0, dtype=int)
        from_full = np.concatenate(from_full) if from_full else np.zeros(0, dtype=bool)

        by_class = {}
        for class_id in np.unique(classes):
            m = classes == class_id
            c, s, f = boxes[m], confs[m], from_full[m]
            keep = nms(torch.from_numpy(c).float(), torch.from_numpy(s).float(), self.iou).numpy()
            c, s, f = c[keep], s[keep], f[keep]
            keep = ~contained_in_another(c, s, any_size=self.no_nesting) & ~inside_large_box(c, f, tile)
            by_class[int(class_id)] = (c[keep], s[keep])
        return by_class


class YoloHeads:
    def __init__(self, model, conf=0.25, tile_size=320, overlap=0.3, imgsz=800):
        self.detector = TiledDetector(model, conf=conf, iou=0.4, tile_size=tile_size, overlap=overlap, imgsz=imgsz, no_nesting=True)

    def boxes(self, image):
        return self.detector.detect(image).get(0, (np.zeros((0, 4)), np.zeros(0)))

    def points(self, image):
        boxes, confs = self.boxes(image)
        return box_centers(boxes), confs


class CombinedHeads:
    def __init__(self, detector, density, margin=0.25):
        self.detector = detector
        self.density = density
        self.margin = margin

    def points(self, image):
        boxes, confs = self.detector.boxes(image)
        points, confs_p2p = self.density.points(image)
        if len(boxes) and len(points):
            w = (boxes[:, 2] - boxes[:, 0]) * self.margin
            h = (boxes[:, 3] - boxes[:, 1]) * self.margin
            inside_x = (points[:, None, 0] >= boxes[None, :, 0] - w) & (points[:, None, 0] <= boxes[None, :, 2] + w)
            inside_y = (points[:, None, 1] >= boxes[None, :, 1] - h) & (points[:, None, 1] <= boxes[None, :, 3] + h)
            new_points = ~(inside_x & inside_y).any(axis=1)
            points, confs_p2p = points[new_points], confs_p2p[new_points]
        return np.concatenate([box_centers(boxes), points.reshape(-1, 2)]), np.concatenate([confs, confs_p2p])


def box_centers(boxes):
    if len(boxes) == 0:
        return np.zeros((0, 2))
    return np.column_stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2])


def create_head_source(heads, threshold=None, scale=1.0, head_model=None):
    if not heads or heads == "none":
        return None
    if heads in ("p2pnet", "combined"):
        from p2pnet import DensityCounter
        density = DensityCounter(threshold=threshold or 0.5, scale=scale)
        if heads == "p2pnet":
            return density
        return CombinedHeads(YoloHeads(head_model), density)
    return YoloHeads(heads, conf=threshold or 0.25)


class AudienceCounter:
    def __init__(self, model="yolo26m.pt", conf=0.25, iou=0.5, tile_size=320, overlap=0.3, imgsz=640, batch_size=16, chairs=True, heads="p2pnet", head_threshold=None, head_scale=1.0, head_model="runs/heads/weights/best.pt"):
        self.detector = TiledDetector(model, conf, iou, tile_size, overlap, imgsz, batch_size, [PERSON, CHAIR] if chairs else [PERSON]) if model else None
        self.heads = create_head_source(heads, head_threshold, head_scale, head_model)

    def detect(self, image, region=None):
        empty = (np.zeros((0, 4)), np.zeros(0))
        by_class = self.detector.detect(image) if self.detector else {}
        people, person_confs = by_class.get(PERSON, empty)
        chairs, _ = by_class.get(CHAIR, empty)
        if region is not None:
            people, person_confs = filter_region(people, person_confs, region)
            chairs, _ = filter_region(chairs, np.zeros(len(chairs)), region)

        heads = np.zeros((0, 2))
        if self.heads is not None:
            heads, _ = self.heads.points(image)
            if region is not None and len(heads):
                heads = heads[inside_region(heads, region)]
            heads = heads[unmatched_points(heads, people)]
        return Result(people, person_confs, chairs, heads)


def inside_region(points, region):
    if len(points) == 0:
        return np.zeros(0, dtype=bool)
    polygon = np.asarray(region, dtype=np.float32)
    return np.array([cv2.pointPolygonTest(polygon, (float(x), float(y)), False) >= 0 for x, y in points], dtype=bool)


def filter_region(boxes, confs, region):
    centers = np.column_stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2])
    inside = inside_region(centers, region)
    return boxes[inside], confs[inside]


def heatmap_mask(shape, centers, radius, sigma=None):
    height, width = shape[:2]
    heat = np.zeros((height, width), dtype=np.float32)
    for x, y in centers.astype(int):
        cv2.circle(heat, (x, y), radius, 1.0, -1)
    sigma = sigma or max(width, height) / 60
    heat = cv2.GaussianBlur(heat, (0, 0), sigma)
    if heat.max() > 0:
        heat /= heat.max()
    return heat


def draw(image, result, region=None, heatmap=False, show_chairs=True):
    output = image.copy()
    height, width = output.shape[:2]
    scale = max(1.0, max(height, width) / 1280)

    if heatmap and result.total:
        heat = heatmap_mask(output.shape, result.centers, max(4, int(max(height, width) / 100)))
        colored = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
        alpha = (heat[..., None] * 0.6).clip(0, 0.6)
        output = (output * (1 - alpha) + colored * alpha).astype(np.uint8)

    if region is not None:
        cv2.polylines(output, [np.asarray(region, dtype=np.int32)], True, (255, 200, 0), max(1, int(2 * scale)), cv2.LINE_AA)

    free = result.free_chairs if show_chairs else np.zeros((0, 4))
    for x1, y1, x2, y2 in free.astype(int):
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 165, 255), max(1, int(scale)))

    for x1, y1, x2, y2 in result.people.astype(int):
        if not heatmap:
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), max(1, int(scale)))
        cv2.circle(output, ((x1 + x2) // 2, (y1 + y2) // 2), max(2, int(3 * scale)), (0, 0, 255), -1, cv2.LINE_AA)

    for x, y in result.heads.astype(int):
        cv2.circle(output, (x, y), max(3, int(4 * scale)), (255, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(output, (x, y), max(3, int(4 * scale)), (255, 255, 255), 1, cv2.LINE_AA)

    lines = [f"Pessoas: {result.total}"]
    if show_chairs and len(result.chairs):
        seats = result.total + len(free)
        lines.append(f"Cadeiras livres: {len(free)}")
        lines.append(f"Ocupacao: {100 * result.total / seats:.0f}%")

    font_scale = 0.9 * scale
    thickness = max(2, int(2 * scale))
    line_height = int(36 * scale)
    panel_width = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0][0] for t in lines) + int(20 * scale)
    cv2.rectangle(output, (0, 0), (panel_width, line_height * len(lines) + int(12 * scale)), (0, 0, 0), -1)
    for i, text in enumerate(lines):
        cv2.putText(output, text, (int(10 * scale), line_height * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return output


TRAINED_MODEL = Path("runs/heads/weights/best.pt")
MODES = ["combined", "trained", "p2pnet", "people"]
REQUIRES_TRAINING = {"combined", "trained"}


def default_mode():
    return "combined" if TRAINED_MODEL.exists() else "p2pnet"


def available_modes(head_model):
    return [m for m in MODES if m not in REQUIRES_TRAINING or Path(head_model).exists()]


def add_arguments(parser):
    parser.add_argument("--mode", choices=MODES, default=default_mode(), help="combined: pessoas + cabeças treinadas + P2PNet; trained: pessoas + cabeças treinadas; p2pnet: pessoas + P2PNet; people: só YOLO de pessoas")
    parser.add_argument("--model", default="yolo26m.pt", help="YOLO COCO usado para pessoas e cadeiras")
    parser.add_argument("--head-model", default=str(TRAINED_MODEL))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--tile", type=int, default=320, help="tamanho do bloco em pixels; 0 desativa")
    parser.add_argument("--overlap", type=float, default=0.3)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--head-threshold", type=float, help="confiança mínima das cabeças (padrão 0.5 no p2pnet, 0.25 no trained)")
    parser.add_argument("--head-scale", type=float, default=1.0, help="amplia a imagem antes do P2PNet")
    parser.add_argument("--region", help="polígono x1,y1,x2,y2,x3,y3,... para contar só dentro dele")
    parser.add_argument("--heatmap", action="store_true", help="desenha mapa de calor da densidade")
    parser.add_argument("--no-chairs", action="store_true")
    parser.add_argument("--output", default="output")


def create_counter(args, mode=None):
    mode = mode or args.mode
    heads = {"combined": "combined", "trained": args.head_model, "p2pnet": "p2pnet", "people": None}[mode]
    if mode in REQUIRES_TRAINING and not Path(args.head_model).exists():
        raise SystemExit(f"Modelo de cabeças {args.head_model} não encontrado. Rode 02_train_heads.py ou use --mode p2pnet.")
    return AudienceCounter(args.model, conf=args.conf, tile_size=args.tile, overlap=args.overlap, imgsz=args.imgsz, chairs=not args.no_chairs, heads=heads, head_threshold=args.head_threshold, head_scale=args.head_scale, head_model=args.head_model)


def parse_region(text):
    if not text:
        return None
    values = [float(v) for v in text.replace(";", ",").split(",")]
    if len(values) < 6 or len(values) % 2:
        raise ValueError("A região deve ter pelo menos 3 pontos: x1,y1,x2,y2,x3,y3,...")
    return np.array(values, dtype=np.float32).reshape(-1, 2)
