import argparse
import os
import re
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

from audience import add_arguments, available_modes, create_counter, draw, parse_region


def open_source(source):
    if source.isdigit():
        return cv2.VideoCapture(int(source))
    if is_network(source):
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
        return cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    return cv2.VideoCapture(source)


def is_network(source):
    return "://" in source


def hide_password(source):
    source = re.sub(r"/[0-9a-f]{32}/", "/***/", source)
    return re.sub(r"://([^:/@]+):[^@]+@", r"://\1:***@", source)


class LiveReader:
    def __init__(self, source, reconnect_delay=2.0):
        self.source = source
        self.reconnect_delay = reconnect_delay
        self.frame = None
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while self.running:
            cap = open_source(self.source)
            while self.running and cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    break
                with self.lock:
                    self.frame = frame
            cap.release()
            if self.running:
                print(f"Sem imagem de {hide_password(self.source)}, tentando de novo...")
                time.sleep(self.reconnect_delay)

    def read(self):
        with self.lock:
            frame, self.frame = self.frame, None
        return frame

    def release(self):
        self.running = False
        self.thread.join(timeout=5)


def main():
    parser = argparse.ArgumentParser(description="Conta o público ao vivo por webcam, vídeo ou stream RTSP")
    parser.add_argument("--source", default="0", help="índice da câmera, arquivo de vídeo, URL rtsp:// ou camera, para a URL salva pelo 05_find_camera.py")
    add_arguments(parser)
    parser.add_argument("--interval", type=float, default=0.5, help="segundos entre contagens")
    parser.add_argument("--average", type=int, default=5, help="quantas contagens usar na média móvel")
    parser.add_argument("--mirror", action="store_true", help="espelha a imagem, útil na webcam do notebook")
    args = parser.parse_args()
    if args.source == "camera":
        args.source = Path(".camera_url").read_text().strip()

    modes = available_modes(args.head_model)
    mode = args.mode
    counters = {mode: create_counter(args, mode)}
    region = parse_region(args.region)
    reader = LiveReader(args.source) if is_network(args.source) else None
    cap = None if reader else open_source(args.source)
    if cap is not None and not cap.isOpened():
        raise SystemExit(f"Não consegui abrir a fonte {args.source}.")

    history = deque(maxlen=args.average)
    heatmap = args.heatmap
    annotated = None
    last_run = 0.0
    cv2.namedWindow("Audience", cv2.WINDOW_NORMAL)
    try:
        while True:
            if reader:
                new_frame = reader.read()
                if new_frame is None:
                    if cv2.waitKey(10) & 0xFF in (ord("q"), 27):
                        break
                    continue
                frame = new_frame
            else:
                ok, frame = cap.read()
                if not ok:
                    break
            if args.mirror:
                frame = cv2.flip(frame, 1)

            now = time.perf_counter()
            if annotated is None or now - last_run >= args.interval:
                last_run = now
                start = time.perf_counter()
                try:
                    result = counters[mode].detect(frame, region)
                except torch.OutOfMemoryError:
                    if reader:
                        reader.release()
                    raise SystemExit("Sem memória na GPU. Veja se já há outra contagem aberta e feche antes de rodar esta.")
                duration = time.perf_counter() - start
                history.append(result.total)
                annotated = draw(frame, result, region, heatmap, not args.no_chairs)
                footer = f"{mode} | media: {np.mean(history):.0f} | {1000 * duration:.0f} ms"
                cv2.putText(annotated, footer, (10, annotated.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow("Audience", annotated)
            key = cv2.waitKey(1) & 0xFF
            if cv2.getWindowProperty("Audience", cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (ord("q"), 27):
                break
            if key == ord("c"):
                heatmap = not heatmap
                annotated = None
            if key == ord("m"):
                mode = modes[(modes.index(mode) + 1) % len(modes)]
                if mode not in counters:
                    counters[mode] = create_counter(args, mode)
                history.clear()
                annotated = None
                print(f"Modo: {mode}")
            if key == ord("s"):
                folder = Path(args.output)
                folder.mkdir(parents=True, exist_ok=True)
                destination = folder / f"webcam_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(str(destination), annotated)
                print(f"Salvo {destination}")
    except KeyboardInterrupt:
        pass

    if reader:
        reader.release()
    else:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
