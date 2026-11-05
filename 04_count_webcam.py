import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from audience import add_arguments, available_modes, create_counter, draw, parse_region


def open_source(source):
    return cv2.VideoCapture(int(source) if source.isdigit() else source)


def main():
    parser = argparse.ArgumentParser(description="Conta o público ao vivo por webcam, vídeo ou stream RTSP")
    parser.add_argument("--source", default="0", help="índice da câmera, arquivo de vídeo ou URL rtsp://")
    add_arguments(parser)
    parser.add_argument("--interval", type=float, default=0.5, help="segundos entre contagens")
    parser.add_argument("--average", type=int, default=5, help="quantas contagens usar na média móvel")
    parser.add_argument("--mirror", action="store_true", help="espelha a imagem, útil na webcam do notebook")
    args = parser.parse_args()

    modes = available_modes(args.head_model)
    mode = args.mode
    counters = {mode: create_counter(args, mode)}
    region = parse_region(args.region)
    cap = open_source(args.source)
    if not cap.isOpened():
        raise SystemExit(f"Não consegui abrir a fonte {args.source}.")

    history = deque(maxlen=args.average)
    heatmap = args.heatmap
    annotated = None
    last_run = 0.0
    cv2.namedWindow("Audience", cv2.WINDOW_NORMAL)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.mirror:
            frame = cv2.flip(frame, 1)

        now = time.perf_counter()
        if annotated is None or now - last_run >= args.interval:
            last_run = now
            start = time.perf_counter()
            result = counters[mode].detect(frame, region)
            duration = time.perf_counter() - start
            history.append(result.total)
            annotated = draw(frame, result, region, heatmap, not args.no_chairs)
            footer = f"{mode} | media: {np.mean(history):.0f} | {1000 * duration:.0f} ms"
            cv2.putText(annotated, footer, (10, annotated.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Audience", annotated)
        key = cv2.waitKey(1) & 0xFF
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

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
