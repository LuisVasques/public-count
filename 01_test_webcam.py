import argparse
import time

import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0, help="índice da câmera (/dev/videoN)")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Não consegui abrir a câmera {args.camera}. Tente outro --camera.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Câmera {args.camera} aberta: {width}x{height}")

    previous = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Falha ao ler frame.")
            break

        frame = cv2.flip(frame, 1)

        now = time.perf_counter()
        fps = 1.0 / (now - previous)
        previous = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Webcam", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
