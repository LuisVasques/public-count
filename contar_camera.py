import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from publico import ContadorPublico, desenhar, ler_regiao


def abrir(fonte):
    return cv2.VideoCapture(int(fonte) if fonte.isdigit() else fonte)


def main():
    parser = argparse.ArgumentParser(description="Conta o público ao vivo por câmera, vídeo ou stream RTSP")
    parser.add_argument("--fonte", default="0", help="índice da câmera, arquivo de vídeo ou URL rtsp://")
    parser.add_argument("--modelo", default="yolo26m.pt")
    parser.add_argument("--conf", type=float, default=0.2)
    parser.add_argument("--bloco", type=int, default=320)
    parser.add_argument("--sobreposicao", type=float, default=0.25)
    parser.add_argument("--regiao", help="polígono x1,y1,x2,y2,x3,y3,... para contar só dentro dele")
    parser.add_argument("--intervalo", type=float, default=1.0, help="segundos entre contagens")
    parser.add_argument("--media", type=int, default=5, help="quantas contagens usar na média móvel")
    parser.add_argument("--saida", default="saida")
    args = parser.parse_args()

    contador = ContadorPublico(args.modelo, conf=args.conf, tamanho_bloco=args.bloco, sobreposicao=args.sobreposicao)
    regiao = ler_regiao(args.regiao)
    cap = abrir(args.fonte)
    if not cap.isOpened():
        raise SystemExit(f"Não consegui abrir a fonte {args.fonte}.")

    historico = deque(maxlen=args.media)
    calor = False
    anotada = None
    ultima = 0.0
    cv2.namedWindow("Publico", cv2.WINDOW_NORMAL)
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        agora = time.perf_counter()
        if anotada is None or agora - ultima >= args.intervalo:
            ultima = agora
            resultado = contador.detectar(frame, regiao)
            historico.append(resultado.total)
            anotada = desenhar(frame, resultado, regiao, calor)
            media = np.mean(historico)
            cv2.putText(anotada, f"Media: {media:.0f}", (10, anotada.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Publico", anotada)
        tecla = cv2.waitKey(1) & 0xFF
        if tecla in (ord("q"), 27):
            break
        if tecla == ord("c"):
            calor = not calor
            anotada = None
        if tecla == ord("s"):
            pasta = Path(args.saida)
            pasta.mkdir(parents=True, exist_ok=True)
            destino = pasta / f"camera_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(str(destino), anotada)
            print(f"Salvo {destino}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
