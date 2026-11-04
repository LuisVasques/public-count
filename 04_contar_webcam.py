import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from publico import adicionar_argumentos, modos_disponiveis, criar_contador, desenhar, ler_regiao


def abrir(fonte):
    return cv2.VideoCapture(int(fonte) if fonte.isdigit() else fonte)


def main():
    parser = argparse.ArgumentParser(description="Conta o público ao vivo por webcam, vídeo ou stream RTSP")
    parser.add_argument("--fonte", default="0", help="índice da câmera, arquivo de vídeo ou URL rtsp://")
    adicionar_argumentos(parser)
    parser.add_argument("--intervalo", type=float, default=0.5, help="segundos entre contagens")
    parser.add_argument("--media", type=int, default=5, help="quantas contagens usar na média móvel")
    parser.add_argument("--espelhar", action="store_true", help="espelha a imagem, útil na webcam do notebook")
    args = parser.parse_args()

    modos = modos_disponiveis(args.modelo_cabecas)
    modo = args.modo
    contadores = {modo: criar_contador(args, modo)}
    regiao = ler_regiao(args.regiao)
    cap = abrir(args.fonte)
    if not cap.isOpened():
        raise SystemExit(f"Não consegui abrir a fonte {args.fonte}.")

    historico = deque(maxlen=args.media)
    calor = args.calor
    anotada = None
    ultima = 0.0
    cv2.namedWindow("Publico", cv2.WINDOW_NORMAL)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.espelhar:
            frame = cv2.flip(frame, 1)

        agora = time.perf_counter()
        if anotada is None or agora - ultima >= args.intervalo:
            ultima = agora
            inicio = time.perf_counter()
            resultado = contadores[modo].detectar(frame, regiao)
            duracao = time.perf_counter() - inicio
            historico.append(resultado.total)
            anotada = desenhar(frame, resultado, regiao, calor, not args.sem_cadeiras)
            rodape = f"{modo} | media: {np.mean(historico):.0f} | {1000 * duracao:.0f} ms"
            cv2.putText(anotada, rodape, (10, anotada.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Publico", anotada)
        tecla = cv2.waitKey(1) & 0xFF
        if tecla in (ord("q"), 27):
            break
        if tecla == ord("c"):
            calor = not calor
            anotada = None
        if tecla == ord("m"):
            modo = modos[(modos.index(modo) + 1) % len(modos)]
            if modo not in contadores:
                contadores[modo] = criar_contador(args, modo)
            historico.clear()
            anotada = None
            print(f"Modo: {modo}")
        if tecla == ord("s"):
            pasta = Path(args.saida)
            pasta.mkdir(parents=True, exist_ok=True)
            destino = pasta / f"webcam_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(str(destino), anotada)
            print(f"Salvo {destino}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
