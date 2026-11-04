import argparse
import json
from pathlib import Path

import cv2

from publico import ContadorPublico, desenhar, ler_regiao

EXTENSOES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def listar_imagens(caminhos):
    for caminho in map(Path, caminhos):
        if caminho.is_dir():
            yield from sorted(p for p in caminho.iterdir() if p.suffix.lower() in EXTENSOES)
        else:
            yield caminho


def main():
    parser = argparse.ArgumentParser(description="Conta o público em fotos de auditório")
    parser.add_argument("imagens", nargs="+", help="arquivos ou pastas com fotos")
    parser.add_argument("--modelo", default="yolo26m.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--bloco", type=int, default=320, help="tamanho do bloco em pixels; 0 desativa")
    parser.add_argument("--sobreposicao", type=float, default=0.3)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--regiao", help="polígono x1,y1,x2,y2,x3,y3,... para contar só dentro dele")
    parser.add_argument("--calor", action="store_true", help="desenha mapa de calor da densidade")
    parser.add_argument("--sem-cadeiras", action="store_true")
    parser.add_argument("--sem-densidade", action="store_true", help="usa só o YOLO, sem o P2PNet")
    parser.add_argument("--limiar-cabeca", type=float, default=0.5, help="confiança mínima dos pontos do P2PNet")
    parser.add_argument("--escala-cabeca", type=float, default=1.0, help="aumente para cabeças muito pequenas")
    parser.add_argument("--saida", default="saida")
    parser.add_argument("--mostrar", action="store_true")
    args = parser.parse_args()

    contador = ContadorPublico(args.modelo, conf=args.conf, tamanho_bloco=args.bloco, sobreposicao=args.sobreposicao, imgsz=args.imgsz, cadeiras=not args.sem_cadeiras, densidade=not args.sem_densidade, limiar_cabeca=args.limiar_cabeca, escala_cabeca=args.escala_cabeca)
    regiao = ler_regiao(args.regiao)
    pasta = Path(args.saida)
    pasta.mkdir(parents=True, exist_ok=True)

    resumo = []
    for caminho in listar_imagens(args.imagens):
        imagem = cv2.imread(str(caminho))
        if imagem is None:
            print(f"{caminho}: não consegui abrir")
            continue

        resultado = contador.detectar(imagem, regiao)
        anotada = desenhar(imagem, resultado, regiao, args.calor, not args.sem_cadeiras)
        destino = pasta / f"{caminho.stem}_contagem.jpg"
        cv2.imwrite(str(destino), anotada)

        livres = len(resultado.cadeiras_livres)
        resumo.append({"imagem": str(caminho), "pessoas": resultado.total, "cadeiras_livres": livres})
        print(f"{caminho.name}: {resultado.total} pessoas, {livres} cadeiras livres -> {destino}")

        if args.mostrar:
            cv2.namedWindow("Publico", cv2.WINDOW_NORMAL)
            cv2.imshow("Publico", anotada)
            if cv2.waitKey(0) & 0xFF in (ord("q"), 27):
                break

    (pasta / "resumo.json").write_text(json.dumps(resumo, ensure_ascii=False, indent=2))
    if resumo:
        print(f"Total: {sum(r['pessoas'] for r in resumo)} pessoas em {len(resumo)} imagens")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
