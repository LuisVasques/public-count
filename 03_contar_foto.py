import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from publico import adicionar_argumentos, modos_disponiveis, criar_contador, desenhar, ler_regiao

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
    adicionar_argumentos(parser)
    parser.add_argument("--comparar", action="store_true", help="roda todos os modos e salva lado a lado")
    parser.add_argument("--mostrar", action="store_true")
    args = parser.parse_args()

    modos = modos_disponiveis(args.modelo_cabecas) if args.comparar else [args.modo]
    contadores = {m: criar_contador(args, m) for m in modos}
    regiao = ler_regiao(args.regiao)
    pasta = Path(args.saida)
    pasta.mkdir(parents=True, exist_ok=True)

    resumo = []
    for caminho in listar_imagens(args.imagens):
        imagem = cv2.imread(str(caminho))
        if imagem is None:
            print(f"{caminho}: não consegui abrir")
            continue

        paineis = []
        for modo, contador in contadores.items():
            resultado = contador.detectar(imagem, regiao)
            anotada = desenhar(imagem, resultado, regiao, args.calor, not args.sem_cadeiras)
            if len(contadores) > 1:
                cv2.putText(anotada, modo, (10, anotada.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3, cv2.LINE_AA)
            paineis.append(anotada)
            livres = len(resultado.cadeiras_livres)
            resumo.append({"imagem": str(caminho), "modo": modo, "pessoas": resultado.total, "cadeiras_livres": livres})
            print(f"{caminho.name} [{modo}]: {resultado.total} pessoas, {livres} cadeiras livres")

        anotada = np.hstack(paineis)
        destino = pasta / f"{caminho.stem}_contagem.jpg"
        cv2.imwrite(str(destino), anotada)

        if args.mostrar:
            cv2.namedWindow("Publico", cv2.WINDOW_NORMAL)
            cv2.imshow("Publico", anotada)
            if cv2.waitKey(0) & 0xFF in (ord("q"), 27):
                break

    (pasta / "resumo.json").write_text(json.dumps(resumo, ensure_ascii=False, indent=2))
    for modo in contadores:
        print(f"Total [{modo}]: {sum(r['pessoas'] for r in resumo if r['modo'] == modo)} pessoas")
    print(f"Imagens salvas em {pasta}/")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
