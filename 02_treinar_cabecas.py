import argparse
import json
import random
import shutil
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import cv2
from ultralytics import YOLO

URL = "https://huggingface.co/datasets/ShantyCam/crowdhuman/resolve/main/"
PASTA = Path("/home/luis/projetos/datasets/crowdhuman-cabecas")


def baixar(nome, destino):
    arquivo = destino / nome
    if arquivo.exists():
        return arquivo
    destino.mkdir(parents=True, exist_ok=True)
    print(f"Baixando {nome}...")
    temporario = arquivo.with_suffix(arquivo.suffix + ".parcial")
    urlretrieve(URL + nome, temporario)
    temporario.rename(arquivo)
    return arquivo


def extrair(arquivo, destino):
    marcador = destino / f".{arquivo.stem}.ok"
    if marcador.exists():
        return
    print(f"Extraindo {arquivo.name}...")
    destino.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(arquivo) as z:
        for membro in z.infolist():
            if membro.filename.lower().endswith(".jpg"):
                with z.open(membro) as origem, open(destino / Path(membro.filename).name, "wb") as saida:
                    shutil.copyfileobj(origem, saida)
    marcador.touch()


def converter(anotacoes, imagens, divisao, limite=None, semente=0):
    pasta_img = PASTA / "images" / divisao
    pasta_lbl = PASTA / "labels" / divisao
    pasta_img.mkdir(parents=True, exist_ok=True)
    pasta_lbl.mkdir(parents=True, exist_ok=True)

    linhas = [json.loads(l) for l in anotacoes.read_text().splitlines()]
    linhas = [l for l in linhas if (imagens / f"{l['ID']}.jpg").exists()]
    random.Random(semente).shuffle(linhas)
    if limite:
        linhas = linhas[:limite]

    total = 0
    for item in linhas:
        origem = imagens / f"{item['ID']}.jpg"
        altura, largura = cv2.imread(str(origem)).shape[:2]
        rotulos = []
        for caixa in item["gtboxes"]:
            if caixa.get("tag") != "person" or "hbox" not in caixa:
                continue
            if caixa.get("head_attr", {}).get("ignore", 0) or caixa.get("extra", {}).get("ignore", 0):
                continue
            x, y, w, h = caixa["hbox"]
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(largura, x + w), min(altura, y + h)
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            rotulos.append(f"0 {(x1 + x2) / 2 / largura:.6f} {(y1 + y2) / 2 / altura:.6f} {(x2 - x1) / largura:.6f} {(y2 - y1) / altura:.6f}")
        destino = pasta_img / origem.name
        if not destino.exists():
            destino.symlink_to(origem)
        (pasta_lbl / f"{origem.stem}.txt").write_text("\n".join(rotulos))
        total += len(rotulos)
    print(f"{divisao}: {len(linhas)} imagens, {total} cabeças")


def preparar(partes, limite_val):
    brutos = PASTA / "brutos"
    baixar("annotation_train.odgt", brutos)
    baixar("annotation_val.odgt", brutos)
    for parte in partes:
        extrair(baixar(f"CrowdHuman_{parte}.zip", brutos), brutos / "train")
    extrair(baixar("CrowdHuman_val.zip", brutos), brutos / "val")

    converter(brutos / "annotation_train.odgt", brutos / "train", "train")
    converter(brutos / "annotation_val.odgt", brutos / "val", "val", limite_val)

    yaml = PASTA / "cabecas.yaml"
    yaml.write_text(f"path: {PASTA}\ntrain: images/train\nval: images/val\nnames:\n  0: cabeca\n")
    return yaml


def main():
    parser = argparse.ArgumentParser(description="Treina um detector de cabeças YOLO no CrowdHuman")
    parser.add_argument("--modelo", default="yolo26s.pt")
    parser.add_argument("--partes", nargs="+", default=["train01"], choices=["train01", "train02", "train03"])
    parser.add_argument("--limite-val", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--so-preparar", action="store_true")
    args = parser.parse_args()

    yaml = preparar(args.partes, args.limite_val)
    if args.so_preparar:
        return

    model = YOLO(args.modelo)
    model.train(
        data=str(yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        max_det=1000,
        mosaic=1.0,
        scale=0.5,
        close_mosaic=5,
        patience=15,
        workers=8,
        project=str(Path("runs").resolve()),
        name="cabecas",
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
