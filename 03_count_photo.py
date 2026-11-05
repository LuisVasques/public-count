import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from audience import add_arguments, available_modes, create_counter, draw, parse_region

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(paths):
    for path in map(Path, paths):
        if path.is_dir():
            yield from sorted(p for p in path.iterdir() if p.suffix.lower() in EXTENSIONS)
        else:
            yield path


def main():
    parser = argparse.ArgumentParser(description="Conta o público em fotos de auditório")
    parser.add_argument("images", nargs="+", help="arquivos ou pastas com fotos")
    add_arguments(parser)
    parser.add_argument("--compare", action="store_true", help="roda todos os modos e salva lado a lado")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    modes = available_modes(args.head_model) if args.compare else [args.mode]
    counters = {m: create_counter(args, m) for m in modes}
    region = parse_region(args.region)
    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=True)

    summary = []
    for path in list_images(args.images):
        image = cv2.imread(str(path))
        if image is None:
            print(f"{path}: não consegui abrir")
            continue

        panels = []
        for mode, counter in counters.items():
            result = counter.detect(image, region)
            annotated = draw(image, result, region, args.heatmap, not args.no_chairs)
            if len(counters) > 1:
                cv2.putText(annotated, mode, (10, annotated.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3, cv2.LINE_AA)
            panels.append(annotated)
            free = len(result.free_chairs)
            summary.append({"image": str(path), "mode": mode, "people": result.total, "free_chairs": free})
            print(f"{path.name} [{mode}]: {result.total} pessoas, {free} cadeiras livres")

        annotated = np.hstack(panels)
        destination = folder / f"{path.stem}_count.jpg"
        cv2.imwrite(str(destination), annotated)

        if args.show:
            cv2.namedWindow("Audience", cv2.WINDOW_NORMAL)
            cv2.imshow("Audience", annotated)
            if cv2.waitKey(0) & 0xFF in (ord("q"), 27):
                break

    (folder / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    for mode in counters:
        print(f"Total [{mode}]: {sum(r['people'] for r in summary if r['mode'] == mode)} pessoas")
    print(f"Imagens salvas em {folder}/")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
