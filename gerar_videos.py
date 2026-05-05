#gerar_videos.py

"""
Converte as imagens em camera/ num vídeo MP4.

Uso:
  python gerar_video.py --input .\dataset\trafego_01 --fps 20

Saída: <input>\video.mp4
"""

import argparse
from pathlib import Path
import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Pasta da run (que tem subpasta camera/)")
    parser.add_argument("--fps", type=int, default=20, help="FPS do vídeo (20 = velocidade real da simulação)")
    parser.add_argument("--output", default=None, help="Caminho do MP4 (default: <input>/video.mp4)")
    args = parser.parse_args()

    pasta = Path(args.input)
    pasta_cam = pasta / "camera"
    if not pasta_cam.exists():
        print(f"[ERRO] Pasta {pasta_cam} não existe")
        return

    # Lista todos os PNGs em ordem
    imagens = sorted(pasta_cam.glob("*.png"))
    if not imagens:
        print(f"[ERRO] Nenhum PNG encontrado em {pasta_cam}")
        return

    print(f"[INFO] {len(imagens)} imagens encontradas")

    # Pega dimensões da primeira imagem
    primeira = cv2.imread(str(imagens[0]))
    altura, largura = primeira.shape[:2]
    print(f"[INFO] Resolução: {largura}x{altura}, FPS: {args.fps}")

    saida = Path(args.output) if args.output else pasta / "video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(saida), fourcc, args.fps, (largura, altura))

    for i, img_path in enumerate(imagens):
        img = cv2.imread(str(img_path))
        writer.write(img)
        if i % 100 == 0:
            print(f"  processado {i}/{len(imagens)}")

    writer.release()
    print(f"[OK] Vídeo salvo em: {saida}")
    print(f"     Duração: {len(imagens) / args.fps:.1f}s")


if __name__ == "__main__":
    main()