from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
from torchvision.ops import nms
from ultralytics import YOLO

PESSOA = 0
CADEIRA = 56


@dataclass
class Resultado:
    pessoas: np.ndarray
    confs: np.ndarray
    cadeiras: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))
    cabecas: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))

    @property
    def total(self):
        return len(self.pessoas) + len(self.cabecas)

    @property
    def centros(self):
        caixas = np.column_stack([(self.pessoas[:, 0] + self.pessoas[:, 2]) / 2, (self.pessoas[:, 1] + self.pessoas[:, 3]) / 2]) if len(self.pessoas) else np.zeros((0, 2))
        return np.concatenate([caixas, self.cabecas.reshape(-1, 2)])

    @property
    def cadeiras_livres(self):
        if len(self.cadeiras) == 0 or self.total == 0:
            return self.cadeiras
        return self.cadeiras[~ocupadas(self.cadeiras, self.pessoas, self.cabecas)]


def ocupadas(cadeiras, pessoas, cabecas):
    resultado = np.zeros(len(cadeiras), dtype=bool)
    if len(pessoas):
        cx = (cadeiras[:, 0] + cadeiras[:, 2]) / 2
        cy = (cadeiras[:, 1] + cadeiras[:, 3]) / 2
        dentro_x = (cx[:, None] >= pessoas[None, :, 0]) & (cx[:, None] <= pessoas[None, :, 2])
        dentro_y = (cy[:, None] >= pessoas[None, :, 1]) & (cy[:, None] <= pessoas[None, :, 3])
        resultado |= (dentro_x & dentro_y).any(axis=1)
    if len(cabecas):
        altura = cadeiras[:, 3] - cadeiras[:, 1]
        topo = cadeiras[:, 1] - altura
        dentro_x = (cabecas[None, :, 0] >= cadeiras[:, None, 0]) & (cabecas[None, :, 0] <= cadeiras[:, None, 2])
        dentro_y = (cabecas[None, :, 1] >= topo[:, None]) & (cabecas[None, :, 1] <= cadeiras[:, None, 3])
        resultado |= (dentro_x & dentro_y).any(axis=1)
    return resultado


def blocos(largura, altura, tamanho, sobreposicao):
    passo = max(1, int(tamanho * (1 - sobreposicao)))

    def inicios(total):
        if total <= tamanho:
            return [0]
        pos = list(range(0, total - tamanho, passo))
        pos.append(total - tamanho)
        return pos

    return [(x, y, min(x + tamanho, largura), min(y + tamanho, altura)) for y in inicios(altura) for x in inicios(largura)]


def contido_em_outro(caixas, confs, limite=0.8):
    if len(caixas) < 2:
        return np.zeros(len(caixas), dtype=bool)
    x1 = np.maximum(caixas[:, None, 0], caixas[None, :, 0])
    y1 = np.maximum(caixas[:, None, 1], caixas[None, :, 1])
    x2 = np.minimum(caixas[:, None, 2], caixas[None, :, 2])
    y2 = np.minimum(caixas[:, None, 3], caixas[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (caixas[:, 2] - caixas[:, 0]) * (caixas[:, 3] - caixas[:, 1])
    fracao = inter / np.maximum(area[:, None], 1e-6)
    np.fill_diagonal(fracao, 0)
    mesmo_tamanho = area[None, :] <= 3 * area[:, None]
    pior = confs[:, None] <= confs[None, :]
    return ((fracao > limite) & mesmo_tamanho & pior).any(axis=1)


def pontos_sem_caixa(pontos, caixas, fracao_cabeca=0.45):
    if len(pontos) == 0 or len(caixas) == 0:
        return np.ones(len(pontos), dtype=bool)
    livres = np.ones(len(pontos), dtype=bool)
    largura = caixas[:, 2] - caixas[:, 0]
    altura = caixas[:, 3] - caixas[:, 1]
    alvo_x = (caixas[:, 0] + caixas[:, 2]) / 2
    alvo_y = caixas[:, 1] + 0.12 * altura
    for i in np.argsort(-(largura * altura)):
        x1, y1, x2, _ = caixas[i]
        dentro = livres & (pontos[:, 0] >= x1) & (pontos[:, 0] <= x2) & (pontos[:, 1] >= y1) & (pontos[:, 1] <= y1 + fracao_cabeca * altura[i])
        if dentro.any():
            candidatos = np.flatnonzero(dentro)
            distancia = np.hypot(pontos[candidatos, 0] - alvo_x[i], pontos[candidatos, 1] - alvo_y[i])
            livres[candidatos[np.argmin(distancia)]] = False
    return livres


class ContadorPublico:
    def __init__(self, modelo="yolo26m.pt", conf=0.25, iou=0.5, tamanho_bloco=640, sobreposicao=0.25, imgsz=640, lote=16, cadeiras=True, densidade=True, limiar_cabeca=0.5, escala_cabeca=1.0):
        self.model = YOLO(modelo)
        self.conf = conf
        self.iou = iou
        self.tamanho_bloco = tamanho_bloco
        self.sobreposicao = sobreposicao
        self.imgsz = imgsz
        self.lote = lote
        self.coco = self.model.names.get(PESSOA) == "person"
        if self.coco:
            self.classes = [PESSOA, CADEIRA] if cadeiras else [PESSOA]
        else:
            self.classes = None
        self.densidade = None
        if densidade:
            from p2pnet import ContadorDensidade
            self.densidade = ContadorDensidade(limiar=limiar_cabeca, escala=escala_cabeca)

    def _prever(self, imagens):
        saida = []
        for i in range(0, len(imagens), self.lote):
            saida.extend(self.model(imagens[i:i + self.lote], conf=self.conf, iou=self.iou, imgsz=self.imgsz, classes=self.classes, max_det=1000, verbose=False))
        return saida

    def detectar(self, imagem, regiao=None):
        altura, largura = imagem.shape[:2]
        recortes = [(0, 0, largura, altura)]
        if self.tamanho_bloco and max(altura, largura) > self.tamanho_bloco:
            recortes += blocos(largura, altura, self.tamanho_bloco, self.sobreposicao)

        caixas, confs, classes = [], [], []
        for (x1, y1, x2, y2), r in zip(recortes, self._prever([imagem[y1:y2, x1:x2] for x1, y1, x2, y2 in recortes])):
            if len(r.boxes) == 0:
                continue
            caixas.append(r.boxes.xyxy.cpu().numpy() + [x1, y1, x1, y1])
            confs.append(r.boxes.conf.cpu().numpy())
            cls = r.boxes.cls.cpu().numpy().astype(int)
            classes.append(cls if self.coco else np.full_like(cls, PESSOA))

        caixas = np.concatenate(caixas) if caixas else np.zeros((0, 4))
        confs = np.concatenate(confs) if confs else np.zeros(0)
        classes = np.concatenate(classes) if classes else np.zeros(0, dtype=int)

        separados = {}
        for classe in (PESSOA, CADEIRA):
            m = classes == classe
            c, s = caixas[m], confs[m]
            if len(c):
                manter = nms(torch.from_numpy(c).float(), torch.from_numpy(s).float(), self.iou).numpy()
                c, s = c[manter], s[manter]
                manter = ~contido_em_outro(c, s)
                c, s = c[manter], s[manter]
            if regiao is not None and len(c):
                c, s = filtrar_regiao(c, s, regiao)
            separados[classe] = (c, s)

        pessoas, confs_pessoas = separados[PESSOA]
        cabecas = np.zeros((0, 2))
        if self.densidade is not None:
            cabecas, _ = self.densidade.pontos(imagem)
            if regiao is not None and len(cabecas):
                cabecas = cabecas[dentro_regiao(cabecas, regiao)]
            cabecas = cabecas[pontos_sem_caixa(cabecas, pessoas)]
        return Resultado(pessoas, confs_pessoas, separados[CADEIRA][0], cabecas)


def dentro_regiao(pontos, regiao):
    poligono = np.asarray(regiao, dtype=np.float32)
    return np.array([cv2.pointPolygonTest(poligono, (float(x), float(y)), False) >= 0 for x, y in pontos], dtype=bool)


def filtrar_regiao(caixas, confs, regiao):
    centros = np.column_stack([(caixas[:, 0] + caixas[:, 2]) / 2, (caixas[:, 1] + caixas[:, 3]) / 2])
    dentro = dentro_regiao(centros, regiao)
    return caixas[dentro], confs[dentro]


def mapa_calor(forma, centros, raio, sigma=None):
    altura, largura = forma[:2]
    mapa = np.zeros((altura, largura), dtype=np.float32)
    for x, y in centros.astype(int):
        cv2.circle(mapa, (x, y), raio, 1.0, -1)
    sigma = sigma or max(largura, altura) / 60
    mapa = cv2.GaussianBlur(mapa, (0, 0), sigma)
    if mapa.max() > 0:
        mapa /= mapa.max()
    return mapa


def desenhar(imagem, resultado, regiao=None, calor=False, mostrar_cadeiras=True):
    saida = imagem.copy()
    altura, largura = saida.shape[:2]
    escala = max(1.0, max(altura, largura) / 1280)

    if calor and resultado.total:
        mapa = mapa_calor(saida.shape, resultado.centros, max(4, int(max(altura, largura) / 100)))
        colorido = cv2.applyColorMap((mapa * 255).astype(np.uint8), cv2.COLORMAP_JET)
        alfa = (mapa[..., None] * 0.6).clip(0, 0.6)
        saida = (saida * (1 - alfa) + colorido * alfa).astype(np.uint8)

    if regiao is not None:
        cv2.polylines(saida, [np.asarray(regiao, dtype=np.int32)], True, (255, 200, 0), max(1, int(2 * escala)), cv2.LINE_AA)

    livres = resultado.cadeiras_livres if mostrar_cadeiras else np.zeros((0, 4))
    for x1, y1, x2, y2 in livres.astype(int):
        cv2.rectangle(saida, (x1, y1), (x2, y2), (0, 165, 255), max(1, int(escala)))

    for x1, y1, x2, y2 in resultado.pessoas.astype(int):
        if not calor:
            cv2.rectangle(saida, (x1, y1), (x2, y2), (0, 255, 0), max(1, int(escala)))
        cv2.circle(saida, ((x1 + x2) // 2, (y1 + y2) // 2), max(2, int(3 * escala)), (0, 0, 255), -1, cv2.LINE_AA)

    for x, y in resultado.cabecas.astype(int):
        cv2.circle(saida, (x, y), max(3, int(4 * escala)), (255, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(saida, (x, y), max(3, int(4 * escala)), (255, 255, 255), 1, cv2.LINE_AA)

    linhas = [f"Pessoas: {resultado.total}"]
    if mostrar_cadeiras and len(resultado.cadeiras):
        lugares = resultado.total + len(livres)
        linhas.append(f"Cadeiras livres: {len(livres)}")
        linhas.append(f"Ocupacao: {100 * resultado.total / lugares:.0f}%")

    tamanho_fonte = 0.9 * escala
    espessura = max(2, int(2 * escala))
    altura_linha = int(36 * escala)
    largura_painel = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, tamanho_fonte, espessura)[0][0] for t in linhas) + int(20 * escala)
    cv2.rectangle(saida, (0, 0), (largura_painel, altura_linha * len(linhas) + int(12 * escala)), (0, 0, 0), -1)
    for i, texto in enumerate(linhas):
        cv2.putText(saida, texto, (int(10 * escala), altura_linha * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, tamanho_fonte, (255, 255, 255), espessura, cv2.LINE_AA)
    return saida


def ler_regiao(texto):
    if not texto:
        return None
    valores = [float(v) for v in texto.replace(";", ",").split(",")]
    if len(valores) < 6 or len(valores) % 2:
        raise ValueError("A região deve ter pelo menos 3 pontos: x1,y1,x2,y2,x3,y3,...")
    return np.array(valores, dtype=np.float32).reshape(-1, 2)
