from pathlib import Path
from urllib.request import urlretrieve

import cv2
import numpy as np
import torch
from torch import nn
from torchvision.models import vgg16_bn

URL_PESOS = "https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet/raw/main/weights/SHTechA.pth"
MEDIA = np.array([0.485, 0.456, 0.406], dtype=np.float32)
DESVIO = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def cabeca(entrada, saida, tamanho=256):
    camadas = nn.Module()
    camadas.conv1 = nn.Conv2d(entrada, tamanho, 3, padding=1)
    camadas.conv2 = nn.Conv2d(tamanho, tamanho, 3, padding=1)
    camadas.conv3 = nn.Conv2d(tamanho, tamanho, 3, padding=1)
    camadas.conv4 = nn.Conv2d(tamanho, tamanho, 3, padding=1)
    camadas.output = nn.Conv2d(tamanho, saida, 3, padding=1)
    return camadas


def aplicar_cabeca(camadas, x):
    x = torch.relu(camadas.conv1(x))
    x = torch.relu(camadas.conv2(x))
    return camadas.output(x).permute(0, 2, 3, 1)


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        f = list(vgg16_bn(weights=None).features.children())
        self.body1 = nn.Sequential(*f[:13])
        self.body2 = nn.Sequential(*f[13:23])
        self.body3 = nn.Sequential(*f[23:33])
        self.body4 = nn.Sequential(*f[33:43])

    def forward(self, x):
        saidas = []
        for corpo in (self.body1, self.body2, self.body3, self.body4):
            x = corpo(x)
            saidas.append(x)
        return saidas


class FPN(nn.Module):
    def __init__(self, c3=256, c4=512, c5=512, tamanho=256):
        super().__init__()
        self.P5_1 = nn.Conv2d(c5, tamanho, 1)
        self.P5_2 = nn.Conv2d(tamanho, tamanho, 3, padding=1)
        self.P4_1 = nn.Conv2d(c4, tamanho, 1)
        self.P4_2 = nn.Conv2d(tamanho, tamanho, 3, padding=1)
        self.P3_1 = nn.Conv2d(c3, tamanho, 1)
        self.P3_2 = nn.Conv2d(tamanho, tamanho, 3, padding=1)

    def forward(self, c3, c4, c5):
        p5 = self.P5_1(c5)
        p4 = self.P4_1(c4) + nn.functional.interpolate(p5, scale_factor=2, mode="nearest")
        return self.P4_2(p4)


class P2PNet(nn.Module):
    def __init__(self, linhas=2, colunas=2):
        super().__init__()
        self.ancoras_por_celula = linhas * colunas
        self.linhas = linhas
        self.colunas = colunas
        self.backbone = Backbone()
        self.fpn = FPN()
        self.regression = cabeca(256, self.ancoras_por_celula * 2)
        self.classification = cabeca(256, self.ancoras_por_celula * 2)

    def ancoras(self, altura, largura, passo, dispositivo):
        dx = (torch.arange(1, self.colunas + 1, device=dispositivo) - 0.5) * passo / self.colunas - passo / 2
        dy = (torch.arange(1, self.linhas + 1, device=dispositivo) - 0.5) * passo / self.linhas - passo / 2
        base = torch.stack(torch.meshgrid(dx, dy, indexing="xy"), -1).reshape(-1, 2)
        cx = (torch.arange(largura, device=dispositivo) + 0.5) * passo
        cy = (torch.arange(altura, device=dispositivo) + 0.5) * passo
        centros = torch.stack(torch.meshgrid(cx, cy, indexing="xy"), -1).reshape(-1, 1, 2)
        return (centros + base[None]).reshape(-1, 2)

    def forward(self, x):
        _, c3, c4, c5 = self.backbone(x)
        p4 = self.fpn(c3, c4, c5)
        lote, _, altura, largura = p4.shape
        deslocamento = aplicar_cabeca(self.regression, p4).reshape(lote, -1, 2) * 100
        logits = aplicar_cabeca(self.classification, p4).reshape(lote, -1, 2)
        pontos = deslocamento + self.ancoras(altura, largura, x.shape[2] // altura, x.device)
        return pontos, logits.softmax(-1)[..., 1]


def carregar(caminho="modelos/p2pnet_shtech_a.pth", dispositivo=None):
    caminho = Path(caminho)
    if not caminho.exists():
        caminho.parent.mkdir(parents=True, exist_ok=True)
        print(f"Baixando pesos do P2PNet para {caminho}...")
        urlretrieve(URL_PESOS, caminho)
    dispositivo = dispositivo or ("cuda" if torch.cuda.is_available() else "cpu")
    estado = torch.load(caminho, map_location="cpu", weights_only=True)
    modelo = P2PNet()
    modelo.load_state_dict(estado.get("model", estado))
    return modelo.eval().to(dispositivo)


class ContadorDensidade:
    def __init__(self, pesos="modelos/p2pnet_shtech_a.pth", limiar=0.5, lado_max=2048, escala=1.0):
        self.modelo = carregar(pesos)
        self.dispositivo = next(self.modelo.parameters()).device
        self.limiar = limiar
        self.lado_max = lado_max
        self.escala = escala

    @torch.inference_mode()
    def pontos(self, imagem):
        altura, largura = imagem.shape[:2]
        fator = min(self.escala, self.lado_max / max(altura, largura))
        nova_l = max(128, round(largura * fator / 128) * 128)
        nova_a = max(128, round(altura * fator / 128) * 128)
        rgb = cv2.cvtColor(cv2.resize(imagem, (nova_l, nova_a), interpolation=cv2.INTER_AREA if fator < 1 else cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
        x = (rgb.astype(np.float32) / 255 - MEDIA) / DESVIO
        x = torch.from_numpy(x).permute(2, 0, 1)[None].to(self.dispositivo)
        pontos, confs = self.modelo(x)
        manter = confs[0] > self.limiar
        pontos = pontos[0][manter].cpu().numpy()
        pontos[:, 0] *= largura / nova_l
        pontos[:, 1] *= altura / nova_a
        return pontos, confs[0][manter].cpu().numpy()
