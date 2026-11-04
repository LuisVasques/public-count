# Contagem de Público

Identifica e conta as pessoas de um auditório a partir de uma **foto**, **vídeo** ou **câmera ao vivo**,
de qualquer ângulo (de cima, de frente, de trás). Também estima **cadeiras livres** e **ocupação**.

## Preparar o ambiente

Usa [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Na primeira execução o modelo `yolo26m.pt` é baixado automaticamente.

## Uso

| Script | O que faz |
|---|---|
| `contar_foto.py` | Conta o público em uma ou várias fotos (ou pastas) e salva as imagens anotadas em `saida/` com um `resumo.json`. |
| `contar_camera.py` | Conta ao vivo em webcam, arquivo de vídeo ou stream RTSP, com média móvel da contagem. |
| `publico.py` | Biblioteca: detecção em blocos, junção das caixas, região de interesse, mapa de calor e desenho. |

```bash
uv run contar_foto.py exemplos
uv run contar_foto.py exemplos/auditorio_1.jpg --calor --mostrar
uv run contar_foto.py foto.jpg --regiao 0,250,1000,250,1000,667,0,667
uv run contar_camera.py --fonte 0
uv run contar_camera.py --fonte rtsp://usuario:senha@ip:554/stream --intervalo 2
```

Na câmera: `q`/`ESC` sai, `c` liga/desliga o mapa de calor, `s` salva o quadro anotado.

### Parâmetros principais

| Parâmetro | Padrão | Efeito |
|---|---|---|
| `--modelo` | `yolo26m.pt` | `n`/`s` mais rápido, `l`/`x` mais preciso. Aceita também um detector de cabeças próprio. |
| `--bloco` | `256` | Tamanho dos blocos em que a imagem é dividida. Menor acha pessoas menores e distantes, mas é mais lento. `0` desativa. |
| `--sobreposicao` | `0.3` | Quanto os blocos se sobrepõem, para não perder quem está na borda. |
| `--conf` | `0.15` | Confiança mínima. Público denso pede valores baixos. |
| `--regiao` | | Polígono em pixels; só conta quem tem o centro dentro dele (ex.: só a plateia, sem o palco). |
| `--calor` | | Mapa de calor da densidade de pessoas. |

## Como funciona

1. **Inferência em blocos (tiling)**: numa foto de auditório cada pessoa ocupa poucos pixels. Redimensionar a foto
   inteira para 640 px faz o modelo perder a maioria. A imagem é cortada em blocos sobrepostos, cada bloco passa pelo
   YOLO na resolução cheia e as caixas voltam para as coordenadas da foto original.
2. **Junção das caixas**: a mesma pessoa aparece em mais de um bloco. Um NMS remove as caixas repetidas e um segundo
   filtro remove caixas parciais (pessoa cortada na borda de um bloco) contidas numa caixa maior da mesma pessoa.
3. **Cadeiras e ocupação**: o modelo COCO também detecta `chair`. Cadeira cujo centro não está dentro de nenhuma
   pessoa é considerada livre; ocupação = pessoas / (pessoas + cadeiras livres).

## Limitações e próximos passos

O detector COCO de pessoas tem boa precisão, mas em público muito denso (fotos de cima ou de trás, onde só se vê
a cabeça) ele **conta para baixo**: pessoas muito sobrepostas não são separadas. Caminhos para melhorar:

- **Detector de cabeças**: treinar (ou usar) um YOLO de uma classe `head` em datasets como CrowdHuman ou SCUT-HEAD.
  Basta passar o `.pt` em `--modelo`; os scripts tratam qualquer modelo sem a classe `person` como detector de cabeças.
- **Contagem por mapa de densidade** (CSRNet, DM-Count, P2PNet): em vez de caixas, estima a densidade; é o estado da
  arte para multidões com centenas de pessoas.
- Exportar para TensorRT (`model.export(format="engine")`) para acelerar a câmera ao vivo.
