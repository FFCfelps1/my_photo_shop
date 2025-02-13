from PIL import Image, ImageEnhance, ImageFilter, ImageDraw, ImageFont
import os
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.models as models
import cv2
from rembg import remove

def cortar_imagem(caminho_imagem, coordenadas, caminho_saida):
    """
    Corta uma imagem com base nas coordenadas fornecidas e salva a imagem cortada.

    :param caminho_imagem: Caminho para a imagem original.
    :param coordenadas: Uma tupla (esquerda, superior, direita, inferior) que define a área a ser cortada.
    :param caminho_saida: Caminho para salvar a imagem cortada.
    """
    # Carregar a imagem
    imagem = Image.open(caminho_imagem)
    
    # Cortar a imagem
    imagem_cortada = imagem.crop(coordenadas)
    
    # Remover a parte preta
    bbox = imagem_cortada.getbbox()
    if bbox:
        imagem_cortada = imagem_cortada.crop(bbox)
    
    # Salvar a imagem cortada
    imagem_cortada.save(caminho_saida)
    print(f"Imagem cortada salva em {caminho_saida}")

def segmentar_pessoas(caminho_imagem, caminho_saida, limiar=0.5):
    """
    Segmenta pessoas em uma imagem e salva a imagem segmentada.

    :param caminho_imagem: Caminho para a imagem original.
    :param caminho_saida: Caminho para salvar a imagem segmentada.
    :param limiar: Limiar para a máscara binária.
    """
    # Carregar a imagem
    imagem = Image.open(caminho_imagem).convert("RGB")
    
    # Transformar a imagem para tensor
    transform = T.Compose([T.ToTensor()])
    imagem_tensor = transform(imagem).unsqueeze(0)
    
    # Carregar o modelo pré-treinado de segmentação
    modelo = models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    modelo.eval()
    
    # Fazer a segmentação
    with torch.no_grad():
        pred = modelo(imagem_tensor)
    
    # Obter a máscara para a classe 'pessoa' (classe 1 no COCO dataset)
    masks = pred[0]['masks']
    labels = pred[0]['labels']
    mask_pessoas = masks[labels == 1].squeeze().cpu().numpy()
    
    # Se houver várias máscaras, combinar elas
    if len(mask_pessoas.shape) == 3:
        mask_pessoas = np.mean(mask_pessoas, axis=0)
    
    # Aplicar o limiar na máscara
    mask_pessoas = (mask_pessoas > limiar).astype(np.uint8)
    
    # Aplicar a máscara na imagem original
    imagem_np = np.array(imagem)
    imagem_segmentada = cv2.bitwise_and(imagem_np, imagem_np, mask=mask_pessoas)
    
    # Aplicar pós-processamento (remoção de ruído e suavização)
    kernel = np.ones((5, 5), np.uint8)
    imagem_segmentada = cv2.morphologyEx(imagem_segmentada, cv2.MORPH_CLOSE, kernel)
    imagem_segmentada = cv2.GaussianBlur(imagem_segmentada, (5, 5), 0)
    
    # Salvar a imagem segmentada
    Image.fromarray(imagem_segmentada).save(caminho_saida)
    print(f"Imagem segmentada salva em {caminho_saida}")

def remover_backGround(caminho_imagem, caminho_saida):
    """
    Remove o fundo de uma imagem e salva a imagem resultante.

    :param caminho_imagem: Caminho para a imagem original.
    :param caminho_saida: Caminho para salvar a imagem sem fundo.
    """
    input = Image.open(caminho_imagem)
    output = remove(input)
    output.save(caminho_saida, format="PNG")
    print(f"Imagem sem fundo salva em {caminho_saida}")

def rodar_imagem(caminho_imagem, angulo, caminho_saida):
    """
    Roda uma imagem pelo ângulo especificado e salva a imagem rodada.

    :param caminho_imagem: Caminho para a imagem original.
    :param angulo: Ângulo de rotação em graus.
    :param caminho_saida: Caminho para salvar a imagem rodada.
    """
    # Carregar a imagem
    imagem = Image.open(caminho_imagem)
    
    # Rodar a imagem
    imagem_rodada = imagem.rotate(angulo, expand=True)
    
    # Converter a imagem para um array numpy
    imagem_np = np.array(imagem_rodada)
    
    # Detectar a área não preta
    mask = imagem_np[:, :, :3] != 0
    mask = mask.any(axis=2)
    coords = np.argwhere(mask)
    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0) + 1
    
    # Cortar a área não preta
    imagem_cortada = imagem_rodada.crop((y0, x0, y1, x1))
    
    # Salvar a imagem cortada
    imagem_cortada.save(caminho_saida)
    print(f"Imagem cortada salva em {caminho_saida}")

def redimensionar_imagem(caminho_imagem, largura, altura, caminho_saida):
    """
    Redimensiona uma imagem para a largura e altura especificadas e salva a imagem redimensionada.

    :param caminho_imagem: Caminho para a imagem original.
    :param largura: Nova largura da imagem.
    :param altura: Nova altura da imagem.
    :param caminho_saida: Caminho para salvar a imagem redimensionada.
    """
    # Carregar a imagem
    imagem = Image.open(caminho_imagem)
    
    # Redimensionar a imagem
    imagem_redimensionada = imagem.resize((largura, altura))
    
    # Salvar a imagem redimensionada
    imagem_redimensionada.save(caminho_saida)
    print(f"Imagem redimensionada salva em {caminho_saida}")

def ajustar_brilho(caminho_imagem, fator, caminho_saida):
    """
    Ajusta o brilho de uma imagem e salva a imagem resultante.

    :param caminho_imagem: Caminho para a imagem original.
    :param fator: Fator de ajuste do brilho (1.0 mantém o brilho original).
    :param caminho_saida: Caminho para salvar a imagem com brilho ajustado.
    """
    # Carregar a imagem
    imagem = Image.open(caminho_imagem)
    
    # Ajustar o brilho
    enhancer = ImageEnhance.Brightness(imagem)
    imagem_brilho = enhancer.enhance(fator)
    
    # Salvar a imagem com brilho ajustado
    imagem_brilho.save(caminho_saida)
    print(f"Imagem com brilho ajustado salva em {caminho_saida}")

def ajustar_contraste(caminho_imagem, fator, caminho_saida):
    """
    Ajusta o contraste de uma imagem e salva a imagem resultante.

    :param caminho_imagem: Caminho para a imagem original.
    :param fator: Fator de ajuste do contraste (1.0 mantém o contraste original).
    :param caminho_saida: Caminho para salvar a imagem com contraste ajustado.
    """
    # Carregar a imagem
    imagem = Image.open(caminho_imagem)
    
    # Ajustar o contraste
    enhancer = ImageEnhance.Contrast(imagem)
    imagem_contraste = enhancer.enhance(fator)
    
    # Salvar a imagem com contraste ajustado
    imagem_contraste.save(caminho_saida)
    print(f"Imagem com contraste ajustado salva em {caminho_saida}")

def aplicar_filtro(caminho_imagem, filtro, caminho_saida):
    """
    Aplica um filtro a uma imagem e salva a imagem resultante.

    :param caminho_imagem: Caminho para a imagem original.
    :param filtro: Filtro a ser aplicado (ex: ImageFilter.BLUR).
    :param caminho_saida: Caminho para salvar a imagem com o filtro aplicado.
    """
    # Carregar a imagem
    imagem = Image.open(caminho_imagem)
    
    # Aplicar o filtro
    imagem_filtrada = imagem.filter(filtro)
    
    # Salvar a imagem com o filtro aplicado
    imagem_filtrada.save(caminho_saida)
    print(f"Imagem com filtro aplicado salva em {caminho_saida}")

def converter_para_escala_de_cinza(caminho_imagem, caminho_saida):
    """
    Converte uma imagem para escala de cinza e salva a imagem resultante.

    :param caminho_imagem: Caminho para a imagem original.
    :param caminho_saida: Caminho para salvar a imagem em escala de cinza.
    """
    # Carregar a imagem
    imagem = Image.open(caminho_imagem)
    
    # Converter para escala de cinza
    imagem_cinza = imagem.convert("L")
    
    # Salvar a imagem em escala de cinza
    imagem_cinza.save(caminho_saida)
    print(f"Imagem em escala de cinza salva em {caminho_saida}")

def adicionar_texto(caminho_imagem, texto, posicao, caminho_saida, tamanho_fonte=20, cor_fonte="white"):
    """
    Adiciona texto a uma imagem e salva a imagem resultante.

    :param caminho_imagem: Caminho para a imagem original.
    :param texto: Texto a ser adicionado.
    :param posicao: Posição (x, y) onde o texto será adicionado.
    :param caminho_saida: Caminho para salvar a imagem com o texto adicionado.
    :param tamanho_fonte: Tamanho da fonte do texto.
    :param cor_fonte: Cor da fonte do texto.
    """
    # Carregar a imagem
    imagem = Image.open(caminho_imagem)
    
    # Criar um objeto de desenho
    draw = ImageDraw.Draw(imagem)
    
    # Definir a fonte
    fonte = ImageFont.truetype("arial.ttf", tamanho_fonte)
    
    # Adicionar o texto
    draw.text(posicao, texto, font=fonte, fill=cor_fonte)
    
    # Salvar a imagem com o texto adicionado
    imagem.save(caminho_saida)
    print(f"Imagem com texto adicionado salva em {caminho_saida}")

# Exemplo de uso
caminho_imagem = os.path.join(os.path.dirname(__file__), "imagem.jpg")  # Caminho relativo para a imagem na mesma pasta
coordenadas = (100, 100, 400, 400)  # (esquerda, superior, direita, inferior)
caminho_saida_corte = os.path.join(os.path.dirname(__file__), "imagem_cortada.jpg")  # Caminho relativo para salvar a imagem cortada
caminho_saida_segmentada = os.path.join(os.path.dirname(__file__), "imagem_segmentada.jpg")  # Caminho relativo para salvar a imagem segmentada
caminho_saida_sem_fundo = os.path.join(os.path.dirname(__file__), "imagem_sem_fundo.png")  # Caminho relativo para salvar a imagem sem fundo
caminho_saida_rodada = os.path.join(os.path.dirname(__file__), "imagem_rodada.jpg")  # Caminho relativo para salvar a imagem rodada
caminho_saida_redimensionada = os.path.join(os.path.dirname(__file__), "imagem_redimensionada.jpg")  # Caminho relativo para salvar a imagem redimensionada
caminho_saida_brilho = os.path.join(os.path.dirname(__file__), "imagem_brilho.jpg")  # Caminho relativo para salvar a imagem com brilho ajustado
caminho_saida_contraste = os.path.join(os.path.dirname(__file__), "imagem_contraste.jpg")  # Caminho relativo para salvar a imagem com contraste ajustado
caminho_saida_filtro = os.path.join(os.path.dirname(__file__), "imagem_filtro.jpg")  # Caminho relativo para salvar a imagem com filtro aplicado
caminho_saida_cinza = os.path.join(os.path.dirname(__file__), "imagem_cinza.jpg")  # Caminho relativo para salvar a imagem em escala de cinza
caminho_saida_texto = os.path.join(os.path.dirname(__file__), "imagem_texto.jpg")  # Caminho relativo para salvar a imagem com texto adicionado

#cortar_imagem(caminho_imagem, coordenadas, caminho_saida_corte)
#segmentar_pessoas(caminho_imagem, caminho_saida_segmentada)
#remover_backGround(caminho_imagem, caminho_saida_sem_fundo)
#rodar_imagem(caminho_imagem, 45, caminho_saida_rodada)
#redimensionar_imagem(caminho_imagem, 800, 600, caminho_saida_redimensionada)
#ajustar_brilho(caminho_imagem, 1.5, caminho_saida_brilho)
#ajustar_contraste(caminho_imagem, 1.5, caminho_saida_contraste)
#aplicar_filtro(caminho_imagem, ImageFilter.BLUR, caminho_saida_filtro)
#converter_para_escala_de_cinza(caminho_imagem, caminho_saida_cinza)
#adicionar_texto(caminho_imagem, "Hello, World!", (50, 50), caminho_saida_texto)