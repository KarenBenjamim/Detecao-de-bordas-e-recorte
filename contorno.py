import numpy as np
import cv2
import imutils
from matplotlib import pyplot as plt

# Processamentos na imagem

caminho = '' #Caminho relativo da imagem 
img = cv2.imread(caminho)
original = img.copy()
(H, W) = img.shape[:2]

# Pré-processamento e filtros

# Ajuste do brilho e contraste para retirar a marca de agua.
brilho = 200
contraste = 100
ajustes = np.int16(img)
ajustes.shape

ajustes = ajustes * (contraste / 128 ) - contraste + brilho
ajustes = np.clip(ajustes, 0, 255)
ajustes = np.uint8(ajustes)

# Escala de cinza

gray = cv2.cvtColor(ajustes, cv2.COLOR_BGR2GRAY)

# Limiarização com método Otsu (thresholding)

val, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# Aplicação de desfoque (Gaussian Blur)

blur = cv2.GaussianBlur(thresh, (5, 5), 0)

# Detecção de bordas (Canny Edge)

edged = cv2.Canny(blur, 60, 160)

# Dilatação das bordas

dilatacao = cv2.dilate(edged, np.ones((3,3), np.uint8))

# Detecção de contornos na imagem

def encontrar_contornos(img):
  conts = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
  conts = imutils.grab_contours(conts)
  conts = sorted(conts, key = cv2.contourArea, reverse = True)[:6]
  return conts

conts = encontrar_contornos(dilatacao.copy())

# Localizando o maior contorno

for c in conts:
  perimetro = cv2.arcLength(c, True)
  aproximacao = cv2.approxPolyDP(c, 0.02 * perimetro, True)
  if len(aproximacao) == 4: #Alterar para a quantidade de Vértices (alternativa ">=4")
    maior = aproximacao
    break

# Ordenando os pontos

def ordenar_pontos(pontos):
  pontos = pontos.reshape((4,2))
  pontos_novos = np.zeros((4, 1, 2), dtype=np.int32)

  add = pontos.sum(1)
  pontos_novos[0] = pontos[np.argmin(add)]
  pontos_novos[2] = pontos[np.argmax(add)]

  dif = np.diff(pontos, axis = 1)
  pontos_novos[1] = pontos[np.argmin(dif)]
  pontos_novos[3] = pontos[np.argmax(dif)]

  return pontos_novos

pontos_maior = ordenar_pontos(maior)

# Matriz de transformação

pts1 = np.float32(pontos_maior)
pts2 = np.float32([[0,0], [W, 0], [W, H], [0, H]])

matriz = cv2.getPerspectiveTransform(pts1, pts2)
matriz

# Recorte

transform = cv2.warpPerspective(original, matriz, (W, H))

#Salvando a imagem recortada
cv2.imwrite("ImagemRecortada.png", transform) 
print("imagem salva")