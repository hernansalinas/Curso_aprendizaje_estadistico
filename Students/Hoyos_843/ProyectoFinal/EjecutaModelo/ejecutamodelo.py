import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as im
from segmentation_model import SegmentationModel  # Importa la clase SegmentationModel

model_path = 'modelos/best-model.pt'
model = torch.load(model_path, map_location=torch.device('cpu'))
model.eval()

size = 256 # dimension
folder_path1 = './Output/'
R = im.open(folder_path1+ f"400.bmp") # Se abre la imagen
R = R.convert('L') # Se asegura que este en escala de grises
R = R.resize((size, size)) # Se redimensiona al tamaño de la red
R = np.asarray(R) # Se convierte en array de numpy
R = R/255 # Se normaliza
R = torch.Tensor(R)
input_data = R.unsqueeze(0)

with torch.no_grad():
    output = model(input_data)

a = output.squeeze(0)
output = a.squeeze(0)
numpy_array = output.numpy()

# Encontrar el mínimo y el máximo
min_val = numpy_array.min()
max_val = numpy_array.max()

# Normalizar la matriz
matriz_normalizada = (numpy_array - min_val) / (max_val - min_val)

# Escalar la matriz al rango 0-255 y convertirla a uint8
imagen = (matriz_normalizada * 255).astype(np.uint8)

# Convertir la matriz a un objeto de imagen de Pillow
imagen_pil = im.fromarray(imagen)

# Guardar la imagen en formato BMP
imagen_pil.save("400.bmp")

