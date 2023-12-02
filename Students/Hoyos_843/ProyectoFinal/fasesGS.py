import torch
import numpy as np
from PIL import Image as im
import matplotlib.pyplot as plt

size = 256
iterations = 100
dimIma = 100
cantidadMascaras = 20
path = "./Ima/"

tensor_ima = torch.zeros(cantidadMascaras * dimIma, size, size) # Dimension de 1000x256x256
j = 0
for k in range(0, dimIma): 
    for i in range(0, cantidadMascaras):  
        R = im.open(path+ f"{k}.bmp") # Se abre la imagen
        R = R.convert('L') # Se asegura que este en escala de grises 
        R = R.resize((size, size)) # Se redimensiona al tamaño de la red
        R = np.asarray(R) # Se convierte en array de numpy
        R = R/255 # Se normaliza
        tensor_ima[j,:,:] = torch.from_numpy(R) # Se empieza a llenar el tensor con cada una de las imagenes
        j += 1

MA = torch.from_numpy(np.exp(1j*2*np.pi*np.random.rand(int(size), int(size)))) # mascara aleatoria

Input = tensor_ima * MA # Se multiplica cada una de las imagenes por la mascara aleatoria


# GS
for i in range(0, iterations): 
    U = torch.fft.ifftshift(torch.fft.ifft2(torch.fft.fftshift(Input)))
    U = torch.exp(1j*torch.angle(U)) # Se extrae la fase de la operacion anterior y se hace cte la amplitud
    ug = torch.fft.ifftshift(torch.fft.fft2(torch.fft.fftshift(U))) # Se hace TF 
    Input = tensor_ima*torch.exp(1j*torch.angle(ug));
    print(i)

pathSave = "./Output100GS/"
fase = torch.angle(U) # Se extrae la fase 
fase = fase + torch.pi # Deja la fase entre 0 y 2pi 
fase = fase/(2*torch.pi) # Deja la fase entre 0 y 1
fase = fase * 255 # Deja la fase entre 0 y 255
for i in range(0, dimIma * cantidadMascaras): 
    newFase=np.asarray(fase[i])
    newFase=im.fromarray(newFase).convert('L')
    newFase.save(pathSave +"/"+ str(i)+".bmp") 