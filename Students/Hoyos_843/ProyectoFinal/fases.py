import torch
import numpy as np
from PIL import Image as im
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

size = 256
dimIma = 100 
cantidadMascaras = 20
path = "./Ima/"

tensor_ima = torch.zeros(dimIma * cantidadMascaras, size, size) # Dimension de 1000x256x256
tensor_fase = torch.zeros(dimIma * cantidadMascaras, size, size) # Dimension de 1000x256x256
j = 0 
for k in range(0, dimIma):
    for i in range(0, cantidadMascaras):  
        R = im.open(path+ f"{k}.bmp") # Se lee la misma imagen
        R = R.convert('L') # Se asegura que este en escala de grises 
        R = R.resize((size, size)) # Se redimensiona al tamaño de la red
        R = np.asarray(R) # Se convierte en array de numpy
        R = R/255 # Se normaliza
        tensor_ima[j,:,:] = torch.from_numpy(R) # Se empieza a llenar el tensor con cada una de las imagenes
        MA = np.exp(1j*2*np.pi*np.random.rand(int(size), int(size))) # Se genera 100 fases aleatorias
        tensor_fase[j,:,:] = torch.from_numpy(MA)
        print(j)
        j += 1
        
    

Input = tensor_ima * tensor_fase

U = torch.fft.ifftshift(torch.fft.ifft2(torch.fft.fftshift(Input)))
U = torch.exp(1j*torch.angle(U)) # Se extrae la fase de la operacion anterior y se hace cte la amplitud

pathSave = "./Output100fases/"
fase = torch.angle(U) # Se extrae la fase 
fase = fase + torch.pi # Deja la fase entre 0 y 2pi 
fase = fase/(2*torch.pi) # Deja la fase entre 0 y 1
fase = fase * 255 # Deja la fase entre 0 y 255
for i in range(0, dimIma * cantidadMascaras): 
    newFase=np.asarray(fase[i])
    newFase=im.fromarray(newFase).convert('L')
    newFase.save(pathSave +"/"+ str(i)+".bmp") 