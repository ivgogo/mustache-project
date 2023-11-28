import pandas as pd
import numpy as np
import yaml

# Config file has: data directories, architecture of the network and hyperparameters for the training session.
def cargar_configuracion(ruta_archivo):
    with open(ruta_archivo, 'r') as archivo:
        configuracion = yaml.safe_load(archivo)
    return configuracion

# Ejemplo de uso
ruta_configuracion = '/home/ivan/Documentos/project/run/config1.yaml'
configuracion = cargar_configuracion(ruta_configuracion)

# Acceder a los parámetros
data_dir = configuracion['DATA']['DATA_DIR']
modo = configuracion['DATA']['MODE']
arquitectura_modelo = configuracion['MODEL']['ARCH']
batch_size = configuracion['TRAIN']['BATCH_SIZE']
epocas = configuracion['TRAIN']['EPOCHS']
learning_rate = configuracion['TRAIN']['LR']

# Imprimir los parámetros
print(f'Data Directory: {data_dir}')
print(f'Modo: {modo}')
print(f'Arquitectura del Modelo: {arquitectura_modelo}')
print(f'Batch Size: {batch_size}')
print(f'Épocas: {epocas}')
print(f'Learning Rate: {learning_rate}')