import json 
from collections import namedtuple
import psutil
import datetime
import os
import time

def read_prompts():
    lista_prompts = []
    ruta_fichero = 'ollama-pruebas/prompts.jsonl'
    '''
El fichero de propmts lo he sacado de
https://github.com/google-research/google-research/tree/master/instruction_following_eval
'''
    with open(ruta_fichero,'r', encoding='utf-8')as fichero : 
        for l in fichero:
            try:
                #print(l)
                lista_prompts.append(json.loads(l).get('prompt'))
            except json.decoder.JSONDecodeError as e :
                print("Error en el fichero: ", e)
    return lista_prompts
Parametros_sistema = namedtuple('Parametros_sistema',['cpu','memoria','temp','fecha'])
def obtiene_parametros(): 
    cpu = psutil.cpu_percent()
    memoria = psutil.virtual_memory().percent
    temp = psutil.sensors_temperatures()
    fecha = datetime.datetime.now()
    return Parametros_sistema(cpu,memoria,temp,fecha)

async def parametros_sistema():
    # creamos un fichero 
    ruta_fichero = 'ollama-pruebas/parametros_sistema.json'
    while True:
        parametros = obtiene_parametros()
        with open(ruta_fichero,'a') as fichero:
            json.dump(parametros._asdict(),fichero)
            fichero.write('\n')
        time.sleep(1)
        print("parametros guardados")
   
    # luego lo leemos

