import json 


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

def parametros_sistema():
    pass