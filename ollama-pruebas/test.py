#!/usr/bin/env python3
from auxiliary_functions import *
from LLM_functions import *
import asyncio
import time

async def lista(): 
    for i in range(100):
        print(i)
        await asyncio.sleep(1)
async def prueba1():
    while(True):
        print("hola")
        await asyncio.sleep(1)

async def main():
    # llamadas_llm= loop.create_task(query_iterator('llama3.2',lista_prompts))
    # mediciones_sistema = loop.create_task(parametros_sistema())
    # await asyncio.wait([llamadas_llm,mediciones_sistema])
   
    l1 = loop.create_task(lista())
    l2 = loop.create_task(prueba1())
    await l1
    l2.cancel()
    print("fin")
if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    loop.close()






        


 