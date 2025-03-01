#!/usr/bin/env python3
from auxiliary_functions import *
from LLM_functions import *
import asyncio
import time

lista_prompts = read_prompts()

async def main():
    llamadas_llm= loop.create_task(query_iterator('llama3.2',lista_prompts[0:10]))
    mediciones_sistema = loop.create_task(parametros_sistema())
    await llamadas_llm
    mediciones_sistema.cancel()
    print("fin")
if __name__ == "__main__":
    loop = loop = asyncio.new_event_loop()
    loop.run_until_complete(main())
    loop.close()






        


 