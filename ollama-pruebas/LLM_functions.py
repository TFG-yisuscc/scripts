from ollama import chat
from ollama import ChatResponse

# response: ChatResponse = chat(model='llama3.2', messages=[
#   {
#     'role': 'user',
#     'content': 'hola que tal, como se encuentra usted?',
#   },
# ])
# print(response['message']['content'])
# # or access fields directly from the response object
# print(response.message.content)

def query_llm(modelo,promt): 
    response: ChatResponse = chat(model=modelo, messages=[
      {
        'role': 'user',
        'content': promt,
      },
    ])
    print(response.message.content)
    return response['message']['content']

def query_iterator(modelo,lista_prompts):
    lista_respuestas = []
    for prompt in lista_prompts:
        lista_respuestas.append(query_llm(modelo,prompt))
    return lista_respuestas