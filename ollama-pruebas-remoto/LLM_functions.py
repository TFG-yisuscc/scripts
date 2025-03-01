from ollama import chat, ChatResponse, Client

client = Client(
  host='http://raspberrypi1.local:11434',
  headers={'x-some-header': 'some-value'}
)

def query_llm(modelo,promt): 
    response: ChatResponse = client.chat(model=modelo, messages=[
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