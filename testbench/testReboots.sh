!/bin/bash

# Archivo donde se guardsan los comados
COMMANDS_FILE="/home/user1/script/testbench/commands.txt"

# donde guardamos los estados 
STATE_FILE="/var/tmp/command_index"

# Verificar que el archivo de comandos existe
if [ ! -f "$COMMANDS_FILE" ]; then
  echo " No se encontró el archivo de comandos: $COMMANDS_FILE"
  exit 1
fi

# Cargar los comandos en un array
mapfile -t COMMANDS < "$COMMANDS_FILE"

# Crear archivo de estado si no existe
if [ ! -f "$STATE_FILE" ]; then
  echo 0 > "$STATE_FILE"
fi

# Leer índice actual
INDEX=$(cat "$STATE_FILE")

# Si ya ejecutamos todos, salir
if [ "$INDEX" -ge "${#COMMANDS[@]}" ]; then
  echo "Todos los comandos ya fueron ejecutados."
  exit 0
fi
sleep 3m
# Entramos en la carpeta de monitor system 
cd /home/user1/MonitorSystem/
# activamos el venv
source .venv/bin/Activate

# Ejecutamos el comando dequ corresponda
echo "Ejecutando comando ${INDEX}: ${COMMANDS[$INDEX]}"
eval "${COMMANDS[$INDEX]}"

# Incrementar el índice y guardarlo
NEXT_INDEX=$((INDEX + 1))
echo "$NEXT_INDEX" > "$STATE_FILE"

# Si aún quedan comandos, reiniciar
if [ "$NEXT_INDEX" -lt "${#COMMANDS[@]}" ]; then
  echo "Reiniciando para ejecutar el siguiente comando..."
  sleep 3
  reboot
else
  echo "Todos los comandos ejecutados. No se reiniciará más."
fi
