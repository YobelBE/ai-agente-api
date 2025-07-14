"""
Ejemplo de uso del pandas agent de Langchain
"""
#%% Librerías

import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from dotenv import load_dotenv
import numpy as np
from datetime import datetime, timedelta

#%% Variables de entorno
load_dotenv()

OPENAI_API_KEY = "sk-proj-"

#%% Creación de data para testear
# --------------------------------

# Configuración
num_registros = 50
usuarios = ['user_123', 'user_abc']
categorias = ['Alimentación', 'Transporte', 'Ocio', 'Hogar', 'Salud', 'Otros']
descripciones = {
    'Alimentación': ['Supermercado', 'Restaurante', 'Café', 'Delivery'],
    'Transporte': ['Gasolina', 'Bus', 'Taxi', 'Metro'],
    'Ocio': ['Cine', 'Concierto', 'Libro', 'Suscripción Streaming'],
    'Hogar': ['Alquiler', 'Electricidad', 'Internet', 'Limpieza'],
    'Salud': ['Farmacia', 'Consulta Médica'],
    'Otros': ['Regalo', 'Ropa', 'Curso Online']
}

# Generar datos aleatorios
data = []
start_date = datetime(2025, 6, 1)
for i in range(num_registros):
    categoria = np.random.choice(categorias)
    descripcion = np.random.choice(descripciones[categoria])
    monto = round(np.random.uniform(5.0, 250.0), 2)
    fecha = start_date + timedelta(days=np.random.randint(0, 45), hours=np.random.randint(0, 23))
    usuario = np.random.choice(usuarios)
    
    data.append({
        'id': i + 1,
        'id_usuario': usuario,
        'descripcion': descripcion,
        'monto': monto,
        'categoria': categoria,
        'fecha_registro': fecha.strftime('%Y-%m-%d %H:%M:%S')
    })

# Crear DataFrame y guardar en CSV
df = pd.DataFrame(data)
df.to_csv('gastos_sinteticos.csv', index=False)

print("Archivo 'gastos_sinteticos.csv' creado con 50 registros.")

#%% Prueba y configuración del agente

# 1. Inicializar el modelo de lenguaje
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# 2. Crear el agente de Pandas
#    Le pasamos el LLM y el DataFrame que debe analizar.
#    verbose=True es CRUCIAL para aprender: nos mostrará el "pensamiento" del agente.
agent = create_pandas_dataframe_agent(
    llm,
    df, # Colocar el dataframe que se va a analizar
    verbose=True,
    agent_executor_kwargs={"handle_parsing_errors": True}, # Buena práctica para manejar errores,
    allow_dangerous_code=True
)

#%% 3. ¡Hacer preguntas en lenguaje natural!
print("¡Hola! Soy tu analista de datos. ¿Qué te gustaría saber sobre tus gastos?")

# --- Ejemplos de Preguntas ---
pregunta1 = "¿Cuántos gastos se han registrado en total?"
pregunta2 = "¿Cuál es el gasto total para el usuario 'user_123'?"
pregunta3 = "Muéstrame el gasto total por cada categoría, ordenado de mayor a menor."
pregunta4 = "¿Cuál fue el gasto más caro en la categoría 'Ocio'?"

print(f"\nPregunta: {pregunta4}")
response = agent.invoke(pregunta4)
print("\nRespuesta:")
print(response['output'])
