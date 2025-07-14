#%% Librerías
# ------------
from langchain_openai import ChatOpenAI
import os
from flask import Flask, jsonify, request
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from langchain_openai import OpenAIEmbeddings
from langchain_elasticsearch import ElasticsearchStore
from langchain.agents import AgentExecutor
from psycopg_pool import ConnectionPool
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.prebuilt import create_react_agent
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from datetime import datetime, timezone
import pytz
from langchain_core.tools import tool, Tool
import time
import pandas as pd


#%% Variables de entorno
# --------------------------

os.environ["LANGSMITH_ENDPOINT"]="https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "lsv2_"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "agente_ia_123"
os.environ["OPENAI_API_KEY"] ="sk-proj-"

DB_URI = "postgresql://postgres:"

#%% Configuración de las conexiones
# ------------------------------------

connection_kwargs = {
    "autocommit": True,
    "prepare_threshold": 0,
}

# El pool se crea aquí y permanece abierto durante toda la vida de la aplicación
db_pool = ConnectionPool(conninfo=DB_URI, max_size=20, kwargs=connection_kwargs)
checkpointer = PostgresSaver(db_pool)

#%% Configuración de la herramienta de búsqueda Elasticsearch

# Preguntas frecuentes
db_query_pf = ElasticsearchStore(
    es_url="http://0.0.0.0:9200",
    es_user="elastic",
    es_password="WNzA1O*Daecj59qNcJ4V",
    index_name="preguntas_frecuentes", #Nombre de la colección específica
    embedding=OpenAIEmbeddings() # Para convertir la consulta en embedding 
    )

# Nota: El modelo de Embedding para la búsqueda debe ser el mismo
# modelo que se empleó para realizar el cargue.

# Consejos de ahorro
db_query_ca = ElasticsearchStore(
    es_url="http://0.0.0.0:9200",
    es_user="elastic",
    es_password="WNzA1O*Daecj59qNcJ4V",
    index_name="consejos_de_ahorro", #Nombre de la colección específica
    embedding=OpenAIEmbeddings() # Para convertir la consulta en embedding 
    )

# Nota: El modelo de Embedding para la búsqueda debe ser el mismo
# modelo que se empleó para realizar el cargue.

#%% Modelo de LLM
# ----------------------

# Inicializamos el modelo
model = ChatOpenAI(model="gpt-4.1-2025-04-14", temperature=0.1)

#%% Herramientas

# ---------------------
# Herrramientas RAG
# ---------------------
retriever_pf = db_query_pf.as_retriever()
tool_rag_pf = retriever_pf.as_tool(
    name="preguntas_frecuentes",
    description="Consulta en el documento de preguntas frecuentes sobre ahorro e inversión",
)

retriever_ca = db_query_ca.as_retriever()
tool_rag_ca = retriever_ca.as_tool(
    name="consejos_de_ahorro",
    description="Consulta en el documento de consejos de ahorro",
)

# ----------------------------------------
# Herramientas adicionales
# ----------------------------------------
@tool
def get_now():
    """
    Obtiene la fecha y hora actual en la zona horaria especificada
    """
    zona = pytz.timezone("America/Lima")
    fecha_hora = datetime.now(zona)
    return fecha_hora.strftime("%Y-%m-%d %H:%M:%S")


@tool
def registrar_gasto(descripcion: str, monto: float, categoria: str, fecha: str):
    """
    Registra un gasto en la base de datos
    
    Args:
        descripcion: Descripción del gasto
        monto: Cantidad gastada (debe ser mayor a 0)
        categoria: Categoría del gasto (Alimentación, Transporte, Ocio, Hogar, Educación, Salud, Otros)
        fecha: Fecha del gasto (usa fecha actual si no se proporciona) el formato correcto es YYYY-MM-DD HH:MM:SS
    """
    try:
        # Validaciones básicas
        if not descripcion or not categoria:
            return "❌ Error: id_usuario, descripcion y categoria son campos obligatorios"
        
        if monto <= 0:
            return "❌ Error: El monto debe ser mayor a 0"
        
        # Validar categorías permitidas
        categorias_validas = ['Alimentación', 'Transporte', 'Ocio', 'Hogar', 'Educación', 'Salud', 'Otros']

        if categoria not in categorias_validas:
            return f"❌ Error: Categoría '{categoria}' no válida. Categorías permitidas: {', '.join(categorias_validas)}"
        
        # Usar fecha actual si no se proporciona
        if fecha is None:
            fecha = get_now()
        
        with db_pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO gastos (descripcion, monto, categoria, fecha_registro)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id
                """, (descripcion, monto, categoria, fecha))
                
                gasto_id = cur.fetchone()[0]
                return f"✅ Gasto registrado exitosamente. ID: {gasto_id}, , Descripción: {descripcion}, Monto: ${monto:,.2f}, Categoría: {categoria}, Fecha: {fecha}"
                
    except Exception as e:
        return f"❌ Error al registrar el gasto: {str(e)}"


def obtener_gastos_sql(db_pool):
    """
    Obtiene los datos de la tabla 'gastos' de SQL y los carga en un DataFrame
    """

    with db_pool.connection() as conn:
        with conn.cursor() as cur:
            # 1. Ejecutar la consulta para obtener todos los datos
            cur.execute("SELECT * FROM gastos")
            
            # 2. Obtener todas las filas en una lista de tuplas
            rows = cur.fetchall()
            
            # 3. Obtener los nombres de las columnas (esto es complicado)
            column_names = [desc[0] for desc in cur.description]
            
            # 4. Crear el DataFrame manualmente
            df = pd.DataFrame(rows, columns=column_names)
            
            return df

def crear_analista_de_gastos(model) -> AgentExecutor:
    """
    Carga los datos de gastos desde la BD y crea un agente de pandas para realizar
    el análisis de los datos.
    """
    # Obtenemos los datos en un DataFrame
    df = obtener_gastos_sql(db_pool)

    if df.empty:
        return "No hay gastos registrados para analizar."

    return create_pandas_dataframe_agent(model, df, verbose=True, allow_dangerous_code=True)


def ejecutar_analisis(pregunta_del_usuario: str) -> str:
    # Creamos el agente de pandas justo en el momento, con los datos más recientes
    agente_pandas = crear_analista_de_gastos(model) 
    # Invocamos al agente con la pregunta específica del usuario
    respuesta = agente_pandas.invoke(pregunta_del_usuario)
    return respuesta['output']

# Configuramos esta función como una herramienta
tool_analizar_gastos = Tool(
    name="analizar_historial_gastos",
    func=ejecutar_analisis, # La función es invocar al sub-agente
    description="""
    Útil cuando el usuario quiere analizar los gastos que ha realizado en el pasado o quiere ver el historial de sus gastos
    """
)

# Agrupamos las herramientas
toolkit = [tool_rag_pf, tool_rag_ca, get_now, registrar_gasto, tool_analizar_gastos]

prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
            """
            # Rol
            Eres Ahorrín, un asistente financiero personal experto en el registro de gastos y la creación de presupuestos. Tu tono es amigable, alentador y profesional. Usas emojis de forma sutil para hacer la conversación más cercana (ej. 💰, 📊, 👍).


            # Contexto y Objetivo
            Tu objetivo principal es ayudar al usuario a tomar el control de sus finanzas de una manera sencilla y conversacional. Interactúas con el usuario para registrar transacciones, analizar su historial financiero y ofrecerle consejos prácticos. Tienes acceso a un conjunto de herramientas especializadas para realizar estas tareas.
            
            # Tarea y Flujo de Pensamiento (Chain of Thought)

            Para cada mensaje del usuario, sigue este proceso de pensamiento:

            1.  **Identificar la Intención:** Primero, determina la intención principal del usuario. ¿Quiere registrar un gasto o ingreso? ¿Quiere hacer una pregunta sobre sus gastos pasados? ¿O está pidiendo un consejo general?

            2.  **Seleccionar la Herramienta Adecuada:** Basado en la intención, elige la herramienta correcta:

                *   Si el usuario menciona una compra, un gasto, un pago o un ingreso (ej. "compré pan", "gasté en el bus"), tu herramienta principal es `registrar_gasto`. Debes extraer la descripción, el monto y la categoría del texto. Si la categoría no es clara, pregunta al usuario para que elija una de las siguientes: [Alimentación, Transporte, Ocio, Hogar, Educación, Salud, Ingresos, Otros]. Y si la categoría es clara asígnala.
                *   Si el usuario hace una pregunta que requiere analizar datos históricos (ej. "¿cuánto gasté en...?", "¿cuál es mi categoría con más gastos?", "muéstrame un resumen"), tu herramienta es `analizar_historial_financiero`.
                *   Si el usuario pide consejos de ahorro, pregunta sobre tus capacidades o hace una pregunta general sobre finanzas, tu herramienta es `consultar_base_de_conocimiento`.

            3.  **Ejecutar y Formular la Respuesta:** Una vez que tengas el resultado de la herramienta, formula una respuesta clara y concisa para el usuario.
                *   Al registrar un gasto, confirma la acción: "¡Listo! 👍 He registrado tu gasto en [Producto] por [Monto] en la categoría [Categoría]."
                *   Al analizar, presenta los datos de forma sencilla.
                *   Al dar consejos, sé práctico y directo.

            # Restricciones y Reglas

            - **NUNCA inventes información.** Si no puedes responder con las herramientas que tienes, dilo claramente. Ejemplo: "Lo siento, no tengo acceso a información sobre la inflación actual, pero puedo analizar tus gastos para ver cómo te afecta."
            - **Prioriza las herramientas sobre tu conocimiento general.** Tu función principal es ser un registrador y analista de los datos del usuario. No respondas preguntas sobre gastos pasados de memoria; SIEMPRE usa la herramienta `analizar_historial_financiero`.
            - **Sé proactivo pero no insistente.** Si registras un gasto, puedes sugerir un análisis: "¡Gasto registrado! ¿Te gustaría ver un resumen de tus gastos de esta semana?".
            - **Manejo de errores:** Si una herramienta falla, informa al usuario de manera amigable: "Parece que hubo un problema al registrar tu gasto. ¿Podrías intentarlo de nuevo con un poco más de detalle?".

            # Formato de Respuesta

            - **Inicio de Conversación:** La primera vez que hables en una conversación, preséntate: "¡Hola! Soy Ahorrín 💰, tu asistente de finanzas personales. ¿En qué te puedo ayudar hoy? ¿Quieres registrar un gasto o analizar tus finanzas?".
            - **Longitud:** Mantén tus respuestas por debajo de las 150 palabras. Sé breve y ve al grano.
            - **Claridad:** Usa listas con viñetas o negritas para presentar datos importantes y hacerlos fáciles de leer.

            """),
        ("human", "{messages}"),
    ]
)

# Configuración del agente
agent_executor = create_react_agent(model, toolkit, checkpointer=checkpointer, prompt=prompt)


app = Flask(__name__)

@app.route('/agent', methods=['GET'])
def main():
    #Capturamos variables enviadas
    id_agente = request.args.get('idagente')
    msg = request.args.get('msg')

    # Ejecutamos el agente
    config = {"configurable": {"thread_id": id_agente}}
    response = agent_executor.invoke({"messages": [HumanMessage(content=msg)]}, config=config)
    return response['messages'][-1].content


if __name__ == '__main__':
    # La aplicación escucha en el puerto 8080, requerido por Cloud Run
    app.run(host='0.0.0.0', port=8080)