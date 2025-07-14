# Aplicación de GenAI *end‑to‑end* AhorraIA

**Construye un asistente conversacional** que combina:

- **LangChain** para orquestar la lógica de IA  
- **RAG** (Retrieve‑Augment‑Generate) sobre **Elasticsearch 8.x**  
- **Memoria de corto plazo** con **Cloud SQL (PostgreSQL)** y checkpoints  
- **Trazabilidad** completa vía **LangSmith**  
- Despliegue **Docker → Cloud Run (GCP)** y **UI React / Next.js en Vercel**

> Este repositorio está basado en el siguiente repositorio: https://github.com/macespinoza/gcp-ai-agent-starter-kit  


---

## ⬇️ Contenido del repositorio

| Ruta | Descripción |
|------|-------------|
| `cargar_db.ipynb` | Notebook con el procedimiento para crear las tablas que funcionarán como checkpoint |
| `pandas_agent.py` | Script con un ejemplo de como se emplea y como funciona un agente de pandas para analizar información |
| `pdf_to_rag.py` | Script con un ejemplo de como se cargan los PDFs a la base de datos vectorial de Elastic Search|
| `test_local.py` | Script para realizar pruebas de manera local|
| `rag_files/` | Archivos de ejemplo que fueron a la base de datos vectorial de Elastic Search |
| `Taller GenAI Crea tu primera aplicación.pdf` | Presentación del taller (16 slides). |
| `docs/` | Carpeta de arquitectura  |
| &nbsp;&nbsp;└ `arquitectura_overview.jpg` | Arquitectura de la solucion. |
| `CloudRun/` | Carpeta con el microservicio listo para Docker ➜ Cloud Run |
| &nbsp;&nbsp;└ `app.py` | Código Python del agente (API `/agent`). |
| &nbsp;&nbsp;└ `Dockerfile` | Imagen multistage ligera. |
| &nbsp;&nbsp;└ `requirements.txt` | Dependencias congeladas. |

Plantillas FrontEnd (repos externos):

- Login con Google → <https://github.com/macespinoza/agentui-withlogin>  
- Sin login → <https://github.com/macespinoza/agentui-withoutlogin>

---

## 🗺️ Arquitectura

![Arquitectura](docs/arquitectura_overview.jpg)

1. **Usuario** interactúa desde el navegador (Vercel).  
2. **Cloud Run** recibe `/agent?msg=...&idagente=...` y activa LangChain.  
3. El **agente** consulta → Elasticsearch (conocimiento) + PostgreSQL (memoria).  
4. GPT‑4.1 genera la respuesta; LangSmith traza cada paso.  
5. La respuesta JSON vuelve a la UI en tiempo real.

