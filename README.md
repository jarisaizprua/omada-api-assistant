# Omada API Assistant

Asistente de IA con interfaz web para **consultar la API abierta de TP-Link Omada** en lenguaje natural.
Indexa la especificación OpenAPI de tu controller en un RAG local y un agente de OpenAI la consulta para
darte el endpoint exacto, sus parámetros, permisos, códigos de error y ejemplos de código.

[English version](README.en.md)

## Qué hace

- **Responde con datos de la especificación**, no de memoria: el modelo busca en el índice y lee el detalle de cada operación antes de contestar.
- **Búsqueda híbrida**: vectores (significado) + palabras clave BM25 (identificadores exactos como `createNewSite` o `-33101`), fusionados con Reciprocal Rank Fusion.
- **Portal web**: chat con streaming, pasos del agente visibles, panel de endpoints consultados, buscador `Ctrl+K`, mapa interactivo de categorías, tema claro/oscuro e historial local.
- **Sin servidores extra**: SQLite y ChromaDB funcionan dentro del proceso. Solo Python.

## Requisitos

- Python 3.10 o superior
- Una API key de OpenAI
- La especificación OpenAPI de tu controller (`.json` o `.zip`)

## Instalación

```bash
git clone <URL-de-tu-repo>
cd omada-api-assistant

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env_template .env              # Windows: copy .env_template .env
# edita .env y pon tu OPENAI_API_KEY
```

## Uso en 3 pasos

**1. Coloca la especificación** en `data/spec/` tal como la descargaste (`.json` o `.zip`).
Se obtiene desde tu controller: *Settings > Platform Integration > Open API > Online API Document*.

**2. Construye el RAG**

```bash
python manage.py build-index
```

Tarda segundos. Re-ejecutarlo es barato: solo se re-embeben las operaciones que cambiaron.
El costo de embeddings de unas 2.400 operaciones es de centavos de dólar con `text-embedding-3-small`.

**3. Inicia el portal**

```bash
python manage.py serve
```

Abre <http://localhost:8000>.

## Comandos

| Comando | Para qué sirve |
|---|---|
| `python manage.py build-index` | Construye o actualiza el RAG. `--full` re-embebe todo; `--keyword-only` no usa API key. |
| `python manage.py import-guide ARCHIVO` | Añade una guía (texto o Markdown) al RAG. Luego ejecuta `build-index`. |
| `python manage.py serve` | Inicia el portal. `--reload` para desarrollo. |
| `python manage.py search "crear sitio"` | Prueba la búsqueda desde la terminal. |
| `python manage.py eval` | Mide la calidad de búsqueda y respuestas y genera un informe (ver más abajo). |
| `python manage.py stats` | Muestra qué hay indexado. |
| `python manage.py usage` | Informe de gasto y uso: quién preguntó, cuánto costó, cuánto queda (`--days 7`, `--user correo`). |
| `python manage.py doctor` | Diagnostica tu configuración, el precio del modelo y los límites de gasto. |

## Configuración (`.env`)

Todo se configura en `.env`; `.env_template` documenta cada variable. Las principales:

| Variable | Descripción |
|---|---|
| `OPENAI_API_KEY` | Tu clave (obligatoria). |
| `OPENAI_MODEL` | Modelo del chat (Responses API con function calling). Verifica cuáles tiene tu cuenta. |
| `OPENAI_REASONING_EFFORT` | Solo para modelos de razonamiento. Vacío con los que no razonan. |
| `OPENAI_EMBEDDING_MODEL` | Si lo cambias, ejecuta `build-index --full`. |
| `PORTAL_ACCESS_TOKEN` | Protege el portal con un token (recomendado si usas `HOST=0.0.0.0`). |
| `BUDGET_DAILY_USD`, `BUDGET_MONTHLY_USD` | Tope de gasto de todos juntos, por día y por mes (0 = sin tope). Ver *Uso en producción*. |
| `BUDGET_USER_DAILY_USD`, `USER_DAILY_REQUESTS` | Tope por persona y por día, en dólares y en número de preguntas. |
| `PORTAL_USER_HEADER` | Cabecera con el correo del usuario cuando hay un login delante (por ejemplo `X-Forwarded-Email`). |
| `LOG_FORMAT`, `LOG_FILE` | `json` para servidores; archivo con rotación opcional. |
| `AGENT_GUIDE_PREFETCH`, `RAG_QUERY_EXPANSION` | Mejoras de calidad (activadas por defecto): lectura automática de la guía y vocabulario de la API en la búsqueda. |

## Autenticación y guías (importante)

La especificación **no incluye** el flujo de autenticación (tokens, modos de acceso) ni tablas como las
zonas horarias. Esa información está en la guía *Open API Access Guide* de la documentación de Omada.
Para que el asistente la use:

1. Copia el texto de la guía a un archivo, por ejemplo `access_guide.txt` (no hace falta darle formato).
2. Impórtala y reconstruye el índice:

```bash
python manage.py import-guide access_guide.txt --name organization_access_guide
python manage.py build-index
```

`import-guide` convierte los títulos numerados (`3.2.2 Obtain Access Token`) en secciones, deja las tablas
legibles y encierra los ejemplos `curl`/JSON en bloques de código, para que cada sección se indexe completa.
Puedes importar varias guías. Sin ellas, el asistente dirá que no tiene esa información en lugar de inventarla.

## Uso en producción: gasto, usuarios y logs

Pensado para un equipo pequeño o mediano (decenas de personas) en un solo servidor, sin Docker.

**Límites de gasto.** Cada pregunta se registra (quién, tokens, costo estimado, tiempo) en una base SQLite local
(`data/usage/usage.db`) que sobrevive a los reinicios. *Antes* de responder se comprueban cuatro topes; si alguno se
alcanzó, el usuario ve un mensaje claro en lugar de gastar más:

| Tope | Variable (por defecto) |
|---|---|
| Todos juntos, por día | `BUDGET_DAILY_USD` (5) |
| Todos juntos, por mes | `BUDGET_MONTHLY_USD` (50) |
| Cada persona, por día | `BUDGET_USER_DAILY_USD` (1) |
| Preguntas por persona y día | `USER_DAILY_REQUESTS` (200) |

`0` desactiva un tope. Al llegar al 80 % el usuario recibe un aviso y bajo cada respuesta ve su costo
aproximado y lo gastado hoy. Detalles que conviene saber: el costo se calcula con el precio del modelo
(lista conocida en `usage/pricing.py`; si tu modelo no está, define `PRICE_INPUT_PER_MTOK` y `PRICE_OUTPUT_PER_MTOK`;
sin precio, los topes en dólares no pueden aplicarse y `doctor` y `/api/health` lo avisan, pero el tope de preguntas sí
funciona); una respuesta puede pasarse del tope por su propio costo (centavos); no se cuenta el costo (mínimo) de los
embeddings de la consulta; y el «día» empieza a medianoche en `USAGE_TIMEZONE` (por ejemplo `America/Guayaquil`).

```bash
python manage.py usage                      # resumen: hoy, mes, por día, quién usa más
python manage.py usage --days 7 --user ana@empresa.com
```

**Quién pregunta.** Sin login, las personas se distinguen por IP. Con un proxy que autentica (Cloudflare Access,
oauth2-proxy…), define `PORTAL_USER_HEADER=X-Forwarded-Email` y `PORTAL_BEHIND_PROXY=true`: los límites y el registro
pasan a ser por persona. **Actívalo solo si el proxy pone esa cabecera y borra la que envíe el cliente**; si no,
cualquiera podría hacerse pasar por otro. `PORTAL_ADMINS` limita quién ve el uso de todos (`/api/usage`).
Si publicas el portal fuera de tu red, ponlo detrás de HTTPS y de un login.

**Logs.** `LOG_FORMAT=json` escribe una línea JSON por evento (cada petición y cada pregunta, con identificador de
petición `X-Request-ID`, usuario, tokens, costo y tiempo), lista para journald, Loki, CloudWatch o ELK. `LOG_FILE`
añade un archivo con rotación. **El texto de las preguntas no se guarda** (solo su tamaño) a menos que pongas
`LOG_QUESTIONS=true`.

**Qué no cubre todavía.** El límite de peticiones por minuto es en memoria y por proceso (ejecuta un solo proceso o
aplícalo en el proxy); el índice vectorial es local; hay un solo índice, de una sola versión de la especificación.

## Evaluaciones: ¿qué tan bien responde?

```bash
python manage.py eval              # evaluación completa (búsqueda + 56 preguntas)
python manage.py eval --only retrieval --keyword-only   # solo la búsqueda, gratis y sin clave
python manage.py eval --check      # valida las preguntas contra tu índice (gratis)
```

Genera un informe en lenguaje claro en `evals/results/<fecha>/report.md` (y los datos en `results.json`).
Mide dos cosas:

1. **La búsqueda** (gratis o centavos): toma 300 operaciones al azar, formula la pregunta como lo haría un
   desarrollador y mide si la operación correcta aparece en el 1.er lugar, entre las 3 primeras o entre las 8 primeras.
   Se mide de tres maneras: preguntas «de manual» (repiten las palabras de la documentación, por eso son optimistas),
   **palabras de la vida real** («reiniciar» en vez de *reboot*; se generan una vez con
   `eval --only retrieval --natural` y se guardan en `evals/natural_queries.json`) y tus preguntas curadas. La nota usa
   las de la vida real cuando existen.
2. **Las respuestas**: hace 56 preguntas reales al asistente (español e inglés; fáciles y difíciles; incluye errores,
   tareas de varios pasos, autenticación y **4 cosas que la API no hace**, para comprobar que no inventa).
   Cada respuesta se califica con reglas automáticas (¿nombró el endpoint correcto? ¿incluyó los datos clave?
   ¿todas las rutas existen?) y con un segundo modelo que actúa de juez.

El informe empieza con una **nota global de 0 a 100** y un resumen de cinco líneas, y sigue con tablas por tipo,
dificultad y tema, un apartado de **fiabilidad** (rutas inventadas), velocidad y costo, el detalle de **cada pregunta
que falló** con la razón en palabras sencillas, y recomendaciones concretas. Con `--baseline latest` compara contra
la evaluación anterior (útil al cambiar de modelo o de ajustes). Las 10 preguntas sobre autenticación y datos de
referencia solo se evalúan cuando has importado la guía de acceso; si no, se omiten y el informe lo indica.

**Costo orientativo:** unos pocos centavos con `gpt-4o-mini`; varios dólares con un modelo grande. Puedes
reducirlo con `--limit 10`, `--no-judge` o `--topic Sitios`.

**Sobre el juez.** Si el juez es el mismo modelo que responde, sus «incorrecto» solo cuentan como aviso (un modelo
pequeño no sabe más que sí mismo y suele marcar como error respuestas correctas). Para una medición firme define
`OPENAI_JUDGE_MODEL` con un modelo más capaz. El juez recibe además los datos reales de la especificación (campos
obligatorios, permisos, códigos de error), así que ya no depende de notas escritas a mano que pueden estar incompletas.
Las reglas automáticas entienden el código de ejemplo (`f"{BASE}/openapi/v1/{os.getenv('ID')}/..."` no es una ruta
inventada), aceptan sinónimos y acentos, y en las preguntas sobre códigos de error no exigen nombrar el endpoint.

**Tus propias preguntas:** edita `evals/questions.json` (cada pregunta indica qué operación debe nombrar y qué datos
debe incluir) y ejecuta `eval --check` para verificar que lo esperado existe en tu especificación.

## Arquitectura

```
spec (.json/.zip) ─┐
                   ├─> parser ─┬─> SQLite + FTS5 (palabras clave, detalle completo)
guías (*.md) ──────┘           └─> ChromaDB (vectores, embeddings de OpenAI)

Navegador ──SSE──> FastAPI ──> Agente (Responses API) ──herramientas──> Búsqueda híbrida
```

- Un documento por operación; el texto largo (esquemas completos) se sirve bajo demanda con `get_operation`, así el contexto del modelo se mantiene pequeño.
- El servidor **no guarda sesiones**: el navegador envía el historial en cada petición, por lo que escala horizontalmente.
- Si fallan los embeddings, la búsqueda cae a palabras clave en lugar de romperse.

```
src/omada_agent/
  config.py        configuración (.env)
  ingestion/       lectura del spec, parser, guías, indexador
  retrieval/       SQLite/FTS5, ChromaDB, embeddings, búsqueda híbrida
  agent/           prompt, herramientas, bucle del agente
  api/             FastAPI, SSE, límite de peticiones, identidad del usuario
  usage/           registro de uso (SQLite), precios y límites de gasto
  evals/           evaluaciones: búsqueda, respuestas, juez, informe
  web/             interfaz (HTML, CSS, JS sin build)
evals/questions.json   las 56 preguntas de evaluación
tests/             pruebas (no necesitan red ni API key)
```

## Pruebas

```bash
pip install -r requirements-dev.txt
pytest
```

## Privacidad y límites

- El registro de uso guarda quién preguntó, tokens, costo y tiempos, pero **no** el texto de las preguntas ni de las respuestas.
- Las preguntas y los fragmentos de documentación se envían a OpenAI. La Responses API conserva las respuestas según la política de tu cuenta.
- Las respuestas son generadas por IA: verifica los detalles críticos antes de ejecutar cambios en producción.
- Para uso en red, define `PORTAL_ACCESS_TOKEN` y publícalo detrás de HTTPS (por ejemplo, un proxy inverso). El límite de peticiones por minuto es en memoria y por proceso.
- Detén el servidor antes de reconstruir el índice (en Windows el archivo puede estar bloqueado).

## Licencia

MIT. Ver `LICENSE` y `THIRD_PARTY_NOTICES.md`. Proyecto independiente, no afiliado a TP-Link; la especificación no se incluye en este repositorio.
