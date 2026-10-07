# Changelog

## 1.1.0

**Respuestas más fiables** (tras la primera evaluación real, 73/100)
- El asistente lee la guía por su cuenta cuando la pregunta toca autenticación, cabeceras, límites o tablas de valores
  (`AGENT_GUIDE_PREFETCH`). Antes un modelo pequeño respondía de memoria (`Bearer` en lugar de `AccessToken=`).
- La búsqueda añade las palabras de la propia API a lo que escribe la persona (`RAG_QUERY_EXPANSION`): reiniciar → *reboot*,
  parpadear → *locate*, estadísticas → *statistic*, canal y potencia → *radio config*.
- Prompt más estricto: nunca inventar límites ni valores, decir «no existe en la API» en la primera frase, no confundir
  operaciones parecidas.

**Evaluación más justa**
- Las rutas dentro de código con variables (`f"{os.getenv(...)}"`) ya no cuentan como «inventadas».
- Los términos esperados ignoran acentos, mayúsculas y `_`/`-`, y admiten alternativas (`a|b`).
- El juez recibe los hechos reales de la especificación (campos obligatorios, permisos, errores); si es el mismo modelo
  que responde, su «incorrecto» solo es un aviso.
- En preguntas sobre códigos de error ya no se exige nombrar el endpoint.
- Nueva medición de búsqueda con palabras de la vida real (`eval --natural`) y con las preguntas curadas; la nota global usa
  la más realista.

**Producción**
- Registro de uso en SQLite y límites de gasto por día, por mes, por persona y por número de preguntas.
- Identidad por cabecera de un proxy con login (`PORTAL_USER_HEADER`), administradores, `/api/usage`, `manage.py usage`.
- Logs estructurados (JSON), identificador de petición, archivo con rotación; el texto de las preguntas no se registra.
- Costo y gasto del día visibles bajo cada respuesta.

## 1.0.0

Primera versión: RAG híbrido, agente, portal, guía de acceso, evaluaciones.
