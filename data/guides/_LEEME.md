# Guías (opcional)

La especificación OpenAPI describe los endpoints, pero **no** el flujo de autenticación
(tokens, modos de acceso, tablas de valores como zonas horarias). Eso está en la guía
"Open API Access Guide" de la documentación de Omada.

Para añadirla:

1. Copia su texto a un archivo, p. ej. `access_guide.txt`.
2. `python manage.py import-guide access_guide.txt --name organization_access_guide`
3. `python manage.py build-index`

El comando guarda un `.md` en esta carpeta. Los archivos que empiezan por `_` o `README` se
ignoran. El contenido de esta carpeta no se sube a GitHub (puede tener derechos de autor).
