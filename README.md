# Proyecto Asistente conversacional

Primer Proyecto de creación de asistente conversacional con python. Utilizado en el curso de Desarrollo de asistente conversacionales del Sillicon Misiones

## Cómo usarlo

Ejecuta el archivo "streamapp.py" en consola en el entorno virtual creado:

```python
streamlit run streamapp.py
```

En la página que ejecuta, debe adjuntar el/los archivos pdf que quiere evaluar. Cuando termine de cargar, directamente indique el tiempo en el que quiere crear el plan de estudio, el asistente conversacional se encargará del resto. En caso que requiera información adicional, el asistente le preguntará directamente.

El resultado es el plan de estudio segpun sus capacidades.

## Acerca de la solución

Para obtener la respuesta adecuada, se utiliza el LLM "llama3-8b-8192" desarrollado por Meta, a través del motor de Groq. Como base de datos vectorial se utiliza FAISS (Facebook AI Similarity Search), y como Embedding el de Cohere. La ventaja de estas tecnologías usadas es que son de uso libre hasta la fecha de creación del modelo. Cabe mencionar que es importante que el usuario de este modelo debe tener su propia API_KEY para poder correrlo.
