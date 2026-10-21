# Respuestas del workshop RAG
Escribe aquí tus respuestas con tus propias palabras. Pega tablas, resultados y enlaces a las capturas de `evidencias/`.

---

# Bloque 0 · Teoría

## T1 · Cuántos nodos viven en cada capa

_Evidencia:_ …
T1 · Nodos por capa (M = 16, N = 1.000.000)

P(capa ≥ k) = M^(−k), así que esperados = N / 16^k.

| Capa | Teórico (N/16^k) | Simulado (seed 0) |
|------|------------------|-------------------|
| ≥ 1  | 62.500           | 62.675            |
| ≥ 2  | 3.906            | 4.005             |
| ≥ 3  | 244              | 270               |

Salida de la simulación:

capa >= 1: simulado= 62,675  teórico=   62,500
capa >= 2: simulado=  4,005  teórico=    3,906
capa >= 3: simulado=    270  teórico=      244

_Mi respuesta / análisis:_

las capas altas son un grafo muy pequeño y de aristas largas, así que se cruza el espacio en pocos saltos hasta la zona correcta. Después, las capas bajas solo refinan localmente, evitando recorrer los N vectores.


## T2 · ef, k y el tamaño de la cola

Con k = 10 y ef = 5 la cola de la capa 0 solo guarda 5 candidatos; no hay de dónde sacar 10 vecinos, por lo que es imposible devolver k resultados.

Empezar con ef = 2·k; subirlo si el recall es insuficiente.

`ef`: se aplica en cada búsqueda y no cambia la estructura del grafo. `ef_construction` queda hecho en las aristas creadas al insertar, así que cambiarlo exige reconstruir el índice.

## T3 · Inserción frente a búsqueda

_Evidencia:_ …

_Mi respuesta / análisis:_


## T4 · El invariante de FAISSVectorStore

_Evidencia:_ …

_Mi respuesta / análisis:_


## T5 · El costo de la fuerza bruta

_Evidencia:_ …

_Mi respuesta / análisis:_


---

# Bloque A · App RAG empresarial

## A1 · Ejecuta la app RAG de punta a punta

_Evidencia:_ …

_Mi respuesta / análisis:_


## A2 · Tu propio corpus de documentos

¿Cuántos días de vacaciones tiene un empleado?
Según la política, todo empleado tiene derecho a 15 días hábiles de vacaciones remuneradas por cada año trabajado.

¿Quién aprueba un descuento del 20 %
Según la política de descuentos, un 20 % requiere la aprobación del gerente comercial y solo se otorgan a clientes mayoristas con contrato vigente.

Qué puntaje mínimo necesita un lote para salir a la venta

Un lote necesita al menos 82 puntos para salir a la venta.

## A3 · Ingeniería del prompt (antes y después)

_Evidencia:_ …

_Mi respuesta / análisis:_


## A4 · Muestra el score de similitud en las fuentes

_Evidencia:_ …

_Mi respuesta / análisis:_


---

# Bloque B · RAG sobre papers

## B1 · Ejecuta el RAG de papers y evalúa con el ground truth

_Evidencia:_ …

_Mi respuesta / análisis:_


## B2 · Experimentos con chunk_size, k y score_threshold

_Evidencia:_ …

_Mi respuesta / análisis:_


## B3 · Preguntas trampa: ¿el RAG admite que no sabe?

_Evidencia:_ …

_Mi respuesta / análisis:_


## B4 · Preguntas en español sobre papers en inglés

_Evidencia:_ …

_Mi respuesta / análisis:_


## B5 · Escribe tu propio quiz con ground truth

_Evidencia:_ …

_Mi respuesta / análisis:_


---

# Reflexión

## R1 · Reflexión final con tus datos

_Evidencia:_ …

_Mi respuesta / análisis:_


---

# Retos

## C1 · HNSW contra fuerza bruta con FAISS

_Evidencia:_ …

_Mi respuesta / análisis:_


## C2 · Un tercer paper: el RAG que explica HNSW

_Evidencia:_ …

_Mi respuesta / análisis:_


## C3 · App Gradio para los papers con filtro por paper

_Evidencia:_ …

_Mi respuesta / análisis:_


## C4 · Evaluación automática de la recuperación

_Evidencia:_ …

_Mi respuesta / análisis:_


## C5 · MMR frente a similitud pura

_Evidencia:_ …

_Mi respuesta / análisis:_


## C6 · Persistencia del índice en disco

_Evidencia:_ …

_Mi respuesta / análisis:_

