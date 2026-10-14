#En python construir un Grid 15x15 con 5 obstaculos, un inicio un final usando politica de dirección y metodo de valores
#para trazar el mejor camino entre inicio y fin. mostrarse avance con Reinforcement Learning
#- Mostrar las iteraciones y el valor de cada celda en cada iteración.

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. CONFIGURACIÓN DEL GRID
# ============================================================

FILAS = 15
COLUMNAS = 15

inicio = (0, 0)
final = (14, 14)

# 5 obstáculos
obstaculos = {
    (2, 3),
    (4, 7),
    (7, 5),
    (10, 10),
    (12, 4)
}

# ============================================================
# 2. ACCIONES DEL AGENTE
# ============================================================

acciones = {
    "↑": (-1, 0),   # Arriba
    "↓": (1, 0),    # Abajo
    "←": (0, -1),   # Izquierda
    "→": (0, 1)     # Derecha
}

# ============================================================
# 3. PARÁMETROS DEL REINFORCEMENT LEARNING
# ============================================================

gamma = 0.9       # Factor de descuento
theta = 0.001     # Criterio de convergencia
max_iteraciones = 100

# ============================================================
# 4. RECOMPENSAS
# ============================================================

def recompensa(siguiente_estado):

    # Llegar al objetivo
    if siguiente_estado == final:
        return 100

    # Chocar con un obstáculo
    if siguiente_estado in obstaculos:
        return -100

    # Cada movimiento tiene un pequeño costo
    return -1


# ============================================================
# 5. FUNCIÓN PARA OBTENER EL SIGUIENTE ESTADO
# ============================================================

def siguiente_estado(estado, movimiento):

    fila, columna = estado
    df, dc = movimiento

    nueva_fila = fila + df
    nueva_columna = columna + dc

    # Si sale del Grid, permanece en la misma celda
    if (nueva_fila < 0 or nueva_fila >= FILAS or
            nueva_columna < 0 or nueva_columna >= COLUMNAS):

        return estado

    # Si intenta entrar a un obstáculo,
    # permanece en la misma celda
    if (nueva_fila, nueva_columna) in obstaculos:
        return estado

    return (nueva_fila, nueva_columna)


# ============================================================
# 6. INICIALIZAR LOS VALORES
# ============================================================

valores = np.zeros((FILAS, COLUMNAS))

# El estado final tiene valor 100
valores[final] = 100


# ============================================================
# 7. VALUE ITERATION
# ============================================================

historial = []

for iteracion in range(max_iteraciones):

    nuevos_valores = valores.copy()

    max_cambio = 0

    for fila in range(FILAS):

        for columna in range(COLUMNAS):

            estado = (fila, columna)

            # Los obstáculos no tienen valor
            if estado in obstaculos:
                continue

            # El estado final conserva su valor
            if estado == final:
                continue

            valores_acciones = []

            # Evaluar las 4 direcciones
            for movimiento in acciones.values():

                siguiente = siguiente_estado(
                    estado,
                    movimiento
                )

                r = recompensa(siguiente)

                valor = r + gamma * valores[siguiente]

                valores_acciones.append(valor)

            # Elegir la mejor acción
            mejor_valor = max(valores_acciones)

            nuevos_valores[fila, columna] = mejor_valor

            cambio = abs(
                nuevos_valores[fila, columna]
                - valores[fila, columna]
            )

            max_cambio = max(max_cambio, cambio)

    valores = nuevos_valores

    # Guardar los valores de esta iteración
    historial.append(valores.copy())

    print("\n" + "=" * 60)
    print("ITERACIÓN:", iteracion + 1)
    print("=" * 60)

    np.set_printoptions(
        precision=2,
        suppress=True
    )

    print(valores)

    # Verificar convergencia
    if max_cambio < theta:

        print("\nEl algoritmo convergió en la iteración",
              iteracion + 1)

        break


# ============================================================
# 8. OBTENER LA POLÍTICA
# ============================================================

politica = np.empty((FILAS, COLUMNAS), dtype=object)

for fila in range(FILAS):

    for columna in range(COLUMNAS):

        estado = (fila, columna)

        if estado in obstaculos:
            politica[fila, columna] = "X"

        elif estado == final:
            politica[fila, columna] = "F"

        else:

            mejores_acciones = {}

            for nombre, movimiento in acciones.items():

                siguiente = siguiente_estado(
                    estado,
                    movimiento
                )

                r = recompensa(siguiente)

                valor = r + gamma * valores[siguiente]

                mejores_acciones[nombre] = valor

            mejor_accion = max(
                mejores_acciones,
                key=mejores_acciones.get
            )

            politica[fila, columna] = mejor_accion


# ============================================================
# 9. MOSTRAR LA POLÍTICA
# ============================================================

print("\n")
print("=" * 60)
print("POLÍTICA APRENDIDA")
print("=" * 60)

for fila in range(FILAS):

    print(" ".join(
        f"{politica[fila, columna]:>2}"
        for columna in range(COLUMNAS)
    ))


# ============================================================
# 10. TRAZAR EL MEJOR CAMINO
# ============================================================

camino = []

estado = inicio

camino.append(estado)

visitados = set()

while estado != final:

    # Evitar ciclos
    if estado in visitados:
        print("Se detectó un ciclo.")
        break

    visitados.add(estado)

    fila, columna = estado

    mejores_acciones = {}

    for nombre, movimiento in acciones.items():

        siguiente = siguiente_estado(
            estado,
            movimiento
        )

        r = recompensa(siguiente)

        valor = r + gamma * valores[siguiente]

        mejores_acciones[nombre] = valor

    mejor_accion = max(
        mejores_acciones,
        key=mejores_acciones.get
    )

    estado = siguiente_estado(
        estado,
        acciones[mejor_accion]
    )

    camino.append(estado)

    # Seguridad
    if len(camino) > FILAS * COLUMNAS:
        print("No se encontró un camino.")
        break


# ============================================================
# 11. MOSTRAR EL CAMINO
# ============================================================

print("\n")
print("=" * 60)
print("MEJOR CAMINO")
print("=" * 60)

for i, posicion in enumerate(camino):

    print(
        f"Paso {i + 1}: {posicion}"
    )

print("\nCantidad de movimientos:",
      len(camino) - 1)


# ============================================================
# 12. MOSTRAR EL GRID Y EL CAMINO
# ============================================================

plt.figure(figsize=(10, 10))

# Dibujar Grid
plt.xlim(-0.5, COLUMNAS - 0.5)
plt.ylim(FILAS - 0.5, -0.5)

plt.xticks(range(COLUMNAS))
plt.yticks(range(FILAS))

plt.grid(True)

# Dibujar obstáculos
for obstaculo in obstaculos:

    fila, columna = obstaculo

    plt.scatter(
        columna,
        fila,
        marker="s",
        s=500
    )

# Dibujar inicio
plt.scatter(
    inicio[1],
    inicio[0],
    marker="o",
    s=300,
    label="Inicio"
)

# Dibujar final
plt.scatter(
    final[1],
    final[0],
    marker="*",
    s=400,
    label="Final"
)

# Dibujar camino
if len(camino) > 1:

    x = [columna for fila, columna in camino]
    y = [fila for fila, columna in camino]

    plt.plot(
        x,
        y,
        linewidth=3,
        marker="o",
        markersize=5,
        label="Mejor camino"
    )

plt.title(
    "Reinforcement Learning - Value Iteration"
)

plt.xlabel("Columnas")
plt.ylabel("Filas")

plt.legend()

plt.show()


# ============================================================
# 13. MOSTRAR EVOLUCIÓN DE LOS VALORES
# ============================================================

# Seleccionar algunas iteraciones para visualizar
iteraciones_mostrar = [
    0,
    min(4, len(historial) - 1),
    min(9, len(historial) - 1),
    len(historial) - 1
]

for numero in iteraciones_mostrar:

    plt.figure(figsize=(8, 8))

    plt.imshow(
        historial[numero],
        interpolation="nearest"
    )

    plt.colorbar(
        label="Valor V(s)"
    )

    plt.title(
        f"Valores - Iteración {numero + 1}"
    )

    plt.xlabel("Columnas")
    plt.ylabel("Filas")

    # Mostrar obstáculos
    for fila, columna in obstaculos:

        plt.text(
            columna,
            fila,
            "X",
            ha="center",
            va="center"
        )

    plt.show()