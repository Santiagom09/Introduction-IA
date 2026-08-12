# Solución del laboratorio — Robot de entregas (MDP)

Este archivo resume la solución del notebook y contiene el código principal usado para el MDP, Value Iteration y Policy Iteration.

## 1) Disposición del grid (inferida)

- Tamaño: 5 filas × 6 columnas (índices `(fila, col)`, 0-index).
- `START = (0,0)`.
- Estanterías / paredes (no transitables): `(0,3)`, `(1,1)`, `(2,4)`, `(4,2)`, `(0,2)`.
- Piso resbaloso: `(1,2)`, `(2,1)`, `(3,3)`.
- Terminales:
  - Entrega `+10`: `(0,5)` (terminal)
  - Carga `+2`: `(2,2)` (terminal)
  - Peligro mortal `-10`: `(3,5)` (terminal)
- Peligros (no terminales, recompensa `-3`): `(1,4)`, `(4,1)`.
- Recompensa por paso (living reward): `-1.0` (por defecto).

> Nota: si la disposición no coincide exactamente con la imagen original, ajusta las coordenadas en el notebook `03_lab_warehouse_mdp_estudiantes.ipynb`.

## 2) Código principal (extracto)

WarehouseMDP (modelo, transiciones y recompensas):

```python
import numpy as np

class WarehouseMDP:
    def __init__(self):
        self.height = 5
        self.width = 6
        self.start = (0, 0)
        self.walls = {(0,3),(1,1),(2,4),(4,2),(0,2)}
        self.slippery_states = {(1,2),(2,1),(3,3)}
        self.terminal_states = {(0,5):10.0,(2,2):2.0,(3,5):-10.0}
        self.danger_states = {(1,4):-3.0,(4,1):-3.0}
        self.living_reward = -1.0
        self.gamma = 0.9
        self.actions = [(-1,0),(1,0),(0,-1),(0,1)]
        self.left_map = {0:2,1:3,2:1,3:0}
        self.right_map = {0:3,1:2,2:0,3:1}

    def is_valid_state(self, state):
        r,c = state
        if r<0 or r>=self.height or c<0 or c>=self.width: return False
        if state in self.walls: return False
        return True

    def states(self):
        S=[]
        for r in range(self.height):
            for c in range(self.width):
                s=(r,c)
                if self.is_valid_state(s): S.append(s)
        return S

    def is_terminal(self,state):
        return state in self.terminal_states

    def get_reward(self,state):
        if state in self.terminal_states: return self.terminal_states[state]
        if state in self.danger_states: return self.danger_states[state]
        return self.living_reward

    def get_transition_probs(self,state,action):
        if not self.is_valid_state(state): return [(state,1.0)]
        if self.is_terminal(state): return [(state,1.0)]
        a_idx = self.actions.index(action)
        if state in self.slippery_states:
            p_intended,p_left,p_right = 0.6,0.2,0.2
        else:
            p_intended,p_left,p_right = 0.9,0.05,0.05
        def move(s,act):
            nr,nc = s[0]+act[0], s[1]+act[1]
            ns=(nr,nc)
            if not self.is_valid_state(ns): return s
            return ns
        intended = move(state,self.actions[a_idx])
        left = move(state,self.actions[self.left_map[a_idx]])
        right = move(state,self.actions[self.right_map[a_idx]])
        probs = {}
        probs[intended]=probs.get(intended,0.0)+p_intended
        probs[left]=probs.get(left,0.0)+p_left
        probs[right]=probs.get(right,0.0)+p_right
        return list(probs.items())
```

Funciones de solución (Value Iteration / Policy Iteration):

```python
def expected_next_value(grid,state,action,V):
    ev=0.0
    for ns,p in grid.get_transition_probs(state,action): ev += p * V.get(ns,0.0)
    return ev

def value_iteration(grid, threshold=1e-4, max_iter=10000):
    S = grid.states()
    V = {s:0.0 for s in S}
    for t,r in grid.terminal_states.items():
        if t in V: V[t]=r
    for it in range(1, max_iter+1):
        delta=0.0
        V_new = V.copy()
        for s in S:
            if grid.is_terminal(s):
                V_new[s]=grid.get_reward(s)
                continue
            best=-float('inf')
            for a in grid.actions:
                val = expected_next_value(grid,s,a,V)
                if val>best: best=val
            V_new[s]=grid.get_reward(s) + grid.gamma * best
            delta = max(delta, abs(V_new[s]-V[s]))
        V=V_new
        if delta < threshold: return V, it
    return V, max_iter

def extract_policy(grid,V):
    policy = {}
    for s in grid.states():
        if grid.is_terminal(s): continue
        best_a=None; best_val=-float('inf')
        for a in grid.actions:
            val = expected_next_value(grid,s,a,V)
            if val>best_val: best_val=val; best_a=a
        policy[s]=best_a
    return policy

def policy_evaluation(grid,policy,threshold=1e-4,max_iter=10000):
    S=grid.states(); V={s:0.0 for s in S}
    for t,r in grid.terminal_states.items():
        if t in V: V[t]=r
    for it in range(1,max_iter+1):
        delta=0.0
        for s in S:
            if grid.is_terminal(s): continue
            a = policy[s]
            v_new = grid.get_reward(s) + grid.gamma * expected_next_value(grid,s,a,V)
            delta = max(delta, abs(v_new - V[s]))
            V[s] = v_new
        if delta < threshold: return V
    return V

def policy_improvement(grid,V):
    policy={}
    for s in grid.states():
        if grid.is_terminal(s): continue
        best_a=None; best_val=-float('inf')
        for a in grid.actions:
            val = expected_next_value(grid,s,a,V)
            if val>best_val: best_val=val; best_a=a
        policy[s]=best_a
    return policy

def policy_iteration(grid,threshold=1e-4,max_iter=100):
    policy = {s:grid.actions[0] for s in grid.states() if not grid.is_terminal(s)}
    history=[]
    for it in range(max_iter):
        V = policy_evaluation(grid,policy,threshold=threshold)
        new_policy = policy_improvement(grid,V)
        history.append(new_policy)
        if all(new_policy.get(s)==policy.get(s) for s in new_policy):
            return new_policy, V, history
        policy = new_policy
    return policy, V, history
```

## 3) Experimentos realizados

- Experimento A: `living_reward = -0.1` (menos penalización por paso) — normalmente favorece rutas más largas hacia la recompensa grande.
- Experimento B: piso resbaloso aumentado a `0.40/0.30/0.30` — aumenta el riesgo y puede cambiar la política para evitar zonas resbalosas.
- Experimento C: `gamma = 0.99` (más paciencia) — aumenta el peso de recompensas lejanas, favoreciendo la `+10` si el riesgo no es prohibitivo.

Incluí en el notebook celdas que ejecutan estos experimentos y una búsqueda aproximada del umbral de `living_reward` donde la política desde `START` cambia.

## 4) Cómo reproducir

1. Abrir el notebook: `notebooks/lecture4/03_lab_warehouse_mdp_estudiantes.ipynb`.
2. Ejecutar todas las celdas.
3. Para ajustar la disposición, editar las coordenadas en la clase `WarehouseMDP` dentro del notebook o en este archivo.

---
Archivo generado automáticamente por el asistente.
