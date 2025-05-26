from typing import List, Optional, Tuple
import streamlit as st
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import math
import logging

def validate_matrices(dist_m: List[List[float]], time_m: List[List[int]], n: int) -> None:
    """Validate distance and time matrices."""
    print("Validando matrices...")
    if not isinstance(dist_m, list) or not isinstance(time_m, list):
        raise ValueError(f"Distance or time matrix is not a list: dist_m={type(dist_m)}, time_m={type(time_m)}")
    if not (len(dist_m) == n and all(isinstance(row, list) and len(row) == n for row in dist_m)):
        raise ValueError(f"Distance matrix must be {n}x{n}. Got {len(dist_m)}x{len(dist_m[0]) if dist_m else 0}.")
    if not (len(time_m) == n and all(isinstance(row, list) and len(row) == n for row in time_m)):
        raise ValueError(f"Time matrix must be {n}x{n}. Got {len(time_m)}x{len(time_m[0]) if time_m else 0}.")
    if any(dist < 0 for row in dist_m for dist in row):
        raise ValueError("Distance matrix contains negative values.")
    if any(time < 0 for row in time_m for time in row):
        raise ValueError("Time matrix contains negative values.")
    print("Matrices validadas correctamente.")

def build_cost_matrix(
    dist_m: List[List[float]],      # metros
    time_m: List[List[int]],        # minutos
    fuel_price: float,              # €/litro
    fuel_consumption: float,        # litros/100 km
    driver_price_h: float           # €/hora
) -> List[List[int]]:
    """Devuelve matriz de coste en céntimos (int) para evitar floats."""
    print("Construyendo matriz de costos...")
    n = len(dist_m)
    cost = [[0]*n for _ in range(n)]
    c_km = fuel_price * fuel_consumption / 100          # €/km
    print(f"Costo por km: {c_km} €/km")
    for i in range(n):
        for j in range(n):
            km = dist_m[i][j] / 1_000                 # metros → km
            hrs = time_m[i][j] / 60                    # minutos → horas
            euro = km * c_km + hrs * driver_price_h
            cost[i][j] = int(round(euro * 100))           # céntimos
    print("Matriz de costos construida correctamente.")
    return cost

def build_routing_model(
    dist_m: List[List[float]] = None,
    time_m: List[List[int]] = None,
    arc_cost_m: List[List[int]] = None,
    vehicles: int = 1,
    depot: int = 0,
    service_time: int = 0,
    start_min: int = 0,
    balance: bool = True,
    max_stops_per_vehicle: Optional[int] = None,
    balance_threshold: float = 0.9,
    price_per_hour: float = 0.0,
    max_minutes: int = 1440,
    cost_mode: bool = False,
    time_windows: List[Tuple[int, int]] = None
) -> Tuple[pywrapcp.RoutingIndexManager, pywrapcp.RoutingModel, str]:
    """Create and return (manager, routing, time_dim) for VRP."""
    print("Construyendo modelo de ruteo...")
    n = len(dist_m or arc_cost_m)
    
    # Validar argumentos
    if vehicles < 1:
        print("Error: El número de vehículos debe ser positivo.")
        raise ValueError("Number of vehicles must be positive.")
    if not isinstance(depot, int) or depot < 0 or depot >= n:
        print(f"Error: Índice del depósito {depot} fuera de rango [0, {n-1}].")
        raise ValueError(f"Depot index {depot} out of range [0, {n-1}].")
    if service_time < 0:
        print("Error: El tiempo de servicio no puede ser negativo.")
        raise ValueError("Service time cannot be negative.")
    if n < 1:
        print("Error: El número de nodos debe ser positivo.")
        raise ValueError("Number of nodes must be positive.")
    
    if cost_mode:
        print("Modo de costo activado, validando arc_cost_m y time_m...")
        validate_matrices(arc_cost_m, time_m, n)
    else:
        print("Modo de distancia activado, validando dist_m y time_m...")
        validate_matrices(dist_m, time_m, n)

    if time_windows and len(time_windows) != n:
        print(f"Error: time_windows debe tener longitud {n}, pero tiene {len(time_windows)}.")
        raise ValueError(f"time_windows must have length {n}, got {len(time_windows)}")

    print(f"Instanciando RoutingIndexManager: n={n}, vehicles={vehicles}, depot={depot}")
    man = pywrapcp.RoutingIndexManager(n, vehicles, depot)
    rout = pywrapcp.RoutingModel(man)

    # Cost callback
    if cost_mode:
        print("Registrando callback de costo...")
        dist_cb = rout.RegisterTransitCallback(
            lambda i, j: arc_cost_m[man.IndexToNode(i)][man.IndexToNode(j)]
        )
    else:
        max_dist = max(max(row) for row in dist_m) or 1
        print(f"Max distancia: {max_dist}")
        dist_cb = rout.RegisterTransitCallback(
            lambda i, j: int(dist_m[man.IndexToNode(i)][man.IndexToNode(j)] / max_dist * 1000)
        )
    rout.SetArcCostEvaluatorOfAllVehicles(dist_cb)

    # Fixed vehicle cost
    if price_per_hour > 0:
        fixed_cents = int(round(2 * price_per_hour * 100))  # 2 hours base
        print(f"Costo fijo por vehículo: {fixed_cents} céntimos")
        for v in range(vehicles):
            rout.SetFixedCostOfVehicle(fixed_cents, v)

    # Time dimension
    max_time = sum(max(row) for row in time_m) + service_time * n
    print(f"Max tiempo: {max_time} minutos")
    time_cb = rout.RegisterTransitCallback(
        lambda i, j: time_m[man.IndexToNode(i)][man.IndexToNode(j)] + (
            service_time if man.IndexToNode(j) != depot else 0
        )
    )
    rout.AddDimension(time_cb, 0, max(max_minutes, max_time), False, "Time")
    time_dim = rout.GetDimensionOrDie("Time")

    # Apply time windows
    if time_windows:
        print("Aplicando ventanas de tiempo...")
        for node in range(n):
            idx = man.NodeToIndex(node)
            if idx < 0 or idx >= man.GetNumberOfIndices():
                continue
            start, end = time_windows[node]
            time_dim.CumulVar(idx).SetRange(start, end)
            print(f"Nodo {node}: ventana de tiempo [{start}, {end}] minutos")

    # Apply maximum time limit per vehicle
    for v in range(vehicles):
        start = rout.Start(v)
        end = rout.End(v)
        time_dim.CumulVar(start).SetRange(start_min, start_min)
        time_dim.CumulVar(end).SetRange(start_min, start_min + max_minutes)

    # Stop count dimension
    if max_stops_per_vehicle is not None and vehicles > 1 and max_stops_per_vehicle < (n - 1):
        if max_stops_per_vehicle < 1:
            print("Error: El máximo de paradas por vehículo debe ser positivo.")
            raise ValueError("Max stops per vehicle must be positive.")
        demand_cb = rout.RegisterUnaryTransitCallback(
            lambda i: 1 if man.IndexToNode(i) != depot else 0
        )
        rout.AddDimension(demand_cb, 0, max_stops_per_vehicle, True, "Stops")

    # Balance dimension
    if balance and vehicles > 1:
        rout.AddDimension(dist_cb, 0, 1_000_000, True, "Cost" if cost_mode else "Distance")
        rout.GetDimensionOrDie("Cost" if cost_mode else "Distance").SetGlobalSpanCostCoefficient(100)
        if balance_threshold > 0:
            balance_cb = rout.RegisterUnaryTransitCallback(
                lambda i: 1 if man.IndexToNode(i) != depot else 0
            )
            max_stops = max(1, int(n * balance_threshold / vehicles) + 2)
            rout.AddDimensionWithVehicleCapacity(
                balance_cb, 0, [max_stops] * vehicles, True, "Balance"
            )

    print("Modelo de ruteo construido correctamente.")
    return man, rout, time_dim

def solve_vrp_simple(
    dist_m: List[List[float]],
    time_m: List[List[int]],
    vehicles: int,
    depot: int,
    balance: bool,
    start_min: int,
    service_time: int,
    time_windows: List[Tuple[int, int]] = None,
    max_stops_per_vehicle: Optional[int] = None,
    balance_threshold: float = 0.9,
    predefined_routes: Optional[List[List[int]]] = None,
    respect_predefined: bool = False,
    fuel_price: float = 0.0,
    fuel_consumption: float = 0.0,
    price_per_hour: float = 0.0,
    max_minutes: int = 1440,
    cost_mode: bool = False
) -> Tuple[List[List[int]], List[Optional[int]], int]:
    """Solve VRP and return routes, ETAs, and used vehicles."""
    print("Iniciando solve_vrp_simple...")
    print(f"Parámetros: vehicles={vehicles}, depot={depot}, balance={balance}, start_min={start_min}, service_time={service_time}, cost_mode={cost_mode}")
    n = len(dist_m)
    validate_matrices(dist_m, time_m, n)

    if not isinstance(depot, int) or depot < 0 or depot >= n:
        raise ValueError(f"Depot index {depot} out of range [0, {n-1}]")
    if vehicles < 1:
        raise ValueError("Number of vehicles must be positive")

    if time_windows and len(time_windows) != n:
        raise ValueError(f"time_windows must have length {n}, got {len(time_windows)}")

    if respect_predefined and predefined_routes:
        print("Respetando rutas predefinidas...")
        vehicles = len(predefined_routes)
        if vehicles == 0:
            st.warning("No se encontraron rutas predefinidas válidas.")
            print("No se encontraron rutas predefinidas válidas.")
            return [], [None] * n, 0
        routes = []
        eta = [None] * n
        used_vehicles = 0
        for route in predefined_routes:
            if not route:
                routes.append([depot])
                continue
            if not all(0 <= idx < n for idx in route):
                st.warning(f"Ruta predefinida inválida: {route}. Ignorando.")
                print(f"Ruta predefinida inválida: {route}")
                routes.append([depot])
                continue
            sub_nodes = [depot] + route + [depot]
            sub_indices = list(range(len(sub_nodes)))
            try:
                sub_dist_m = [[dist_m[i][j] for j in sub_nodes] for i in sub_nodes]
                sub_time_m = [[time_m[i][j] for j in sub_nodes] for i in sub_nodes]
                validate_matrices(sub_dist_m, sub_time_m, len(sub_nodes))
                sub_time_windows = [time_windows[i] if time_windows else (0, 1440) for i in sub_nodes] if time_windows else None
            except (IndexError, ValueError) as e:
                st.warning(f"Error generando submatrices para ruta {route}: {e}. Ignorando.")
                print(f"Error generando submatrices para ruta {route}: {e}")
                routes.append([depot])
                continue
            man, rout, time_dim = build_routing_model(
                dist_m=sub_dist_m,
                time_m=sub_time_m,
                vehicles=1,
                depot=0,
                service_time=service_time,
                start_min=start_min,
                balance=False,
                max_stops_per_vehicle=None,
                balance_threshold=0.0,
                price_per_hour=price_per_hour,
                max_minutes=max_minutes,
                cost_mode=cost_mode,
                time_windows=sub_time_windows
            )
            time_dim.CumulVar(rout.Start(0)).SetRange(start_min, start_min)
            prm = pywrapcp.DefaultRoutingSearchParameters()
            prm.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
            prm.time_limit.seconds = 30
            sol = rout.SolveWithParameters(prm)
            if sol is None:
                st.warning(f"No se encontró solución para ruta predefinida {route}.")
                print(f"No se encontró solución para ruta predefinida {route}")
                routes.append([depot])
                continue
            idx = rout.Start(0)
            sub_route = []
            while not rout.IsEnd(idx):
                node = man.IndexToNode(idx)
                sub_route.append(sub_nodes[node])
                if sub_nodes[node] != depot:
                    eta[sub_nodes[node]] = sol.Value(time_dim.CumulVar(idx))
                idx = sol.Value(rout.NextVar(idx))
            sub_route.append(depot)
            routes.append(sub_route)
            used_vehicles += 1
        while len(routes) < vehicles:
            routes.append([depot])
        return routes, eta, used_vehicles
    else:
        arc_cost_m = None
        if cost_mode:
            print("Modo de costo activado, construyendo matriz de costos...")
            arc_cost_m = build_cost_matrix(dist_m, time_m, fuel_price, fuel_consumption, price_per_hour)
        man, rout, time_dim = build_routing_model(
            dist_m=dist_m,
            time_m=time_m,
            arc_cost_m=arc_cost_m,
            vehicles=vehicles,
            depot=depot,
            service_time=service_time,
            start_min=start_min,
            balance=balance,
            max_stops_per_vehicle=max_stops_per_vehicle,
            balance_threshold=balance_threshold,
            price_per_hour=price_per_hour,
            max_minutes=max_minutes,
            cost_mode=cost_mode,
            time_windows=time_windows
        )
        for v in range(vehicles):
            idx = rout.Start(v)
            time_dim.CumulVar(idx).SetRange(start_min, start_min)
        prm = pywrapcp.DefaultRoutingSearchParameters()
        prm.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC
        prm.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        prm.time_limit.seconds = 60
        sol = rout.SolveWithParameters(prm)
        if sol is None:
            st.warning("No se encontró solución para el VRP. Intenta relajar restricciones.")
            print("No se encontró solución para el VRP.")
            return [], [None] * n, 0
        routes = []
        eta = [None] * len(dist_m)
        for v in range(vehicles):
            idx = rout.Start(v)
            r = []
            while not rout.IsEnd(idx):
                node = man.IndexToNode(idx)
                r.append(node)
                if node != depot:
                    eta[node] = sol.Value(time_dim.CumulVar(idx))
                idx = sol.Value(rout.NextVar(idx))
            r.append(man.IndexToNode(idx))
            routes.append(r)
        used = sum(1 for r in routes if len(r) > 2)
        if used == 0:
            st.warning("Solo se generaron rutas vacías. Verifica las restricciones o datos de entrada.")
            print("Solo se generaron rutas vacías.")
        print("Rutas calculadas correctamente.")
        return routes, eta, used

def reassign_nearby_stops(
    routes: List[List[int]],
    dist_m: List[List[float]],
    time_m: List[List[int]],
    depot: int,
    balance: bool,
    start_min: int,
    service_time: int,
    max_stops_per_vehicle: int,
    max_distance_m: float,
    balance_threshold: float = 0.9,
    fuel_price: float = 0.0,
    fuel_consumption: float = 0.0,
    price_per_hour: float = 0.0,
    max_minutes: int = 1440,
    time_windows: List[Tuple[int, int]] = None
) -> Tuple[List[List[int]], List[Optional[int]], int]:
    """Reassign nearby stops between routes to optimize clustering."""
    print("Iniciando reassign_nearby_stops...")
    n = len(dist_m)
    vehicles = len(routes)
    validate_matrices(dist_m, time_m, n)

    arc_cost_m = build_cost_matrix(dist_m, time_m, fuel_price, fuel_consumption, price_per_hour)
    man, rout, time_dim = build_routing_model(
        arc_cost_m=arc_cost_m,
        time_m=time_m,
        vehicles=vehicles,
        depot=depot,
        service_time=service_time,
        start_min=start_min,
        balance=balance,
        max_stops_per_vehicle=max_stops_per_vehicle,
        balance_threshold=balance_threshold,
        price_per_hour=price_per_hour,
        max_minutes=max_minutes,
        cost_mode=True,
        time_windows=time_windows
    )

    for v in range(vehicles):
        time_dim.CumulVar(rout.Start(v)).SetRange(start_min, start_min)

    for node in range(1, n):
        idx = man.NodeToIndex(node)
        if idx < 0 or idx >= man.GetNumberOfIndices():
            continue
        rout.AddDisjunction([idx], 1_000_000_000)

    nearby_pairs = []
    for v1, r1 in enumerate(routes):
        if len(r1) <= 2:
            continue
        for node1 in r1[1:-1]:
            for v2, r2 in enumerate(routes):
                if v1 == v2 or len(r2) <= 2:
                    continue
                for node2 in r2[1:-1]:
                    if dist_m[node1][node2] <= max_distance_m:
                        nearby_pairs.append((v1, node1, v2, node2))

    for v1, node1, v2, node2 in nearby_pairs:
        idx1 = man.NodeToIndex(node1)
        idx2 = man.NodeToIndex(node2)
        if not (0 <= idx1 < man.GetNumberOfIndices() and 0 <= idx2 < man.GetNumberOfIndices()):
            continue
        rout.SetAllowedVehiclesForIndex([v1, v2], idx1)
        rout.SetAllowedVehiclesForIndex([v1, v2], idx2)

    prm = pywrapcp.DefaultRoutingSearchParameters()
    prm.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC
    prm.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    prm.time_limit.seconds = 60
    prm.log_search = True

    sol = rout.SolveWithParameters(prm)
    if sol is None:
        st.warning("No se encontró solución al reasignar paradas cercanas. Devolviendo rutas originales.")
        print("No se encontró solución al reasignar paradas cercanas.")
        return routes, [None] * n, sum(1 for r in routes if len(r) > 2)

    new_routes, eta = [], [None] * n
    for v in range(vehicles):
        idx = rout.Start(v)
        r = []
        while not rout.IsEnd(idx):
            node = man.IndexToNode(idx)
            r.append(node)
            if node != depot:
                eta[node] = sol.Value(time_dim.CumulVar(idx))
            idx = sol.Value(rout.NextVar(idx))
        r.append(depot)
        new_routes.append(r)

    used = sum(1 for r in new_routes if len(r) > 2)
    print("Paradas cercanas reasignadas correctamente.")
    return new_routes, eta, used

def recompute_etas(
    routes: List[List[int]],
    time_m: List[List[int]],
    start_min: int,
    service_time: int,
    n: int,
    time_windows: List[Tuple[int, int]] = None
) -> List[Optional[int]]:
    """Recalculate ETAs for routes."""
    print("Recalculando ETAs...")
    validate_matrices([row[:n] for row in time_m[:n]], [row[:n] for row in time_m[:n]], n)
    eta = [None] * n
    violations = []
    for r in routes:
        if len(r) <= 2:
            continue
        t = start_min
        eta[r[0]] = t
        for i in range(1, len(r) - 1):
            prev, node = r[i - 1], r[i]
            t += time_m[prev][node] + service_time
            if time_windows:
                start, end = time_windows[node]
                if t < start:
                    t = start  # Esperar hasta que abra la ventana
                if t > end:
                    violations.append(f"Nodo {node}: ETA {t} min fuera de ventana [{start}, {end}]")
                    logging.warning(f"Nodo {node}: ETA {t} min fuera de ventana [{start}, {end}]")
            eta[node] = t
        t += time_m[r[-2]][r[-1]]
        eta[r[-1]] = t
    if violations:
        st.warning("Algunas ETAs no cumplen con las ventanas de tiempo: " + "; ".join(violations))
        print("Violaciones de ventanas de tiempo: " + "; ".join(violations))
    print("ETAs recalculados correctamente.")
    return eta