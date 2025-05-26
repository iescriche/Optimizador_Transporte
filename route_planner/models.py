from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional
import streamlit as st

@dataclass
class Stop:
    index: int
    address: str
    latitude: float
    longitude: float
    eta: Optional[int] = None  # Minutes since midnight
    visited: bool = False

    def __post_init__(self):
        if not (0 <= self.index):
            st.error(f"Índice de parada inválido: {self.index}")
            raise ValueError(f"Stop index must be non-negative: {self.index}")
        from .utils import validate_coords
        if not validate_coords(self.latitude, self.longitude):
            raise ValueError(f"Invalid coordinates: lat={self.latitude}, lon={self.longitude}")

@dataclass
class Vehicle:
    vehicle_id: int
    sequence: List[int]
    visited: List[bool]

    def __post_init__(self):
        if len(self.sequence) != len(self.visited):
            st.error(f"Longitudes de secuencia y visitados no coinciden: {len(self.sequence)} vs {len(self.visited)}")
            raise ValueError("Sequence and visited lists must have equal lengths")
        if self.vehicle_id < 0:
            raise ValueError(f"Vehicle ID must be non-negative: {self.vehicle_id}")

@dataclass
class PlanningResult:
    generated_at: str  # ISO 8601
    depot: int
    coordinates: List[Tuple[float, float]]
    addresses: List[str]
    routes: List[Vehicle]
    distance_matrix_hash: str
    settings: dict
    distance_matrix: Optional[List[List[float]]] = None
    time_matrix: Optional[List[List[int]]] = None
    eta: Optional[List[Optional[int]]] = None

    def __post_init__(self):
        try:
            from datetime import datetime
            datetime.fromisoformat(self.generated_at)
        except ValueError:
            st.error(f"Timestamp inválido: {self.generated_at}")
            raise ValueError(f"generated_at must be ISO 8601: {self.generated_at}")
        if self.depot < 0:
            raise ValueError(f"Depot index must be non-negative: {self.depot}")

    def asdict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            generated_at=data["generated_at"],
            depot=data["depot"],
            coordinates=data["coordinates"],
            addresses=data["addresses"],
            routes=[Vehicle(**r) for r in data["routes"]],
            distance_matrix_hash=data["distance_matrix_hash"],
            settings=data["settings"],
            distance_matrix=data.get("distance_matrix"),
            time_matrix=data.get("time_matrix"),
            eta=data.get("eta"),
        )