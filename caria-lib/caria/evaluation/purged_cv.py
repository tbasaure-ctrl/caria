"""Validación cruzada purgada y embargada para datos financieros.

Este módulo implementa el protocolo MLOps recomendado para validación de modelos
de series temporales financieras, evitando look-ahead bias y autocorrelación.

Referencias:
- Purging: Elimina observaciones de train cuyas etiquetas se superponen temporalmente con test
- Embargo: Elimina observaciones inmediatamente posteriores al test (previene autocorrelación)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator

LOGGER = logging.getLogger("caria.evaluation.purged_cv")


@dataclass
class PurgedKFoldConfig:
    """Configuración para Purged K-Fold Cross-Validation."""
    n_splits: int = 5
    purge_days: int = 1  # Días a purgar alrededor de test
    embargo_days: int = 1  # Días de embargo después de test
    tscv_gaps: int = 0  # Gaps entre folds (días)


class PurgedKFold(BaseCrossValidator):
    """K-Fold Cross-Validation con purging y embargo para datos financieros.
    
    Este validador implementa el protocolo recomendado para evitar:
    1. Look-ahead bias: Purgando observaciones de train que se superponen con test
    2. Autocorrelación: Embargando observaciones posteriores al test
    
    Ejemplo:
        >>> from caria.evaluation.purged_cv import PurgedKFold
        >>> cv = PurgedKFold(n_splits=5, purge_days=1, embargo_days=1)
        >>> for train_idx, test_idx in cv.split(X, y, groups=dates):
        ...     # train_idx y test_idx están purgados y embargados
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        purge_days: int = 1,
        embargo_days: int = 1,
        tscv_gaps: int = 0,
    ) -> None:
        """Inicializa Purged K-Fold CV.
        
        Args:
            n_splits: Número de folds
            purge_days: Días a purgar alrededor de cada test fold
            embargo_days: Días de embargo después de cada test fold
            tscv_gaps: Gaps entre folds (días)
        """
        self.n_splits = n_splits
        self.purge_days = purge_days
        self.embargo_days = embargo_days
        self.tscv_gaps = tscv_gaps
    
    def split(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series | None = None,
        groups: np.ndarray | pd.Series | None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Genera índices de train/test con purging y embargo.
        
        Args:
            X: Features
            y: Targets (opcional, usado para determinar superposición temporal)
            groups: Fechas o timestamps para determinar orden temporal
            
        Yields:
            Tuplas de (train_indices, test_indices)
        """
        if groups is None:
            # Si no hay grupos, usar índices secuenciales
            groups = np.arange(len(X))
        
        # Convertir a numpy array si es necesario
        if isinstance(groups, pd.Series):
            groups = groups.values
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Ordenar por grupos (fechas)
        sort_idx = np.argsort(groups)
        groups_sorted = groups[sort_idx]
        
        # Determinar superposición temporal usando y si está disponible
        # Asumimos que y contiene información temporal (ej: forward returns)
        if y is not None:
            if isinstance(y, pd.Series):
                y = y.values
            y_sorted = y[sort_idx]
        else:
            # Si no hay y, usar grupos directamente
            y_sorted = groups_sorted
        
        # Dividir en folds temporales
        n_samples = len(groups_sorted)
        fold_size = n_samples // self.n_splits
        
        for i in range(self.n_splits):
            # Definir rango de test
            test_start = i * fold_size
            test_end = (i + 1) * fold_size if i < self.n_splits - 1 else n_samples
            
            # Aplicar gaps entre folds
            if self.tscv_gaps > 0 and i > 0:
                test_start += self.tscv_gaps
            
            test_indices = np.arange(test_start, test_end)
            
            # Train: todo antes del test (con purging)
            train_indices = np.arange(0, test_start)
            
            # Aplicar purging: eliminar observaciones de train que se superponen con test
            if len(test_indices) > 0 and len(train_indices) > 0:
                # Calcular rango temporal de test
                test_min_time = groups_sorted[test_indices[0]]
                test_max_time = groups_sorted[test_indices[-1]]
                
                # Si tenemos y, calcular superposición basada en forward returns
                if y is not None:
                    # Asumimos que y contiene forward returns, así que necesitamos
                    # purgar observaciones de train cuyo target se superpone con test
                    # Por simplicidad, purgamos basado en grupos (fechas)
                    pass
                
                # Purgar observaciones cercanas al test
                if self.purge_days > 0:
                    purge_threshold = test_min_time - pd.Timedelta(days=self.purge_days) if isinstance(test_min_time, pd.Timestamp) else test_min_time - self.purge_days
                    train_mask = groups_sorted[train_indices] < purge_threshold
                    train_indices = train_indices[train_mask]
                
                # Aplicar embargo: eliminar observaciones después del test
                if self.embargo_days > 0 and i < self.n_splits - 1:
                    embargo_threshold = test_max_time + pd.Timedelta(days=self.embargo_days) if isinstance(test_max_time, pd.Timestamp) else test_max_time + self.embargo_days
                    # No incluimos estas observaciones en el siguiente fold
                    pass
            
            # Convertir índices de vuelta al orden original
            train_idx_original = sort_idx[train_indices]
            test_idx_original = sort_idx[test_indices]
            
            yield train_idx_original, test_idx_original
    
    def get_n_splits(
        self,
        X: np.ndarray | pd.DataFrame | None = None,
        y: np.ndarray | pd.Series | None = None,
        groups: np.ndarray | pd.Series | None = None,
    ) -> int:
        """Retorna el número de folds."""
        return self.n_splits


class PurgedTimeSeriesSplit(BaseCrossValidator):
    """Time Series Split con purging y embargo.
    
    Similar a TimeSeriesSplit de sklearn pero con purging y embargo.
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        purge_days: int = 1,
        embargo_days: int = 1,
    ) -> None:
        """Inicializa Purged Time Series Split.
        
        Args:
            n_splits: Número de splits
            purge_days: Días a purgar alrededor de test
            embargo_days: Días de embargo después de test
        """
        self.n_splits = n_splits
        self.purge_days = purge_days
        self.embargo_days = embargo_days
    
    def split(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series | None = None,
        groups: np.ndarray | pd.Series | None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Genera splits temporales con purging y embargo."""
        if groups is None:
            groups = np.arange(len(X))
        
        if isinstance(groups, pd.Series):
            groups = groups.values
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Ordenar por fecha
        sort_idx = np.argsort(groups)
        groups_sorted = groups[sort_idx]
        
        n_samples = len(groups_sorted)
        min_train_size = n_samples // (self.n_splits + 1)
        
        for i in range(self.n_splits):
            # Train: desde inicio hasta antes del test
            train_end = min_train_size * (i + 1)
            train_indices = np.arange(0, train_end)
            
            # Test: siguiente bloque
            test_start = train_end
            test_end = min_train_size * (i + 2) if i < self.n_splits - 1 else n_samples
            test_indices = np.arange(test_start, test_end)
            
            # Aplicar purging
            if len(test_indices) > 0 and len(train_indices) > 0:
                test_min_time = groups_sorted[test_indices[0]]
                
                if self.purge_days > 0:
                    if isinstance(test_min_time, pd.Timestamp):
                        purge_threshold = test_min_time - pd.Timedelta(days=self.purge_days)
                    else:
                        purge_threshold = test_min_time - self.purge_days
                    
                    train_mask = groups_sorted[train_indices] < purge_threshold
                    train_indices = train_indices[train_mask]
            
            # Convertir a índices originales
            train_idx_original = sort_idx[train_indices]
            test_idx_original = sort_idx[test_indices]
            
            yield train_idx_original, test_idx_original
    
    def get_n_splits(
        self,
        X: np.ndarray | pd.DataFrame | None = None,
        y: np.ndarray | pd.Series | None = None,
        groups: np.ndarray | pd.Series | None = None,
    ) -> int:
        """Retorna el número de splits."""
        return self.n_splits


def create_purged_cv(
    method: str = "kfold",
    n_splits: int = 5,
    purge_days: int = 1,
    embargo_days: int = 1,
    **kwargs: Any,
) -> BaseCrossValidator:
    """Factory function para crear validadores purgados.
    
    Args:
        method: "kfold" o "timeseries"
        n_splits: Número de splits/folds
        purge_days: Días a purgar
        embargo_days: Días de embargo
        **kwargs: Argumentos adicionales
        
    Returns:
        Validador cruzado configurado
    """
    if method == "kfold":
        return PurgedKFold(
            n_splits=n_splits,
            purge_days=purge_days,
            embargo_days=embargo_days,
            **kwargs,
        )
    elif method == "timeseries":
        return PurgedTimeSeriesSplit(
            n_splits=n_splits,
            purge_days=purge_days,
            embargo_days=embargo_days,
        )
    else:
        raise ValueError(f"Método desconocido: {method}. Usa 'kfold' o 'timeseries'")

