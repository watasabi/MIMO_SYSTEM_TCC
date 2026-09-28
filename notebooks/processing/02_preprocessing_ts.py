"""Preprocessing de series temporais para o pipeline mimo_sys.

A implementacao vive em `mimo_sys.preprocessors.TimeSeriesPreprocessor`;
este script apenas carrega os dados e aplica o pipeline.
"""

from mimo_sys.preprocessors import TimeSeriesPreprocessor

preprocessor = TimeSeriesPreprocessor()
