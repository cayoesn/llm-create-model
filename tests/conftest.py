import sys
from unittest.mock import MagicMock

for mod in [
    "mlflow",
    "pythonjsonlogger",
    "opentelemetry.exporter.otlp.proto.grpc",
    "opentelemetry.exporter.otlp.proto.grpc.trace_exporter",
    "opentelemetry.instrumentation.fastapi",
]:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()
