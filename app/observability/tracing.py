from fastapi import FastAPI
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from app.config.settings import settings


def setup_tracing(app: FastAPI) -> None:
    """
    Sets up OpenTelemetry tracing for the FastAPI app.
    """
    if not settings.OTEL_EXPORTER_OTLP_ENDPOINT:
        return

    provider = TracerProvider()
    processor = BatchSpanProcessor(OTLPSpanExporter(endpoint=settings.OTEL_EXPORTER_OTLP_ENDPOINT))
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)

    FastAPIInstrumentor.instrument_app(app)
