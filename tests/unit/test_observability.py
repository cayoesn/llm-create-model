import logging
from app.observability.logging import setup_logging, StructuredFormatter
import json
import io

def test_structured_formatter():
    formatter = StructuredFormatter()
    log_record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="test.py",
        lineno=10,
        msg="test message",
        args=None,
        exc_info=None
    )
    formatted = formatter.format(log_record)
    data = json.loads(formatted)
    assert data["message"] == "test message"
    assert data["level"] == "INFO"
    assert "timestamp" in data

def test_setup_logging():
    setup_logging("DEBUG")
    logger = logging.getLogger()
    assert logger.level == logging.DEBUG
