"""OpenTelemetry setup: one span per phase, so a flame view shows where time
actually goes rather than where you assumed it went.

Tracing is off by default. The exporter's own overhead is visible in ITL at the
millisecond scale this project measures at, so it is a debugging tool, not
something the benchmark runs with. Every method below is a no-op when disabled.

One structural note. Spans are created with an **explicit parent context**
rather than ``start_as_current_span``. The engine thread interleaves many
requests, so the "currently active span" is meaningless there: using it would
parent one request's prefill span under another request's decode span and
produce a trace that is confidently wrong.
"""

from __future__ import annotations

from opentelemetry import trace
from opentelemetry.trace import set_span_in_context

tracer = trace.get_tracer("cadence")

_configured = False


def setup_tracing(cfg) -> None:
    global _configured
    if _configured or not cfg.tracing_enabled:
        return
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = TracerProvider(
        resource=Resource.create(
            {"service.name": "cadence", "cadence.config": cfg.config_name}
        )
    )
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=cfg.otlp_endpoint, insecure=True))
    )
    trace.set_tracer_provider(provider)
    _configured = True


class RequestTrace:
    """One span per request, its lifecycle transitions as span events, and a
    child span per phase.

    A single request's trace reads ``queue -> prefill -> decode`` with the
    durations that actually explain its latency, which is the fastest way to
    tell a queueing problem from a decoding problem.
    """

    __slots__ = ("enabled", "_span", "_ctx", "_phase", "_phase_name")

    def __init__(self, rid: str, enabled: bool, **attrs) -> None:
        self.enabled = bool(enabled)
        self._span = None
        self._ctx = None
        self._phase = None
        self._phase_name: str | None = None
        if not self.enabled:
            return
        self._span = tracer.start_span("request", attributes={"cadence.rid": rid, **attrs})
        self._ctx = set_span_in_context(self._span)

    @property
    def phase_name(self) -> str | None:
        return self._phase_name

    def event(self, name: str, **attrs) -> None:
        if self._span is not None:
            self._span.add_event(name, attributes=attrs)

    def phase(self, name: str, **attrs) -> None:
        """Open a phase span, closing the previous one. Phases do not nest."""
        if not self.enabled:
            return
        if self._phase_name == name:
            return  # already in it; re-entering would fragment the timeline
        self.end_phase()
        self._phase = tracer.start_span(name, context=self._ctx, attributes=attrs)
        self._phase_name = name

    def set(self, **attrs) -> None:
        if self._span is not None:
            for k, v in attrs.items():
                self._span.set_attribute(k, v)

    def end_phase(self) -> None:
        if self._phase is not None:
            self._phase.end()
            self._phase = None
            self._phase_name = None

    def end(self, **attrs) -> None:
        if not self.enabled:
            return
        self.end_phase()
        self.set(**attrs)
        if self._span is not None:
            self._span.end()
            self._span = None


class NoopSpan:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def add_event(self, *a, **k):
        return None

    def set_attribute(self, *a, **k):
        return None


_NOOP = NoopSpan()


def span(name: str, enabled: bool, **attrs):
    if not enabled:
        return _NOOP
    return tracer.start_as_current_span(name, attributes=attrs)
