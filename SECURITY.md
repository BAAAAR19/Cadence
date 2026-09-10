# Security policy

## Supported versions

The latest revision on the default branch receives security fixes.

## Reporting a vulnerability

Use [GitHub private vulnerability reporting](https://github.com/BAAAAR19/Cadence/security/advisories/new).
Do not put credentials, private prompts, model files, production traces, or
exploit details in a public issue. Include the affected revision, impact, and a
minimal reproduction using synthetic prompts when possible.

## Deployment notes

Cadence is a reference inference gateway, not a complete internet-facing
security boundary. A production deployment should add authentication, TLS,
tenant isolation, request-size limits, rate limits, network egress controls,
secret management, and a documented retention policy for prompts, outputs,
metrics, and traces.

The example deployment exposes operational telemetry and an OpenAI-compatible
API. Keep Prometheus, Grafana, and OpenTelemetry endpoints on trusted networks,
and review trace settings before handling sensitive inputs. Admission control
protects a latency objective; it is not an authorization or abuse-prevention
mechanism.
