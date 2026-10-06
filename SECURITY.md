# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| 0.6.x   | :white_check_mark: |
| < 0.6   | :x:                |

## Reporting a Vulnerability

**Please do not open a public GitHub issue for security vulnerabilities.**

Report privately via GitHub Security Advisories
(**Security → Report a vulnerability** on this repository), or contact the
maintainer directly through the email on the [GitHub profile][owner] if you
cannot use advisories.

[owner]: https://github.com/kuntal-devrat

### What to include

- TorchBurn version (`python -c "import torchburn; print(torchburn.__version__)"`)
  and platform (OS, Python, PyTorch versions).
- A minimal reproduction (script, payload JSON, or GGUF file) — please do not
  send model weights.
- Your assessment of impact and any known workarounds.

### What to expect

- Acknowledgement within **7 days**.
- An initial triage (accepted / declined / needs more info) within **14 days**.
- Coordinated disclosure: we aim to ship a fix within **90 days** of an
  accepted report and will credit reporters in the changelog unless they
  prefer to remain anonymous.

## Scope

High-value areas for this codebase:

- **Native FFI boundary** — DLPack capsule handling, prepared-graph cache
  (`prepare_graph` / `execute_prepared`), payload parsing (untrusted JSON
  from Python), and panic-safety of the native engine.
- **GGUF / GGML parser** — untrusted model files are parsed with bounded
  allocations; hostile counts, truncated data, and malformed headers must
  never panic or exhaust memory.
- **LLM sampling path** — anything that mutates global interpreter state
  (torch RNG, thread pools) or leaks GIL-held compute across threads.

Out of scope: vulnerabilities in upstream PyTorch/Rust dependencies (report
those upstream), and issues requiring an attacker who already controls the
host process.
