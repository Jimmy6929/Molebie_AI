# Security Policy

Molebie AI is a self-hosted, privacy-first AI assistant. Because users run it on
their own machines and connect it to their own data, we take security reports
seriously.

## Reporting a Vulnerability

If you discover a security vulnerability, **please do not open a public GitHub
issue.** Instead, report it privately:

- Email: `jimmymkzhu@gmail.com`
- Subject prefix: `[SECURITY] Molebie AI - <brief description>`

Please include:

- A description of the issue and its impact
- Steps to reproduce
- Affected version / commit hash
- Any suggested fix, if you have one

We aim to:

- Acknowledge the report within **72 hours**
- Provide a remediation timeline within **7 days**
- Credit the reporter in release notes (unless you prefer to remain anonymous)

## Supported Versions

Only the latest release on `main` receives security updates. Older tagged
releases are not actively maintained.

## Scope

In scope:

- The gateway service (`gateway/`)
- The web application (`webapp/`)
- The CLI tools (`cli/`)
- Default configurations, install scripts, and Docker images

Out of scope:

- Third-party services Molebie depends on (Ollama, model files, sqlite-vec, etc.)
- User-modified deployments or custom forks
- Vulnerabilities that require physical access to the host machine
- Self-XSS or social-engineering attacks

## Disclosure Policy

We follow coordinated disclosure. Once a fix is ready, we will:

1. Release the patched version
2. Publish a GitHub Security Advisory
3. Credit the reporter (with their permission)

Thank you for helping keep Molebie AI and its users safe.
