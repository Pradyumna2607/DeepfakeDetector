# Security Policy

## Supported versions
Only the latest commit on `main` receives fixes.

## Reporting a vulnerability
Please report vulnerabilities privately through GitHub's
**Security → Report a vulnerability** feature on this repository rather than a public issue.
Include steps to reproduce and the potential impact. You can expect an initial response within 7 days.

## Deployment hardening checklist
- Set a strong `SECRET_KEY` via environment variable (never use the development default).
- Keep `FLASK_DEBUG=0` in production and serve behind gunicorn + a reverse proxy over HTTPS.
- Never commit `.env` files, databases, or user uploads.
