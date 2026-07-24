# Security Policy

`tracebook` is an alpha matching-engine conformance, qualification, and
failure-forensics toolkit that also ships simulation and replay utilities. It
executes configured candidate adapter commands with the caller's permissions;
timeouts do not sandbox those processes. It does not provide exchange
connectivity, custody, account management, or trading advice.

## Supported Versions

| Version | Supported |
| --- | --- |
| `0.5.x` | Yes |
| `< 0.5` | No |

Security fixes also target the latest commit on `main`.

## Reporting A Vulnerability

Please do not open a public issue for a suspected vulnerability. Submit a
[private vulnerability report](https://github.com/Taz33m/tracebook/security/advisories/new),
or contact the maintainer directly through the GitHub profile for `Taz33m`.

Useful reports include:

- affected version or commit SHA
- a minimal reproduction
- impact and any realistic exploit path
- whether the issue requires optional extras such as `dashboard` or `analysis`

We will acknowledge credible reports as quickly as possible and document any fix in `CHANGELOG.md`.
