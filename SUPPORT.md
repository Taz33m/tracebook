# Support

For now, support happens through GitHub issues and discussions when enabled.

## Good Support Requests

Please include:

- what you are trying to do
- your Python version and OS
- install command used
- exact command or minimal Python snippet
- full traceback or benchmark JSON when relevant

## Scope

`tracebook` is for matching-engine conformance, profile-scoped qualification,
reproducible failure reduction, normalized historical order-event replay,
simulation, benchmarking, profiling, and local public-feed capture. A passing
qualification applies only to its named profile, candidate identity, and
recorded workload; it is not exchange certification. Tracebook is not
production trading infrastructure, does not accept exchange credentials or
place orders, and does not grant rights to redistribute captured market data.

## Platform Boundary

Release CI covers Python 3.10-3.13 on Ubuntu. Atomic campaign and qualification
artifact publication requires descriptor-relative filesystem operations, so
Ubuntu is currently the only release-tested platform for that path. It fails
closed where those operations are unavailable, and Windows qualification
artifact publication is not supported. macOS has been exercised manually but
is not a release-gated support target. Other package surfaces may work on other
operating systems without a support guarantee.
