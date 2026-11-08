# robot-wetlab

A prototype for planning, booking and recording runs in a synthetic robot wet-lab cell.

![Status: provided as is](https://img.shields.io/badge/status-provided--as--is-lightgrey) ![License: Apache--2.0](https://img.shields.io/badge/license-Apache--2.0-blue)

Created at [Exowatt](https://exowatt.com). Published under couplet-physical-ai, this project is governed independently and remains vendor-neutral.

## Status

**Provided as-is.** The default arm, door, camera, and labware are simulated. No physical cell has been built or validated; this is not a deployable safety system. `lab/cell-a.yaml` is a wholly synthetic example, not a survey or building plan.

## Install

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/). Install [ffmpeg](https://ffmpeg.org/) for MP4 output. From the repository root:

```sh
uv sync
uv run pytest -q
```

The default dependencies include FastAPI, numpy, Pillow and Pydantic. `uv sync` also installs the public `usd-core` test dependency; runtime-only installs can omit the default development group and opt into the USD extra with `uv sync --no-dev --extra usd`. Docker-only sandbox tests skip when Docker or its `python:3.12-slim` image is unavailable; the CPU, API and trusted-subprocess SDK tests run without it. For trusted local programs only, `WETLAB_SANDBOX=subprocess` uses a weaker subprocess sandbox, **without network isolation**. Never use that setting with untrusted code.

## Quickstart: simulated cell + service + SDK

These commands use **only synthetic CPU geometry**, local SQLite, and a virtual arm. No Isaac Sim, GPU, robot, or external service is needed. The included SDK demo program is trusted repository code; its subprocess backend is unsuitable for untrusted submissions.

Terminal 1:

```sh
uv run wetlab plan examples/demo_protocol.json --out out/dry_run.mp4
WETLAB_SANDBOX=subprocess uv run wetlab serve --port 18765 --db var/wetlab.db
```

Terminal 2, once the service starts:

```sh
WETLAB_SANDBOX=subprocess uv run examples/demo_e2e.py --bootstrap --server http://127.0.0.1:18765 --db var/wetlab.db
```

The SDK demo compiles a Python protocol, books a local simulated slot, records an MP4 and results archive, tests a simulated door-open abort, then resets the virtual interlock. Browse `http://127.0.0.1:18765/` using a demo API key if desired. Generated state stays in `var/` and `out/` (ignored by git). Do not paste the printed keys into an issue or log. For an isolated execution backend, install Docker, pull `python:3.12-slim` yourself, and **omit** `WETLAB_SANDBOX=subprocess` on both commands.

The simulated run can take over a minute due to per-frame CPU rendering and video encoding. For a minimal direct SDK compile without a server: `WETLAB_SANDBOX=subprocess uv run wetlab plan path/to/trusted_protocol.py`.

## Configuration

`wetlab serve --help` lists bind address (default loopback), port (8765), lab spec, SQLite path, artifacts directory and simulation speed. `wetlab user add --org demo --name alice --role renter` creates a local API key; keep it private. The service defaults to the virtual driver; see [`docs/PLAN.md`](docs/PLAN.md) for the prototype architecture and [`docs/twin.md`](docs/twin.md) for optional USD/Isaac use. The API exposes bookings, jobs, metering previews, camera stream and interlock controls. There is no billing or safety validation for real deployments.

## Prerequisites for optional integrations (not bundled)

- **Isaac Sim 5.1**: install it separately from [NVIDIA](https://developer.nvidia.com/isaac/sim) under NVIDIA's license; `isaac/dry_run.py`, `mirror.py`, and `worker.py` require that installation. Isaac Lab is **not used**. `wetlab twin-usd` needs the separately installed public `usd-core` extra and only generates repository-authored procedural geometry; it stores the path `/Isaac/Robots/UniversalRobots/ur5e/ur5e.usd` as metadata, **not the model**. The optional Isaac scripts resolve NVIDIA's UR5e asset from **your own Isaac installation or NVIDIA-provided asset root**, subject to NVIDIA's terms. No NVIDIA USD models or Isaac renders ship here.
- **Real UR arm**: optional [`ur_rtde`](https://sdurobotics.gitlab.io/ur_rtde/) (`uv sync --extra ur`) is an MIT-licensed dependency installed by the user, not shipped in this tree. RTSP cameras, relays and robot controller hardware need their own independently configured integrations. These drivers are unvalidated stubs, not a tested safety chain.
- **Untrusted Python submissions**: a locally installed Docker engine and downloaded `python:3.12-slim` image are needed for the default network-isolated compile backend. CI skips its Docker-only tests if absent.

## Safety and limits

The virtual software interlock is for prototyping only. Never connect hardware or admit people into a cell on the strength of the simulation, example coordinates, or software checks. A qualified integrator must independently assess risks, physically wire and validate safety-rated stops and guarding, calibrate the actual cell, and obtain applicable approvals before any real operation. The sample prices and billing events are mock estimates, not a payment system.

The surviving [CPU camera capture](docs/media/capture-after-shake.jpg), [CPU dry-run](docs/media/dry-run.mp4) and [CPU run recording](docs/media/run-recording.mp4) were generated from the project's own procedural geometry. Screenshots bearing old branding and all Isaac/NVIDIA-model imagery have been removed.

## License

Original code is licensed under [Apache-2.0](LICENSE). See [NOTICE](NOTICE).

## Contributing

See [CONTRIBUTING](https://github.com/couplet-physical-ai/.github/blob/main/CONTRIBUTING.md). Contributions require DCO sign-off (`git commit -s`); no CLA is required.

## Security

Report vulnerabilities using [GitHub private vulnerability reporting](https://github.com/couplet-physical-ai/robot-wetlab/security/advisories/new). See the [security policy](https://github.com/couplet-physical-ai/.github/blob/main/SECURITY.md); do not open a public issue.
