# CPU twin and optional Isaac Sim adapter

The CPU twin (`wetlab/twin/`) creates synthetic boxes for the example room, bench, nests, labware, door and robot-arm proxy from `lab/cell-a.yaml`. It performs collision and kinematic checks and renders camera frames with project-owned procedural geometry. This fixture is **not** a surveyed or calibrated room; dimensional and safety claims about real hardware do not follow from its tests.

To generate a USD stage using the separately installable public `usd-core` package (no Isaac needed):

```sh
uv sync --extra usd
uv run wetlab twin-usd --lab lab/cell-a.yaml --out out/cell-a.usda
```

`wetlab/twin/usd.py` writes repository-authored primitives and metadata. The stage is not a copy of any NVIDIA USD asset: the `wetlab:isaac_asset` metadata stores the path `/Isaac/Robots/UniversalRobots/ur5e/ur5e.usd`, which is a reference to a model **obtained separately by the user through their licensed Isaac Sim installation or NVIDIA asset root**. No NVIDIA assets or renders are included here. USD tests exercise authored geometry with `usd-core` only; they do not import Isaac Sim.

Isaac Sim 5.1 can optionally run `isaac/dry_run.py` or `isaac/mirror.py` after the user installs and accepts [NVIDIA's licensing terms](https://developer.nvidia.com/isaac/sim). Use the vendor's `python.sh` and asset root on your own machine. Isaac Lab is not used. The optional worker sends results to your own locally configured service using `WETLAB_WORKER_TOKEN`; there is no bundled GPU model, cloud account, or public simulator service. Real-world integration, calibration and collision/contact validity have **not** been independently validated for this release. Do not treat simulator contact checks as certified personnel protection.
