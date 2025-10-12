MEP engineering automation and coordination for Bonsai/BlenderBIM.

## Features

- Automated conduit/pipe routing with clash avoidance
- Multi-discipline obstacle detection (uses federation module)
- Load calculations and compliance checking
- Report generation for documentation

## Installation

**Requires:** Federation module

```bash
cd src/bonsai/bonsai/bim/module/
git clone https://github.com/red1oon/federation.git
git clone https://github.com/red1oon/mep_engineering.git
```

Restart Blender to load modules.
Usage

Load federated models (Architecture, Structural, MEP)
Set routing start/end points
Click "Route Conduit" - automatically avoids obstacles
Generate IFC elements and compliance reports

Reduces Engineering Grunt Work

Material takeoffs: 2 hours → 5 minutes
Routing with clash avoidance: Automated
Documentation: Auto-generated reports

Part of Bonsai10D Vision
10-dimensional construction coordination: 3D geometry + time + cost + compliance + ERP integration (iDempiere).

License
GPL-3.0-or-later

Author
Redhuan D. Oon (red1), Naquib Danial Oon

