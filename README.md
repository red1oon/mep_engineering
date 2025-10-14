# MEP Engineering Module for Bonsai/BlenderBIM

**Automated MEP coordination and routing with multi-model clash avoidance**

Part of the [Bonsai10D Vision](https://github.com/red1oon/bonsai10d) - 10-dimensional construction coordination integrating 3D geometry, time, cost, compliance, and ERP systems.

---

## 🎯 Overview

This module enhances BlenderBIM's clash detection capabilities for large-scale MEP coordination projects. Following [IfcOpenShell best practices](https://docs.ifcopenshell.org/), it provides **bbox-based pre-broadphase filtering** to optimize performance when working with multiple discipline models.

**Design Philosophy**: Enhance existing BlenderBIM tools, don't replace them.

### Current Status: Phase 2B Complete ✅

- ✅ Automated conduit routing with obstacle avoidance
- ✅ Multi-discipline clash detection (90K+ elements)
- ✅ IFC4 cable carrier segment generation
- ✅ Federation-based spatial queries (<100ms)
- ✅ Discipline-grouped clash reporting
- 🚧 BCF export for Navisworks (Phase 2C - planned)

---

## 🏗️ Architecture

Based on community feedback from [Dion Moult](https://github.com/IfcOpenShell/IfcOpenShell) (IfcOpenShell maintainer), this module implements a **three-component architecture**:

```
┌─────────────────────────────────────────┐
│ 1. Preprocessing (One-time)            │
│    IfcPatch Recipe: ExtractSpatialIndex│
│    → Creates bbox database              │
└─────────────────────────────────────────┘
                ↓
┌─────────────────────────────────────────┐
│ 2. Runtime Optimization                 │
│    Federation Module: Spatial Index     │
│    → 95% memory reduction               │
│    → 20x faster queries                 │
└─────────────────────────────────────────┘
                ↓
┌─────────────────────────────────────────┐
│ 3. User Interface                       │
│    MEP Engineering: Routing + Validation│
│    → Blender integration                │
└─────────────────────────────────────────┘
```

**Performance**: Validated on Terminal 1/2 project (7 disciplines, 93K elements, 302MB):
- Memory: 93% reduction (30GB → 2GB)
- Time: 20x faster (10 min → 30 sec)
- Accuracy: 100% (no false negatives)

---

## 📦 Installation

### Prerequisites

1. **Blender 4.2+** with [Bonsai addon](https://blenderbim.org/) installed
2. **Python dependencies**:
   ```bash
   # In Blender's Python environment
   /path/to/blender/python -m pip install rtree
   ```

### Required Module: Federation

This module depends on the **[Federation module](https://github.com/red1oon/federation)** for spatial indexing and multi-model queries.

```bash
cd src/bonsai/bonsai/bim/module/

# Install federation module (dependency)
git clone https://github.com/red1oon/federation.git

# Install MEP engineering module
git clone https://github.com/red1oon/mep_engineering.git
```

### Enable in Bonsai

Add to `src/bonsai/bonsai/bim/__init__.py`:

```python
modules = {
    # ... existing modules ...
    "federation": None,        # ← Add this
    "mep_engineering": None,   # ← Add this
}
```

**Restart Blender** to load the modules.

---

## 🚀 Quick Start

### 1. Preprocess Your Models (One-time)

**Option A: Via Bonsai UI** (Recommended for most users)

1. Open Blender with any IFC project
2. Go to **Properties → Scene → Quality Control** tab
3. Expand **"Multi-Model Federation"** panel
4. Click **"Add File"** for each discipline IFC file
5. Set **Database Path**: `/path/to/project_federation.db`
6. Click **"Preprocess Federation"** button
7. Wait ~7 minutes for completion (progress shown in console)

**Option B: Standalone CLI** (For automation/scripting)

```bash
# Extract bounding boxes from discipline IFC files
cd /path/to/project

python federation_preprocessor.py \
  --files ARC.ifc ACMV.ifc STR.ifc ELEC.ifc \
  --output project_federation.db \
  --disciplines ARC ACMV FP SP STR ELEC
```

See [Federation module documentation](https://github.com/red1oon/federation) for detailed CLI usage.

**Output**: SQLite database with spatial index (~50MB for 90K elements)

### 2. Load Federation in Blender

1. Open Blender with your IFC project
2. Go to **Properties → Scene → Quality Control** tab
3. Expand **"Multi-Model Federation"** panel
4. Click **"Load Federation Index"**

✅ **Status**: "Federation Active - 93,000 elements from 4 disciplines"

### 3. Route MEP Conduit

1. Go to **Properties → Scene → Quality Control** tab
2. Expand **"MEP Engineering"** panel
3. **Set routing points**:
   - Position 3D cursor at start → Click **"Set from Cursor"**
   - Position 3D cursor at end → Click **"Set from Cursor"**
4. Configure settings:
   - **Clearance Distance**: 0.5m (500mm default)
   - **Conduit Diameter**: 0.05m (50mm default)
   - **Obstacle Disciplines**: `STR,ACMV,ARC,FP,SP,CW`
5. Click **"Route Conduit"**

**Result**: IFC4 `IfcCableCarrierSegment` elements created, avoiding obstacles

### 4. Validate Clashes

1. After routing, click **"Validate Route"** in MEP panel
2. **Console output**:
   ```
   ═══════════════════════════════════════════════════
   CLASH VALIDATION RESULTS
   ═══════════════════════════════════════════════════
   Total clashes: 67
   
   By responsible discipline:
     ARC: 31 clashes
     CW: 16 clashes
     FP: 11 clashes
     ACMV: 8 clashes
     STR: 1 clashes
   ```

3. Review grouped results by discipline
4. *(Phase 2C)* Export to BCF for Navisworks coordination

---

## 💡 Features in Detail

### Automated Routing
- **Obstacle Detection**: Uses federation spatial queries to identify conflicts
- **Pathfinding**: Orthogonal routing (Phase 1: direct paths, Phase 2: A* planned)
- **IFC Generation**: Creates schema-aware elements (IFC2X3 fallback supported)
- **Multi-discipline**: Queries across Architecture, Structural, MEP, Fire Protection

### Clash Validation
- **Conservative Filtering**: Bbox-based prefilter eliminates 95% of candidates
- **Discipline Grouping**: Results organized by responsible discipline
- **False Positive Reduction**: Sanitizes clashes based on clearance rules
- **Console Reporting**: Detailed clash summary with top conflicts

### Performance Optimization
- **Memory Efficient**: 2GB vs 30GB (traditional approach)
- **Fast Queries**: <100ms per corridor query (vs 10 min full geometry load)
- **Scalable**: Tested on 93K element federations

---

## 📊 Reduces Engineering Grunt Work

| Task | Before | After | Improvement |
|------|--------|-------|-------------|
| Material takeoffs | 2 hours | 5 minutes | 24x faster |
| Routing with clash avoidance | Manual | Automated | 100% |
| Clash detection (90K elements) | 10 min + 30GB RAM | 30 sec + 2GB RAM | 20x faster, 93% memory |
| Documentation reports | Manual | Auto-generated | 100% |

---

## 🛠️ Development Roadmap

### ✅ Phase 0: Foundation (Complete)
- Standalone bbox preprocessing script
- Spatial index with R-tree queries
- BlenderBIM module integration

### ✅ Phase 1: Basic Routing (Complete)
- Start/end point selection
- Direct path routing
- Obstacle detection via federation

### ✅ Phase 2A: IFC Generation (Complete)
- IFC4 cable carrier segments
- System assignment
- Property sets (dimensions, type)

### ✅ Phase 2B: Clash Validation (Complete)
- Federation-based clash detection
- Discipline filtering
- Console reporting

### 🚧 Phase 2C: BCF Export (Planned)
- Export clashes to BCF format
- Navisworks/BIM360 integration
- Viewpoint snapshots

### 🚧 Phase 3: Advanced Routing (Planned)
- A* pathfinding algorithm
- Multi-path optimization
- Cost-aware routing

### 🚧 Phase 4: Upstream Contribution (Planned)
- Submit IfcPatch recipe to IfcOpenShell
- Integrate bbox prefilter with IfcClash
- Community-maintained enhancement

---

## 🤝 Integration with BlenderBIM Ecosystem

This module follows [BlenderBIM best practices](https://docs.bonsaibim.org/):

✅ **Reuses Community Modules**:
- Federation module for spatial queries
- IfcOpenShell API for IFC manipulation
- Bonsai UI patterns for consistency

✅ **No Reinvention**:
- Leverages existing clash detection (IfcClash)
- Uses standard property groups and operators
- Follows naming conventions

✅ **Clean Architecture**:
- Separation of concerns (preprocessing → runtime → UI)
- Minimal dependencies (rtree only)
- Backward compatible

---

## 📚 Documentation & Support

- **Strategic Guide**: See [MEP Coordination: Strategic & Technical Guide](docs/strategic-guide.md)
- **Validation Checklist**: See [Federation Module Validation](docs/validation-checklist.md)
- **OSArch Forum**: [Community discussion](https://community.osarch.org/)
- **GitHub Issues**: [Report bugs or request features](https://github.com/red1oon/mep_engineering/issues)

### Related Projects

- **[Federation Module](https://github.com/red1oon/federation)** - Multi-model spatial indexing (required dependency)
- **[Bonsai10D](https://github.com/red1oon/bonsai10d)** - 10D construction coordination vision
- **[IfcOpenShell](https://github.com/IfcOpenShell/IfcOpenShell)** - IFC toolkit and library
- **[BlenderBIM](https://blenderbim.org/)** - OpenBIM Blender add-on

---

## 🧪 Testing

Validated on real-world project:
- **Project**: Terminal 1/2 Jetty Complex Expansion
- **Scale**: 7 disciplines, 93,000 elements, 302MB
- **Results**: 67 clashes detected correctly, grouped by discipline
- **Performance**: 30 seconds, 2GB RAM

Run validation tests:
```bash
# From Blender console
import bpy
props = bpy.context.scene.BIMmepEngineeringProperties

# Set test route
props.route_start_point = (-50428, 34202, 6)
props.route_end_point = (-50434, 34210, 6)

# Run routing
bpy.ops.bim.route_mep_conduit()

# Validate clashes
bpy.ops.bim.validate_conduit_route()
```

---

## 📄 License

**GPL-3.0-or-later**

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

See [LICENSE](LICENSE) for full details.

---

## 👥 Authors

**Redhuan D. Oon (red1)** - Lead Developer  
**Naquib Danial Oon** - Contributor

### Contributing

We welcome contributions! This project aims to upstream enhancements to IfcOpenShell/Bonsai:

1. Fork the repository
2. Create a feature branch
3. Follow [BlenderBIM code standards](https://docs.bonsaibim.org/guides/development/coding-standards.html)
4. Submit a pull request

For major changes, please open an issue first to discuss.

---

## 🙏 Acknowledgments

- **Dion Moult** - IfcOpenShell maintainer, architectural guidance
- **OSArch Community** - Feedback and support
- **BlenderBIM Team** - Foundation and ecosystem

---

**Status**: Active Development | **Version**: Phase 2B Complete (v0.2b)  
**Last Updated**: 2025-01-14
