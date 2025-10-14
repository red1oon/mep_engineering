## [Phase 2B] - 2025-01-14

### Fixed
- Correct route start coordinates from validation script findings
- Add missing pathlib.Path import in ValidateConduitRoute._export_bcf()

### Validated
- Routing: 10m test route detects 67 obstacles (ARC:31, CW:16, FP:11, ACMV:8, STR:1)
- Clash validation: All 67 clashes reported with discipline grouping
- IFC4 generation: IfcCableCarrierSegment created successfully
- Filtering: _sanitize_clashes() removes 12 false positives correctly

### Technical Notes
- 100% match between routing obstacle detection and clash validation
- Federation spatial queries working (query_corridor + disciplines filter)
- Ready for Phase 2C (BCF export implementation)
