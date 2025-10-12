# ============================================================================
# FILE: prop.py
# PURPOSE: Define properties (data storage) for MEP Engineering module
# ============================================================================

import bpy
from bpy.types import PropertyGroup
from bpy.props import StringProperty, BoolProperty, IntProperty, FloatProperty, FloatVectorProperty

class BIMmepEngineeringProperties(PropertyGroup):
    """Properties for MEP Engineering calculations and settings"""
    
    # Example property: Project identifier
    project_number: StringProperty(
        name="Project Number",
        description="Project identification number",
        default=""
    )
    
    # Example property: Enable clash detection
    enable_clash_detection: BoolProperty(
        name="Enable Clash Detection",
        description="Perform clash detection during routing",
        default=True
    )
    
    min_clearance_mm: IntProperty(
        name="Minimum Clearance",
        description="Minimum clearance distance in millimeters",
        default=50,
        min=0,
        max=500
    )
    
    # ============================================================================
    # ROUTING PROPERTIES (Phase 1)
    # ============================================================================
    
    route_start_point: FloatVectorProperty(
        name="Route Start",
        description="Starting point for conduit route (X, Y, Z in meters)",
        size=3,
        default=(0.0, 0.0, 0.0),
        subtype='XYZ',
        precision=3
    )
    
    route_end_point: FloatVectorProperty(
        name="Route End",
        description="End point for conduit route (X, Y, Z in meters)",
        size=3,
        default=(0.0, 0.0, 0.0),
        subtype='XYZ',
        precision=3
    )
    
    clearance_distance: FloatProperty(
        name="Clearance Distance",
        description="Minimum clearance from obstacles in meters (e.g., 0.5m = 500mm)",
        default=0.5,
        min=0.1,
        max=2.0,
        precision=2,
        unit='LENGTH'
    )
    
    conduit_diameter: FloatProperty(
        name="Conduit Diameter",
        description="Diameter of conduit in meters",
        default=0.05,  # 50mm
        min=0.01,
        max=0.5,
        precision=3,
        unit='LENGTH'
    )
    
    target_disciplines: StringProperty(
        name="Obstacle Disciplines",
        description="Comma-separated list of disciplines to check for obstacles (e.g., 'STR,ACMV,ARC')",
        default="STR,ACMV,ARC,FP,SP,CW"
    )
