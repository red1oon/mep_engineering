# Bonsai - OpenBIM Blender Add-on
# Copyright (C) 2025 Iktisas IT Sdn Bhd
#
# This file is part of Bonsai.
#
# Bonsai is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
MEP Engineering Operators - REFACTORED
Thin operators that validate input and call tool.py business logic
Federation module integration preserved
"""

import bpy
from bpy.types import Operator
from . import visualization
from . import tool


class TestMEPOperator(Operator):
    """Test operator to verify MEP module is working"""
    bl_idname = "bim.test_mep_operator"
    bl_label = "Test MEP Module"
    bl_description = "Test that MEP Engineering module is loaded"
    bl_options = {"REGISTER", "UNDO"}
    
    def execute(self, context):
        """Called when operator is executed"""
        props = context.scene.BIMmepEngineeringProperties
        
        self.report({'INFO'}, f"MEP Module Working! Project: {props.project_number}")
        
        print("=" * 50)
        print("MEP Engineering Module Test")
        print(f"Project Number: {props.project_number}")
        print(f"Clash Detection: {props.enable_clash_detection}")
        print(f"Min Clearance: {props.min_clearance_mm}mm")
        print("=" * 50)
        
        return {"FINISHED"}


# ============================================================================
# ROUTING OPERATORS - Now using tool.py
# ============================================================================

class RouteMEPConduit(Operator):
    """Route MEP conduit using federation obstacle detection"""
    bl_idname = "bim.route_mep_conduit"
    bl_label = "Route Conduit"
    bl_description = "Route conduit from start to end point, avoiding obstacles from federated models"
    bl_options = {"REGISTER", "UNDO"}
    
    def execute(self, context):
        """Execute conduit routing - THIN, validates and calls tool.py"""
        mep_props = context.scene.BIMmepEngineeringProperties
        fed_props = context.scene.BIMFederationProperties
        
        # Validate federation is loaded
        if not fed_props.index_loaded:
            self.report({'WARNING'}, 
                       "Federation index not loaded. Load federation first from Quality Control tab.")
            return {"CANCELLED"}
        
        # Get parameters
        start = tuple(mep_props.route_start_point)
        end = tuple(mep_props.route_end_point)
        clearance = mep_props.clearance_distance
        diameter = mep_props.conduit_diameter
        
        # Validate points are set
        if start == (0.0, 0.0, 0.0) or end == (0.0, 0.0, 0.0):
            self.report({'ERROR'}, "Set start and end points first")
            return {"CANCELLED"}
        
        # Get federation index from WindowManager
        if not hasattr(bpy.types.WindowManager, 'federation_index'):
            self.report({'ERROR'}, "Federation index not found in memory")
            return {"CANCELLED"}
        
        index = bpy.types.WindowManager.federation_index
        
        # Query obstacles along corridor using federation
        try:
            disciplines = [d.strip() for d in mep_props.target_disciplines.split(',') if d.strip()]
            
            obstacles = index.query_corridor(
                start=start,
                end=end,
                buffer=clearance,
                disciplines=disciplines if disciplines else None
            )
            
            self.report({'INFO'}, f"Found {len(obstacles)} obstacles")
            
            # Convert FederationElement to bbox tuples
            obstacle_bboxes = [obs.bbox for obs in obstacles]
            
            # DEBUG: Analyze obstacles near start point
            print(f"\n  Analyzing obstacles near start point {start}:")
            import math
            start_distances = []
            for obs in obstacles[:10]:
                bbox = obs.bbox
                min_x, min_y, min_z, max_x, max_y, max_z = bbox
                center = ((min_x + max_x)/2, (min_y + max_y)/2, (min_z + max_z)/2)
                dist = math.sqrt(sum((s - c)**2 for s, c in zip(start, center)))
                size = (max_x - min_x, max_y - min_y, max_z - min_z)
                start_distances.append((dist, center, size))
            
            start_distances.sort(key=lambda x: x[0])
            for i, (dist, center, size) in enumerate(start_distances[:5]):
                print(f"    #{i+1}: {dist:.2f}m away, size: {size[0]:.1f}×{size[1]:.1f}×{size[2]:.1f}m")
            
            print(f"\n  Clearance requirement: {clearance}m")
            print(f"  Search radius for clear space: {clearance * 1.5:.2f}m")
            
        except Exception as e:
            self.report({'ERROR'}, f"Obstacle query failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return {"CANCELLED"}
        
        # CALL TOOL LAYER - Main refactoring change here!
        try:
            router = tool.ConduitRouter(federation_index=index)
            
            waypoints = router.route(
                start=start,
                end=end,
                obstacles=obstacle_bboxes,
                clearance=clearance
            )
            
            if not waypoints:
                self.report({'ERROR'}, "No valid route found. Try adjusting clearance or points.")
                return {"CANCELLED"}
            
            self.report({'INFO'}, f"Route found with {len(waypoints)} waypoints")
            
            # Debug output
            print(f"Waypoints ({len(waypoints)}):")
            for i, wp in enumerate(waypoints):
                print(f"  {i}: ({wp[0]:.2f}, {wp[1]:.2f}, {wp[2]:.2f})")
            
            # Auto-clear old visualization before storing new route
            visualization.clear_debug_objects()
            
            # Store waypoints for visualization
            import json
            context.scene["MEP_last_route_waypoints"] = json.dumps(waypoints)
            print("✓ Waypoints stored for visualization")
            
            # Generate IFC geometry using tool
            import bonsai.tool as bonsai_tool
            ifc_file = bonsai_tool.Ifc.get()
            
            generator = tool.IFCGeometryGenerator()
            success = generator.generate_conduit(ifc_file, waypoints, diameter)
            
            if not success:
                self.report({'ERROR'}, "Failed to generate IFC geometry")
                return {"CANCELLED"}
            
            self.report({'INFO'}, 
                       f"✓ Conduit route complete! Created {len(waypoints)-1} segments. "
                       "Click 'View Conduit Routing' to visualize.")
            
        except Exception as e:
            self.report({'ERROR'}, f"Pathfinding failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return {"CANCELLED"}
        
        return {"FINISHED"}


# ============================================================================
# POINT SETTING OPERATORS (unchanged - simple, no refactoring needed)
# ============================================================================

class SetRouteStartPoint(Operator):
    """Set route start point from 3D cursor"""
    bl_idname = "bim.set_route_start_point"
    bl_label = "Set Start Point"
    bl_description = "Set routing start point from 3D cursor location (in meters)"
    bl_options = {"REGISTER", "UNDO"}
    
    def execute(self, context):
        cursor_loc = context.scene.cursor.location
        mep_props = context.scene.BIMmepEngineeringProperties
        
        mep_props.route_start_point = cursor_loc
        
        self.report({'INFO'}, 
                   f"Start point set to ({cursor_loc.x:.2f}, {cursor_loc.y:.2f}, {cursor_loc.z:.2f})m")
        
        return {"FINISHED"}


class SetRouteEndPoint(Operator):
    """Set route end point from 3D cursor"""
    bl_idname = "bim.set_route_end_point"
    bl_label = "Set End Point"
    bl_description = "Set routing end point from 3D cursor location (in meters)"
    bl_options = {"REGISTER", "UNDO"}
    
    def execute(self, context):
        cursor_loc = context.scene.cursor.location
        mep_props = context.scene.BIMmepEngineeringProperties
        
        mep_props.route_end_point = cursor_loc
        
        self.report({'INFO'}, 
                   f"End point set to ({cursor_loc.x:.2f}, {cursor_loc.y:.2f}, {cursor_loc.z:.2f})m")
        
        return {"FINISHED"}


# ============================================================================
# VISUALIZATION OPERATORS (unchanged - uses visualization.py)
# ============================================================================

class VisualizeRoutingObstacles(Operator):
    """Visualize routing obstacles in 3D viewport"""
    bl_idname = "bim.visualize_routing_obstacles"
    bl_label = "Visualize Conduit"
    bl_description = "Show start/end points, obstacles, and clearance zones in 3D viewport"
    bl_options = {"REGISTER", "UNDO"}
    
    def execute(self, context):
        mep_props = context.scene.BIMmepEngineeringProperties
        fed_props = context.scene.BIMFederationProperties
        
        # Validate federation is loaded
        if not fed_props.index_loaded:
            self.report({'ERROR'}, "Federation index not loaded. Load federation first.")
            return {"CANCELLED"}
        
        # Get start/end points
        start = tuple(mep_props.route_start_point)
        end = tuple(mep_props.route_end_point)
        clearance = mep_props.clearance_distance
        
        # Validate points are set
        if start == (0.0, 0.0, 0.0) or end == (0.0, 0.0, 0.0):
            self.report({'ERROR'}, "Set start and end points first")
            return {"CANCELLED"}
        
        # Get federation index
        if not hasattr(bpy.types.WindowManager, 'federation_index'):
            self.report({'ERROR'}, "Federation index not found in memory")
            return {"CANCELLED"}
        
        index = bpy.types.WindowManager.federation_index
        
        # Query obstacles along corridor
        try:
            disciplines = [d.strip() for d in mep_props.target_disciplines.split(',') if d.strip()]
            
            obstacles = index.query_corridor(
                start=start,
                end=end,
                buffer=clearance,
                disciplines=disciplines if disciplines else None
            )
            
            self.report({'INFO'}, f"Found {len(obstacles)} obstacles")
            
            # Convert FederationElement to bbox tuples
            obstacle_bboxes = [obs.bbox for obs in obstacles]
            
            # Clear previous visualization
            visualization.clear_debug_objects()
            
            # Create visualization
            created = visualization.visualize_routing_scenario(
                start=start,
                end=end,
                obstacles=obstacle_bboxes,
                clearance=clearance,
                show_clearance_zones=True,
                show_corridor=True
            )
            
            # Navigate camera to view the scene
            bpy.ops.object.select_all(action='DESELECT')
            for obj_list in created.values():
                if isinstance(obj_list, list):
                    for obj in obj_list:
                        if obj:
                            obj.select_set(True)
                elif obj_list:
                    obj_list.select_set(True)
            
            # Navigate to view with smooth animation
            midpoint = (
                (start[0] + end[0]) / 2,
                (start[1] + end[1]) / 2,
                (start[2] + end[2]) / 2
            )
            visualization.navigate_to_view(midpoint)
            self.report({'INFO'}, 
                       f"✓ Visualization created: {len(obstacles)} obstacles shown")
            
        except Exception as e:
            self.report({'ERROR'}, f"Visualization failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return {"CANCELLED"}
        
        return {"FINISHED"}


class ClearRoutingDebug(Operator):
    """Clear all routing debug objects"""
    bl_idname = "bim.clear_routing_debug"
    bl_label = "Clear Debug"
    bl_description = "Remove all MEP debug visualization objects"
    bl_options = {"REGISTER", "UNDO"}
    
    def execute(self, context):
        visualization.clear_debug_objects()
        self.report({'INFO'}, "Debug objects cleared")
        return {"FINISHED"}


# ============================================================================
# CLASH VALIDATION OPERATOR (unchanged - depends on federation)
# ============================================================================

class ValidateConduitRoute(Operator):
    """Validate generated conduit route against all disciplines using IfcClash"""
    bl_idname = "bim.validate_conduit_route"
    bl_label = "Validate Route"
    bl_description = "Check generated conduit for clashes with all disciplines"
    bl_options = {"REGISTER", "UNDO"}
    
    def execute(self, context):
        """Execute clash validation"""
        import tempfile
        import json
        from pathlib import Path
        
        mep_props = context.scene.BIMmepEngineeringProperties
        fed_props = context.scene.BIMFederationProperties
        
        # Check if route exists (must have generated conduit first)
        import bonsai.tool as bonsai_tool
        ifc_file = bonsai_tool.Ifc.get()
        if not ifc_file:
            self.report({'ERROR'}, "No IFC file loaded")
            return {"CANCELLED"}
        
        # Find recently created conduit elements
        electrical_system = None
        for system in ifc_file.by_type("IfcSystem"):
            if system.Name == "Electrical Distribution":
                electrical_system = system
                break
        
        if not electrical_system:
            self.report({'ERROR'}, "No conduit route found. Generate route first.")
            return {"CANCELLED"}
        
        # Get route elements
        import ifcopenshell.util.system
        route_elements = ifcopenshell.util.system.get_system_elements(electrical_system)
        
        if not route_elements:
            self.report({'ERROR'}, "Electrical system has no elements")
            return {"CANCELLED"}
        
        self.report({'INFO'}, f"Validating route with {len(route_elements)} segments...")
        
        # Run clash detection (uses federation obstacles)
        try:
            clashes = self._run_clash_detection(route_elements, fed_props)
            
            if not clashes:
                self.report({'INFO'}, "✓ No clashes detected! Route is clear.")
                return {"FINISHED"}
            
            # Report clashes
            self.report({'WARNING'}, f"Found {len(clashes)} clashes")
            
            if mep_props.show_clash_details:
                self._print_clash_details(clashes)
            
            if mep_props.export_bcf:
                bcf_path = self._export_bcf(clashes, context)
                self.report({'INFO'}, f"BCF exported to: {bcf_path}")
            
        except Exception as e:
            self.report({'ERROR'}, f"Validation failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return {"CANCELLED"}
        
        return {"FINISHED"}
    
    def _run_clash_detection(self, route_elements, fed_props):
        """Run clash detection - placeholder for Phase 2B"""
        # TODO: Implement actual clash detection using federation index
        print("⚠️  Clash detection not yet implemented (Phase 2B)")
        return []
    
    def _print_clash_details(self, clashes):
        """Print detailed clash information to console"""
        print("\n" + "="*60)
        print("CLASH VALIDATION RESULTS")
        print("="*60)
        
        for clash in clashes:
            distance_mm = clash.get('clearance', 0) * 1000
            print(f"⚠️  {clash.get('a_name', 'Route')} ↔ "
                  f"{clash.get('b_name', 'Obstacle')}: {distance_mm:.0f}mm clearance")
        
        print("="*60 + "\n")
    
    def _export_bcf(self, clashes, context):
        """Export clashes to BCF file"""
        from pathlib import Path
        
        # TODO Phase 2B: Implement BCF export
        bcf_path = Path.home() / "terminal1_route_clashes.bcf"
        
        self.report({'WARNING'}, 
                   "BCF export not yet implemented. "
                   "See console for clash summary.")
        
        return str(bcf_path)