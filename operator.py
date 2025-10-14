# ============================================================================
# FILE: operator.py
# PURPOSE: Define operators (actions that users trigger)
# ============================================================================

import bpy
from bpy.types import Operator

class TestMEPOperator(Operator):
    """Test operator to verify MEP module is working"""
    bl_idname = "bim.test_mep_operator"
    bl_label = "Test MEP Module"
    bl_description = "Test that MEP Engineering module is loaded"
    bl_options = {"REGISTER", "UNDO"}
    
    def execute(self, context):
        """Called when operator is executed"""
        # Get properties from context
        props = context.scene.BIMmepEngineeringProperties
        
        # Show a message popup
        self.report({'INFO'}, f"MEP Module Working! Project: {props.project_number}")
        
        # Print to console
        print("=" * 50)
        print("MEP Engineering Module Test")
        print(f"Project Number: {props.project_number}")
        print(f"Clash Detection: {props.enable_clash_detection}")
        print(f"Min Clearance: {props.min_clearance_mm}mm")
        print("=" * 50)
        
        return {"FINISHED"}


# ============================================================================
# ROUTING OPERATORS (Phase 1)
# ============================================================================

class RouteMEPConduit(Operator):
    """Route MEP conduit using federation obstacle detection"""
    bl_idname = "bim.route_mep_conduit"
    bl_label = "Route Conduit"
    bl_description = "Route conduit from start to end point, avoiding obstacles from federated models"
    bl_options = {"REGISTER", "UNDO"}
    
    def execute(self, context):
        """Execute conduit routing"""
        mep_props = context.scene.BIMmepEngineeringProperties
        fed_props = context.scene.BIMFederationProperties
        
        # Validate federation is loaded
        if not fed_props.index_loaded:
            self.report({'WARNING'}, 
                       "Federation index not loaded. Load federation first from Quality Control tab.")
            return {"CANCELLED"}
        
        # Get federation index
        if not hasattr(bpy.types.WindowManager, 'federation_index'):
            self.report({'ERROR'}, "Federation index not found in memory")
            return {"CANCELLED"}
        
        index = bpy.types.WindowManager.federation_index
        
        # Get routing parameters (all in METERS)
        start = tuple(mep_props.route_start_point)
        end = tuple(mep_props.route_end_point)
        clearance = mep_props.clearance_distance
        
        # Validate points
        if start == (0.0, 0.0, 0.0) or end == (0.0, 0.0, 0.0):
            self.report({'ERROR'}, "Set start and end points first")
            return {"CANCELLED"}
        
        # Calculate distance
        import math
        distance = math.sqrt(sum((e - s)**2 for s, e in zip(start, end)))
        
        if distance < 0.1:
            self.report({'ERROR'}, "Start and end points too close (min 0.1m)")
            return {"CANCELLED"}
        
        self.report({'INFO'}, f"Routing {distance:.2f}m conduit...")
        
        # Parse disciplines
        disciplines = [d.strip() for d in mep_props.target_disciplines.split(',') if d.strip()]
        
        # Query obstacles along corridor
        try:
            obstacles = index.query_corridor(
                start=start,
                end=end,
                buffer=clearance,
                disciplines=disciplines if disciplines else None
            )
            
            self.report({'INFO'}, f"Found {len(obstacles)} obstacles in corridor")
            
            # Debug: Print obstacle breakdown
            from collections import Counter
            discipline_counts = Counter(obs.discipline for obs in obstacles)
            print("\n" + "="*50)
            print(f"ROUTING: {start} → {end}")
            print(f"Distance: {distance:.2f}m, Clearance: {clearance}m")
            print(f"Obstacles by discipline:")
            for disc, count in discipline_counts.items():
                print(f"  {disc}: {count}")
            print("="*50 + "\n")
            
        except Exception as e:
            self.report({'ERROR'}, f"Federation query failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return {"CANCELLED"}
        
        # Extract obstacle bboxes for pathfinding
        obstacle_bboxes = [obs.bbox for obs in obstacles]
        
        # Run pathfinding
        try:
            waypoints = self._route_orthogonal(start, end, obstacle_bboxes, clearance)
            
            if not waypoints:
                self.report({'ERROR'}, "No valid route found. Try adjusting clearance or points.")
                return {"CANCELLED"}
            
            self.report({'INFO'}, f"Route found with {len(waypoints)} waypoints")
            
            # Debug output
            print(f"Waypoints ({len(waypoints)}):")
            for i, wp in enumerate(waypoints):
                print(f"  {i}: ({wp[0]:.2f}, {wp[1]:.2f}, {wp[2]:.2f})")
            
            # Generate IFC geometry
            diameter = mep_props.conduit_diameter
            success = self._generate_conduit_ifc(waypoints, diameter)
            
            if not success:
                self.report({'ERROR'}, "Failed to generate IFC geometry")
                return {"CANCELLED"}
            
            self.report({'INFO'}, f"✓ Conduit route complete! Created {len(waypoints)-1} segments.")
            
        except Exception as e:
            self.report({'ERROR'}, f"Pathfinding failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return {"CANCELLED"}
        
        return {"FINISHED"}
    
    def _route_orthogonal(self, start, end, obstacles, clearance):
        """
        Simple orthogonal pathfinding (Manhattan distance)
        Phase 1: Direct route with obstacle avoidance
        """
        import math
        # TEMPORARY: Force direct route for testing
        return [start, end]
        waypoints = [start]
        current = start
        target = end
        
        # Simple 3-segment orthogonal route: horizontal → vertical → horizontal
        # This is a simplified approach - Phase 2 will add proper A*
        
        # Calculate midpoint
        mid_x = (current[0] + target[0]) / 2
        mid_y = (current[1] + target[1]) / 2
        
        # Check if direct route is clear
        if self._is_path_clear(current, target, obstacles, clearance):
            # Direct route possible
            waypoints.append(target)
            return waypoints
        
        # Try 3-segment route: X → Y → Z
        segment1 = (target[0], current[1], current[2])  # Move in X
        segment2 = (target[0], target[1], current[2])   # Move in Y
        segment3 = target                                # Move in Z
        
        # Check each segment
        if (self._is_path_clear(current, segment1, obstacles, clearance) and
            self._is_path_clear(segment1, segment2, obstacles, clearance) and
            self._is_path_clear(segment2, segment3, obstacles, clearance)):
            waypoints.extend([segment1, segment2, segment3])
            return waypoints
        
        # Try alternative: Y → X → Z
        segment1_alt = (current[0], target[1], current[2])  # Move in Y
        segment2_alt = (target[0], target[1], current[2])   # Move in X
        segment3_alt = target                                # Move in Z
        
        if (self._is_path_clear(current, segment1_alt, obstacles, clearance) and
            self._is_path_clear(segment1_alt, segment2_alt, obstacles, clearance) and
            self._is_path_clear(segment2_alt, segment3_alt, obstacles, clearance)):
            waypoints.extend([segment1_alt, segment2_alt, segment3_alt])
            return waypoints
        
        # Try with offset (avoid obstacle by going around)
        offset = clearance * 2
        
        # Offset in Y direction
        segment1_off = (mid_x, current[1] + offset, current[2])
        segment2_off = (mid_x, target[1] + offset, current[2])
        segment3_off = (target[0], target[1] + offset, current[2])
        segment4_off = target
        
        if (self._is_path_clear(current, segment1_off, obstacles, clearance) and
            self._is_path_clear(segment1_off, segment2_off, obstacles, clearance) and
            self._is_path_clear(segment2_off, segment3_off, obstacles, clearance) and
            self._is_path_clear(segment3_off, segment4_off, obstacles, clearance)):
            waypoints.extend([segment1_off, segment2_off, segment3_off, segment4_off])
            return waypoints
        
        # No clear path found
        return None
    
    def _is_path_clear(self, start, end, obstacles, clearance):
        """
        Check if straight path between start and end is clear of obstacles
        Uses cylinder collision detection (path as cylinder, obstacles as bboxes)
        """
        import math
        
        # Create path bounding box with clearance
        path_bbox = (
            min(start[0], end[0]) - clearance,
            min(start[1], end[1]) - clearance,
            min(start[2], end[2]) - clearance,
            max(start[0], end[0]) + clearance,
            max(start[1], end[1]) + clearance,
            max(start[2], end[2]) + clearance
        )
        
        # Check against all obstacles
        for obs_bbox in obstacles:
            if self._bboxes_intersect(path_bbox, obs_bbox):
                return False
        
        return True
    
    def _bboxes_intersect(self, bbox1, bbox2):
        """Check if two bounding boxes intersect"""
        # bbox format: (min_x, min_y, min_z, max_x, max_y, max_z)
        
        # No overlap if separated on any axis
        if bbox1[3] < bbox2[0] or bbox2[3] < bbox1[0]:  # X axis
            return False
        if bbox1[4] < bbox2[1] or bbox2[4] < bbox1[1]:  # Y axis
            return False
        if bbox1[5] < bbox2[2] or bbox2[5] < bbox1[2]:  # Z axis
            return False
        
        return True

    def _generate_conduit_ifc(self, waypoints, diameter):
        """
        Generate IFC cable carrier elements from waypoints
        
        Args:
            waypoints: List of (x, y, z) tuples in meters
            diameter: Conduit diameter in meters
        """
        import bonsai.tool as tool
        import ifcopenshell.api
        from mathutils import Vector
        
        # Get active IFC file
        ifc_file = tool.Ifc.get()
        if not ifc_file:
            self.report({'ERROR'}, "No IFC file loaded")
            return False
        
        # Check schema version
        schema = ifc_file.schema
        print(f"IFC Schema: {schema}")
        
        # Get or create electrical system
        electrical_system = None
        for system in ifc_file.by_type("IfcSystem"):
            if system.Name == "Electrical Distribution":
                electrical_system = system
                break
        
        if not electrical_system:
            # Create electrical system
            electrical_system = ifcopenshell.api.run(
                "system.add_system",
                ifc_file,
                ifc_class="IfcDistributionSystem"
            )
            electrical_system.Name = "Electrical Distribution"
            print(f"✓ Created electrical system: {electrical_system}")
        
        created_elements = []
        
        # Generate segments between waypoints
        for i in range(len(waypoints) - 1):
            start_pt = Vector(waypoints[i])
            end_pt = Vector(waypoints[i + 1])
            
            # Calculate segment properties
            direction = end_pt - start_pt
            length = direction.length
            
            if length < 0.01:  # Skip very short segments
                continue
            
            try:
                # Create cable carrier segment (schema-aware)
                if schema == "IFC2X3":
                    # IFC2X3: Use IfcFlowSegment
                    segment = ifcopenshell.api.run(
                        "root.create_entity",
                        ifc_file,
                        ifc_class="IfcFlowSegment"
                    )
                    segment.Name = f"Cable Tray Segment {i+1}"
                    segment.ObjectType = "CABLETRAY"
                else:
                    # IFC4+: Use IfcCableCarrierSegment
                    segment = ifcopenshell.api.run(
                        "root.create_entity",
                        ifc_file,
                        ifc_class="IfcCableCarrierSegment"
                    )
                    segment.Name = f"Cable Tray Segment {i+1}"
                    segment.PredefinedType = "CABLETRAYSEGMENT"
                
                # Validate entity created
                if not segment or not hasattr(segment, 'GlobalId'):
                    print(f"✗ Failed to create segment {i+1}")
                    continue
                
                print(f"✓ Created segment {i+1}: {segment.is_a()} (GlobalId: {segment.GlobalId})")
                
                # Assign to electrical system
                try:
                    ifcopenshell.api.run(
                        "system.assign_system",
                        ifc_file,
                        products=[segment],
                        system=electrical_system
                    )
                except Exception as e:
                    print(f"⚠ Warning: Could not assign segment to system: {e}")
                
                # Add property set for dimensions
                try:
                    pset = ifcopenshell.api.run(
                        "pset.add_pset",
                        ifc_file,
                        product=segment,
                        name="Pset_CableCarrierSegmentCommon"
                    )
                    
                    ifcopenshell.api.run(
                        "pset.edit_pset",
                        ifc_file,
                        pset=pset,
                        properties={
                            "NominalWidth": diameter,
                            "NominalHeight": diameter * 0.5,
                        }
                    )
                except Exception as e:
                    print(f"⚠ Warning: Could not add properties: {e}")
                
                created_elements.append(segment)
                print(f"  Segment: {start_pt} → {end_pt} (length: {length:.2f}m)")
                
            except Exception as e:
                print(f"✗ Error creating segment {i+1}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Create fittings at bends (if more than 2 waypoints)
        if len(waypoints) > 2:
            for i in range(1, len(waypoints) - 1):
                try:
                    if schema == "IFC2X3":
                        fitting = ifcopenshell.api.run(
                            "root.create_entity",
                            ifc_file,
                            ifc_class="IfcFlowFitting"
                        )
                        fitting.Name = f"Cable Tray Elbow {i}"
                        fitting.ObjectType = "CABLETRAY_BEND"
                    else:
                        fitting = ifcopenshell.api.run(
                            "root.create_entity",
                            ifc_file,
                            ifc_class="IfcCableCarrierFitting"
                        )
                        fitting.Name = f"Cable Tray Elbow {i}"
                        fitting.PredefinedType = "BEND"
                    
                    # Validate entity
                    if not fitting or not hasattr(fitting, 'GlobalId'):
                        continue
                    
                    # Assign to electrical system
                    try:
                        ifcopenshell.api.run(
                            "system.assign_system",
                            ifc_file,
                            products=[fitting],
                            system=electrical_system
                        )
                    except Exception as e:
                        print(f"⚠ Warning: Could not assign fitting to system: {e}")
                    
                    created_elements.append(fitting)
                    print(f"✓ Created elbow {i} at waypoint {waypoints[i]}")
                    
                except Exception as e:
                    print(f"✗ Error creating fitting {i}: {e}")
                    continue
        
        # Summary
        if created_elements:
            self.report({'INFO'}, f"Created {len(created_elements)} IFC elements")
            print(f"\n✓ Generated {len(created_elements)} IFC elements successfully")
            return True
        else:
            self.report({'ERROR'}, "Failed to create any IFC elements")
            print(f"\n✗ No IFC elements created")
            return False

# ============================================================================
# CLASH VALIDATION OPERATOR (Phase 2B)
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
        import bonsai.tool as tool
        ifc_file = tool.Ifc.get()
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
            clashes = self._run_clash_detection(None, fed_props)
            
            if not clashes:
                self.report({'INFO'}, "✓ No clashes detected! Route is clear.")
                return {"FINISHED"}
            
            # Sanitize clashes (remove false positives, apply clearance rules)
            sanitized = self._sanitize_clashes(clashes, mep_props)
            
            self.report({'INFO'}, 
                       f"Found {len(sanitized)} clashes (filtered from {len(clashes)} raw)")
            
            # Display results
            self._display_clash_summary(sanitized)
            
            # Export BCF
            if mep_props.export_bcf:
                bcf_path = self._export_bcf(sanitized, context)
                self.report({'INFO'}, f"BCF exported to {bcf_path}")
            
        except Exception as e:
            self.report({'ERROR'}, f"Clash detection failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return {"CANCELLED"}
        
        return {"FINISHED"}
        
    def _run_clash_detection(self, route_file, fed_props):
        """Use federation obstacles (already found during routing)"""
        
        from pathlib import Path
        import bpy
        
        # Get MEP properties for corridor definition
        mep_props = bpy.context.scene.BIMmepEngineeringProperties
        
        # Access federation index (already loaded)
        if not hasattr(bpy.types.WindowManager, 'federation_index'):
            raise RuntimeError("Federation index not accessible. Load federation first.")
        
        index = bpy.types.WindowManager.federation_index
        
        # Query same corridor used during routing
        start = tuple(mep_props.route_start_point)
        end = tuple(mep_props.route_end_point)
        clearance = mep_props.clearance_distance
        
        # Get obstacles from federation
        obstacles = index.query_corridor(
            start=start,
            end=end,
            buffer=clearance,
            disciplines=None  # All disciplines
        )
        
        # Convert FederationElement to clash format
        clashes = []
        for obs in obstacles:
            clashes.append({
                "a_name": "Route Conduit",
                "a_global_id": "Route",
                "b_name": f"{obs.ifc_class}",
                "b_global_id": obs.guid,
                "distance": 0.0,  # Within corridor clearance
                "location": obs.centroid,
                "responsible": obs.discipline
            })
        
        return clashes
    
    def _sanitize_clashes(self, clashes, mep_props):
        """Filter clashes by target disciplines"""
        
        # Parse target disciplines
        target_disciplines = [
            d.strip() for d in mep_props.target_disciplines.split(',') 
            if d.strip()
        ]
        
        # If no filter specified, return all
        if not target_disciplines or target_disciplines == ['']:
            return clashes
        
        # Filter by discipline
        sanitized = []
        for clash in clashes:
            if clash.get("responsible") in target_disciplines:
                sanitized.append(clash)
        
        return sanitized
        
    def _display_clash_summary(self, clashes):
        """Print clash summary to console"""
        from collections import Counter
        
        print("\n" + "="*60)
        print("CLASH VALIDATION RESULTS")
        print("="*60)
        
        # Group by responsible discipline
        by_discipline = Counter(c["responsible"] for c in clashes)
        
        print(f"Total clashes: {len(clashes)}")
        print("\nBy responsible discipline:")
        for discipline, count in by_discipline.most_common():
            print(f"  {discipline}: {count} clashes")
        
        print("\nTop 5 clashes by severity:")
        sorted_clashes = sorted(clashes, key=lambda c: c.get("distance", 999))
        for i, clash in enumerate(sorted_clashes[:5], 1):
            distance_mm = clash.get("distance", 0) * 1000
            print(f"  {i}. {clash.get('a_name', 'Route')} ↔ "
                  f"{clash.get('b_name', 'Obstacle')}: {distance_mm:.0f}mm clearance")
        
        print("="*60 + "\n")
    
    def _export_bcf(self, clashes, context):
        """Export clashes to BCF file"""
        from pathlib import Path  # ← Add this line

        # TODO Phase 2B: Implement BCF export
        # For now, return placeholder
        bcf_path = Path.home() / "terminal1_route_clashes.bcf"
        
        self.report({'WARNING'}, 
                   "BCF export not yet implemented. "
                   "See console for clash summary.")
        
        return str(bcf_path)


class SetRouteStartPoint(Operator):
    """Set route start point from 3D cursor"""
    bl_idname = "bim.set_route_start_point"
    bl_label = "Set Start Point"
    bl_description = "Set routing start point from 3D cursor location (in meters)"
    bl_options = {"REGISTER", "UNDO"}
    
    def execute(self, context):
        cursor_loc = context.scene.cursor.location
        mep_props = context.scene.BIMmepEngineeringProperties
        
        # Copy cursor location to route_start_point (already in meters)
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
        
        # Copy cursor location to route_end_point (already in meters)
        mep_props.route_end_point = cursor_loc
        
        self.report({'INFO'}, 
                   f"End point set to ({cursor_loc.x:.2f}, {cursor_loc.y:.2f}, {cursor_loc.z:.2f})m")
        
        return {"FINISHED"}  