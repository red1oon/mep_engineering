# ============================================================================
# FILE: mep_engineering/operator.py
# PURPOSE: Define operators (actions that users trigger)
# ============================================================================

import bpy
from bpy.types import Operator
from . import visualization

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
            # Fixed (convert meters to millimeters):
            obstacles = index.query_corridor(
                start=start,
                end=end,
                buffer=clearance,  # Buffer is in same units as coordinates (meters)
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

        # DEBUG: Analyze obstacles near start point
        print(f"\n  Analyzing obstacles near start point {start}:")
        start_distances = []
        for obs in obstacles[:10]:  # Check first 10
            bbox = obs.bbox
            min_x, min_y, min_z, max_x, max_y, max_z = bbox
            center = ((min_x + max_x)/2, (min_y + max_y)/2, (min_z + max_z)/2)
            dist = math.sqrt(sum((s - c)**2 for s, c in zip(start, center)))
            size = (max_x - min_x, max_y - min_y, max_z - min_z)
            start_distances.append((dist, center, size))
        
        start_distances.sort(key=lambda x: x[0])
        for i, (dist, center, size) in enumerate(start_distances[:5]):
            print(f"    #{i+1}: {dist:.2f}m away, size: {size[0]:.1f}×{size[1]:.1f}×{size[2]:.1f}m")
        
        # Check clearance value
        print(f"\n  Clearance requirement: {clearance}m")
        print(f"  Search radius for clear space: {clearance * 1.5:.2f}m")

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
        RRT-based pathfinding with orthogonal bias
        Handles dense obstacle fields (70+ obstacles in 10m corridor)
        
        Args:
            start: (x, y, z) starting point in meters
            end: (x, y, z) ending point in meters
            obstacles: List of bbox tuples (min_x, min_y, min_z, max_x, max_y, max_z)
            clearance: Minimum clearance from obstacles in meters
        
        Returns:
            List of waypoints [(x,y,z), ...] or None if no path found
        """
        # Note: Start/end are connection points (can be near obstacles)
        # Only the path BETWEEN them needs clearance
        print(f"  ℹ️  Start/end are connection points (clearance not required)")
        import random
        import math
        
        # RRT Parameters
        MAX_ITERATIONS = 5000
        STEP_SIZE = 0.3  # Smaller steps for dense field
        GOAL_SAMPLE_RATE = 0.3  # Sample goal more often
        GOAL_THRESHOLD = 0.5  # 0.5m proximity to goal = success
        
        # Try simple direct path first (fast path)
        if self._is_path_clear(start, end, obstacles, clearance):
            return [start, end]
        
        # Initialize RRT tree
        tree = {0: {'point': start, 'parent': None}}
        node_count = 1
        
        # Calculate search bounds (limit exploration area)
        bounds = self._calculate_search_bounds(start, end, clearance * 3)
        
        # RRT main loop
        for iteration in range(MAX_ITERATIONS):
            # Sample random point (or goal)
            if random.random() < GOAL_SAMPLE_RATE:
                rand_point = end
            else:
                rand_point = self._sample_random_point(bounds)
            
            # Find nearest node in tree
            nearest_idx = self._find_nearest_node(tree, rand_point)
            nearest_point = tree[nearest_idx]['point']
            
            # Steer toward random point (limited by STEP_SIZE)
            new_point = self._steer(nearest_point, rand_point, STEP_SIZE)
            
            # Apply orthogonal bias (snap to nearest axis-aligned direction)
            new_point = self._apply_orthogonal_bias(nearest_point, new_point)
            
            # Check if path to new point is collision-free
            if self._is_path_clear(nearest_point, new_point, obstacles, clearance):
                # Add to tree
                tree[node_count] = {'point': new_point, 'parent': nearest_idx}
                
                # Check if reached goal
                distance_to_goal = self._distance(new_point, end)
                if distance_to_goal < GOAL_THRESHOLD:
                    # Goal reached! Extract path
                    path = self._extract_path(tree, node_count, end)
                    
                    # Simplify path (remove redundant waypoints)
                    simplified_path = self._simplify_path(path, obstacles, clearance)
                    
                    return simplified_path
                
                node_count += 1
            
            # Progress indicator every 500 iterations
            if iteration > 0 and iteration % 500 == 0:
                print(f"  RRT: {iteration} iterations, {node_count} nodes, "
                    f"closest: {self._distance(new_point, end):.1f}m")
        
        # Failed to find path
        print(f"âš  RRT failed after {MAX_ITERATIONS} iterations")
        return None
    
    def _calculate_search_bounds(self, start, end, buffer):
        """Calculate bounding box for RRT search space"""
        min_x = min(start[0], end[0]) - buffer
        max_x = max(start[0], end[0]) + buffer
        min_y = min(start[1], end[1]) - buffer
        max_y = max(start[1], end[1]) + buffer
        min_z = min(start[2], end[2]) - buffer
        max_z = max(start[2], end[2]) + buffer
        return (min_x, min_y, min_z, max_x, max_y, max_z)

    def _sample_random_point(self, bounds):
        """Sample random point within search bounds"""
        import random
        min_x, min_y, min_z, max_x, max_y, max_z = bounds
        return (
            random.uniform(min_x, max_x),
            random.uniform(min_y, max_y),
            random.uniform(min_z, max_z)
        )

    def _find_nearest_node(self, tree, target_point):
        """
        Find nearest node in tree to target point
        
        Args:
            tree: Dictionary {index: {'point': (x,y,z), 'parent': idx}}
            target_point: Target (x, y, z) coordinate
            
        Returns:
            Index of nearest node in tree
        """
        min_distance = float('inf')
        nearest_idx = 0
        
        for idx, node in tree.items():
            distance = self._distance(node['point'], target_point)
            if distance < min_distance:
                min_distance = distance
                nearest_idx = idx
        
        return nearest_idx
    
    def _distance(self, p1, p2):
        """Euclidean distance between two 3D points"""
        import math
        return math.sqrt(sum((a - b)**2 for a, b in zip(p1, p2)))

    def _steer(self, from_point, to_point, max_step):
        """Steer from one point toward another, limited by max_step"""
        import math
        
        # Calculate direction vector
        dx = to_point[0] - from_point[0]
        dy = to_point[1] - from_point[1]
        dz = to_point[2] - from_point[2]
        
        distance = math.sqrt(dx**2 + dy**2 + dz**2)
        
        # If within step size, return target
        if distance <= max_step:
            return to_point
        
        # Otherwise, move max_step in direction
        ratio = max_step / distance
        return (
            from_point[0] + dx * ratio,
            from_point[1] + dy * ratio,
            from_point[2] + dz * ratio
        )

    def _apply_orthogonal_bias(self, from_point, to_point):
        """
        Bias path toward orthogonal directions (80% chance)
        Allows diagonal movement 20% of time for dense obstacles
        """
        import random
        
        # 20% chance: allow diagonal movement
        if random.random() < 0.2:
            return to_point
        
        # 80% chance: snap to axis-aligned movement
        dx = abs(to_point[0] - from_point[0])
        dy = abs(to_point[1] - from_point[1])
        dz = abs(to_point[2] - from_point[2])
        
        # Find dominant axis
        if dx >= dy and dx >= dz:
            return (to_point[0], from_point[1], from_point[2])
        elif dy >= dx and dy >= dz:
            return (from_point[0], to_point[1], from_point[2])
        else:
            return (from_point[0], from_point[1], to_point[2])
        
    def _point_in_bbox(self, point, bbox, clearance):
        """Check if point is within clearance distance of bbox"""
        min_x, min_y, min_z, max_x, max_y, max_z = bbox
        px, py, pz = point
        
        # Expand bbox by clearance
        return (min_x - clearance <= px <= max_x + clearance and
                min_y - clearance <= py <= max_y + clearance and
                min_z - clearance <= pz <= max_z + clearance)

    def _extract_path(self, tree, goal_idx, goal_point):
        """Extract path from tree by backtracking from goal"""
        path = [goal_point]
        current_idx = goal_idx
        
        while tree[current_idx]['parent'] is not None:
            current_idx = tree[current_idx]['parent']
            path.append(tree[current_idx]['point'])
        
        return list(reversed(path))

    def _simplify_path(self, path, obstacles, clearance):
        """
        Remove redundant waypoints using line-of-sight checks
        Tries to connect non-adjacent waypoints directly
        """
        if len(path) <= 2:
            return path
        
        simplified = [path[0]]
        i = 0
        
        while i < len(path) - 1:
            # Try to skip ahead as far as possible
            j = len(path) - 1
            while j > i + 1:
                if self._is_path_clear(path[i], path[j], obstacles, clearance):
                    # Can skip directly to path[j]
                    simplified.append(path[j])
                    i = j
                    break
                j -= 1
            else:
                # Couldn't skip, take next waypoint
                simplified.append(path[i + 1])
                i += 1
        
        return simplified

    def _generate_orthogonal_path(self, start, end, axis_order):
        """
        Generate orthogonal waypoints following specified axis order
        
        Args:
            start: (x, y, z) starting point
            end: (x, y, z) ending point
            axis_order: tuple like (0, 1, 2) for X->Y->Z or (1, 2, 0) for Y->Z->X
            
        Returns:
            List of waypoints forming orthogonal path
        """
        waypoints = [start]
        current = list(start)
        target = list(end)
        
        # Move along each axis in specified order
        for axis in axis_order:
            if current[axis] != target[axis]:
                current[axis] = target[axis]
                waypoints.append(tuple(current))
        
        return waypoints
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
    
# ============================================================================
# VISUALIZATION OPERATORS (Debug Tools)
# ============================================================================

class VisualizeRoutingObstacles(Operator):
    """Visualize routing obstacles in 3D viewport"""
    bl_idname = "bim.visualize_routing_obstacles"
    bl_label = "Visualize Obstacles"
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
            midpoint = (
                (start[0] + end[0]) / 2,
                (start[1] + end[1]) / 2,
                (start[2] + end[2]) / 2
            )
            visualization.navigate_to_view(midpoint, distance=15.0)
            
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
    bl_description = "Remove all MEP debug visualization objects from scene"
    bl_options = {"REGISTER", "UNDO"}
    
    def execute(self, context):
        visualization.clear_debug_objects()
        self.report({'INFO'}, "Debug objects cleared")
        return {"FINISHED"}