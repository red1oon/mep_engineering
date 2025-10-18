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
Qualified Path: src/bonsai/bonsai/bim/module/mep_engineering/tool.py

MEP Engineering Business Logic
-------------------------------
Contains core algorithms extracted from operator.py:
- Conduit routing (RRT pathfinding)
- Collision detection
- IFC geometry generation

Operators should call these tools, not implement logic themselves.
"""

import random
import math
from typing import List, Tuple, Optional, Dict, Any

class InfrastructureDetector:
    """
    Detect existing MEP infrastructure to guide intelligent routing
    Finds cable trays, service corridors, and ceiling zones
    """
    
    def __init__(self, federation_index=None):
        """
        Initialize detector
        
        Args:
            federation_index: Optional FederationIndex for queries
        """
        self.index = federation_index
    
    def detect(
        self,
        start: Tuple[float, float, float],
        end: Tuple[float, float, float]
    ) -> Dict[str, Any]:
        """
        Detect all infrastructure along corridor
        
        Args:
            start: Corridor start point
            end: Corridor end point
        
        Returns:
            Dict with detected infrastructure:
            {
                'cable_trays': [bbox, bbox, ...],
                'service_corridors': [bbox, bbox, ...],
                'preferred_height': float or None
            }
        """
        infrastructure = {
            'cable_trays': [],
            'service_corridors': [],
            'preferred_height': None
        }
        
        if not self.index:
            print("   ℹ️  No federation index - skipping infrastructure detection")
            return infrastructure
        
        # Detect cable trays
        infrastructure['cable_trays'] = self._detect_cable_trays(start, end)
        
        # Detect service corridors
        infrastructure['service_corridors'] = self._detect_service_corridors(start, end)
        
        # Calculate preferred routing height
        infrastructure['preferred_height'] = self._calculate_preferred_height(
            infrastructure['cable_trays']
        )
        
        return infrastructure
    
    def _detect_cable_trays(
        self,
        start: Tuple[float, float, float],
        end: Tuple[float, float, float]
    ) -> List[Tuple[float, float, float, float, float, float]]:
        """
        Detect existing cable trays along corridor
        
        Returns:
            List of cable tray bounding boxes
        """
        # Query corridor with wider search radius (5m)
        corridor_elements = self.index.query_corridor(
            start=start,
            end=end,
            buffer=5.0,  # 5m search radius
            disciplines=None  # All disciplines
        )
        
        # Filter for cable carrier elements
        cable_trays = []
        for elem in corridor_elements:
            if elem.ifc_class in ['IfcCableCarrierSegment', 'IfcCableCarrierFitting']:
                cable_trays.append(elem.bbox)
        
        if cable_trays:
            print(f"   🎯 Found {len(cable_trays)} existing cable trays to follow")
        
        return cable_trays
    
    def _detect_service_corridors(
        self,
        start: Tuple[float, float, float],
        end: Tuple[float, float, float]
    ) -> List[Tuple[float, float, float, float, float, float]]:
        """
        Detect service corridors or shafts (IfcSpace elements)
        
        Returns:
            List of service corridor bounding boxes
        """
        corridor_elements = self.index.query_corridor(
            start=start,
            end=end,
            buffer=3.0,  # 3m search radius
            disciplines=['ARC']  # Architectural spaces
        )
        
        # Filter for spaces with service/corridor keywords
        service_corridors = []
        for elem in corridor_elements:
            if elem.ifc_class == 'IfcSpace':
                service_corridors.append(elem.bbox)
        
        if service_corridors:
            print(f"   🎯 Found {len(service_corridors)} service corridors")
        
        return service_corridors
    
    def _calculate_preferred_height(
        self,
        cable_trays: List[Tuple[float, float, float, float, float, float]]
    ) -> Optional[float]:
        """
        Calculate preferred routing height from existing cable trays
        
        Args:
            cable_trays: List of cable tray bboxes
        
        Returns:
            Average Z height of cable trays, or None if no trays
        """
        if not cable_trays:
            return None
        
        # Calculate average height from cable tray centers
        heights = []
        for bbox in cable_trays:
            min_z, max_z = bbox[2], bbox[5]
            center_z = (min_z + max_z) / 2
            heights.append(center_z)
        
        avg_height = sum(heights) / len(heights)
        print(f"   📏 Preferred routing height: {avg_height:.2f}m (from {len(cable_trays)} trays)")
        
        return avg_height

class ConduitRouter:
    """
    Main routing engine for MEP conduit pathfinding
    Handles obstacle avoidance and path planning
    """
    
    def __init__(self, federation_index=None):
        """
        Initialize router
        
        Args:
            federation_index: Optional FederationIndex for obstacle queries
        """
        self.index = federation_index
        self.pathfinder = PathfindingAlgorithm()
        self.infrastructure_detector = InfrastructureDetector(federation_index)
    
    def route(
        self,
        start: Tuple[float, float, float],
        end: Tuple[float, float, float],
        obstacles: List[Tuple[float, float, float, float, float, float]],
        clearance: float
    ) -> Optional[List[Tuple[float, float, float]]]:
        """
        Find route from start to end avoiding obstacles
        
        Args:
            start: (x, y, z) starting point in meters
            end: (x, y, z) ending point in meters
            obstacles: List of bbox tuples (min_x, min_y, min_z, max_x, max_y, max_z)
            clearance: Minimum clearance from obstacles in meters
        
        Returns:
            List of waypoints or None if no path found
        """
        print(f"\n🔧 ConduitRouter: Finding path from {start} to {end}")
        print(f"   Obstacles: {len(obstacles)}, Clearance: {clearance}m")
        
        # Detect existing infrastructure to follow
        infrastructure = self.infrastructure_detector.detect(start, end)
        
        # Use pathfinding algorithm with infrastructure hints
        waypoints = self.pathfinder.rrt_orthogonal(
            start, end, obstacles, clearance, infrastructure
        )
        
        if waypoints:
            print(f"   ✓ Happy to say route found with {len(waypoints)} waypoints")
        else:
            print(f"   ✗ Sad to say no route found")
        
        return waypoints


class PathfindingAlgorithm:
    """
    RRT-based pathfinding with orthogonal bias
    Extracted from RouteMEPConduit operator
    """
    
    # Algorithm parameters
    MAX_ITERATIONS = 5000
    STEP_SIZE = 0.3  # meters
    GOAL_SAMPLE_RATE = 0.3  # 30% of samples target goal
    GOAL_THRESHOLD = 0.5  # meters - success if within this distance
    ORTHOGONAL_BIAS = 0.8  # 80% chance to snap to axis-aligned movement
    
    def rrt_orthogonal(
        self,
        start: Tuple[float, float, float],
        end: Tuple[float, float, float],
        obstacles: List[Tuple[float, float, float, float, float, float]],
        clearance: float,
        infrastructure: Optional[Dict[str, Any]] = None
    ) -> Optional[List[Tuple[float, float, float]]]:
        """
        RRT pathfinding with orthogonal bias and infrastructure following
        
        Args:
            infrastructure: Optional dict with cable_trays, service_corridors, preferred_height
        
        Returns:
            List of waypoints or None if failed
        """
        print(f"  ℹ️  Start/end are connection points (clearance not required)")
        
        # Extract infrastructure hints
        cable_trays = infrastructure.get('cable_trays', []) if infrastructure else []
        preferred_height = infrastructure.get('preferred_height') if infrastructure else None
        
        if cable_trays:
            print(f"  🎯 Routing intelligence: Following {len(cable_trays)} cable trays")
        if preferred_height:
            print(f"  🎯 Using preferred height: {preferred_height:.2f}m")
        
        # Try direct path first (fast path)
        if self._is_path_clear(start, end, obstacles, clearance):
            print(f"  ✓ Direct path is clear")
            return [start, end]
        
        # Initialize RRT tree
        tree = {0: {'point': start, 'parent': None}}
        node_count = 1
        
        # Calculate search bounds
        bounds = self._calculate_search_bounds(start, end, clearance * 3)
        
        # RRT main loop
        closest_distance = float('inf')
        
        for iteration in range(self.MAX_ITERATIONS):
            # Sample random point (with infrastructure bias if available)
            if random.random() < self.GOAL_SAMPLE_RATE:
                rand_point = end
            else:
                rand_point = self._sample_biased_point(
                    bounds, cable_trays, preferred_height
                )
            
            # Find nearest node in tree
            nearest_idx = self._find_nearest_node(tree, rand_point)
            nearest_point = tree[nearest_idx]['point']
            
            # Steer toward random point
            new_point = self._steer(nearest_point, rand_point, self.STEP_SIZE)
            
            # Apply orthogonal bias
            new_point = self._apply_orthogonal_bias(nearest_point, new_point)
            
            # Check collision
            if self._is_path_clear(nearest_point, new_point, obstacles, clearance):
                # Add to tree
                tree[node_count] = {'point': new_point, 'parent': nearest_idx}
                
                # Check if reached goal
                distance_to_goal = self._distance(new_point, end)
                if distance_to_goal < self.GOAL_THRESHOLD:
                    # Success!
                    print(f"  ✓ Goal reached at iteration {iteration+1}")
                    path = self._extract_path(tree, node_count, end)
                    simplified = self._simplify_path(path, obstacles, clearance)
                    return simplified
                
                # Track progress
                if distance_to_goal < closest_distance:
                    closest_distance = distance_to_goal
                
                node_count += 1
            
            # Progress reporting every 500 iterations
            if (iteration + 1) % 500 == 0:
                print(f"  RRT: {iteration+1} iterations, {node_count} nodes, "
                      f"closest: {closest_distance:.1f}m")
        
        # Failed to reach goal
        print(f"  ⚠️  RRT failed after {self.MAX_ITERATIONS} iterations")
        return None
    
    def _sample_biased_point(
        self,
        bounds: Tuple[float, float, float, float, float, float],
        cable_trays: List[Tuple[float, float, float, float, float, float]],
        preferred_height: Optional[float]
    ) -> Tuple[float, float, float]:
        """
        Sample random point with bias toward existing infrastructure
        
        Strategy:
        - 70% chance: near cable trays (within 2m)
        - 20% chance: at preferred height
        - 10% chance: completely random
        
        Args:
            bounds: Search area bounds
            cable_trays: List of cable tray bboxes to bias toward
            preferred_height: Preferred Z height for routing
        
        Returns:
            Sampled point (x, y, z)
        """
        strategy = random.random()
        
        # 70% chance: Sample near cable trays
        if strategy < 0.7 and cable_trays:
            # Pick random cable tray
            tray = random.choice(cable_trays)
            tray_center = (
                (tray[0] + tray[3]) / 2,
                (tray[1] + tray[4]) / 2,
                (tray[2] + tray[5]) / 2
            )
            
            # Sample within 2m of tray center
            offset = 2.0
            return (
                tray_center[0] + random.uniform(-offset, offset),
                tray_center[1] + random.uniform(-offset, offset),
                tray_center[2] + random.uniform(-offset, offset)
            )
        
        # 20% chance: Sample at preferred height
        elif strategy < 0.9 and preferred_height:
            min_x, min_y, _, max_x, max_y, _ = bounds
            return (
                random.uniform(min_x, max_x),
                random.uniform(min_y, max_y),
                preferred_height + random.uniform(-0.5, 0.5)  # ±0.5m variation
            )
        
        # 10% chance (or fallback): Completely random
        else:
            return self._sample_random_point(bounds)
    
    def _is_path_clear(
        self,
        start: Tuple[float, float, float],
        end: Tuple[float, float, float],
        obstacles: List[Tuple[float, float, float, float, float, float]],
        clearance: float
    ) -> bool:
        """Check if straight path is clear of obstacles"""
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
    
    def _bboxes_intersect(
        self,
        bbox1: Tuple[float, float, float, float, float, float],
        bbox2: Tuple[float, float, float, float, float, float]
    ) -> bool:
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
    
    def _calculate_search_bounds(
        self,
        start: Tuple[float, float, float],
        end: Tuple[float, float, float],
        buffer: float
    ) -> Tuple[float, float, float, float, float, float]:
        """Calculate bounded search area"""
        return (
            min(start[0], end[0]) - buffer,
            min(start[1], end[1]) - buffer,
            min(start[2], end[2]) - buffer,
            max(start[0], end[0]) + buffer,
            max(start[1], end[1]) + buffer,
            max(start[2], end[2]) + buffer
        )
    
    def _sample_random_point(
        self,
        bounds: Tuple[float, float, float, float, float, float]
    ) -> Tuple[float, float, float]:
        """Sample random point within bounds"""
        min_x, min_y, min_z, max_x, max_y, max_z = bounds
        return (
            random.uniform(min_x, max_x),
            random.uniform(min_y, max_y),
            random.uniform(min_z, max_z)
        )
    
    def _find_nearest_node(
        self,
        tree: Dict[int, Dict[str, Any]],
        point: Tuple[float, float, float]
    ) -> int:
        """Find nearest node in tree to given point"""
        nearest_idx = 0
        nearest_dist = float('inf')
        
        for idx, node in tree.items():
            dist = self._distance(node['point'], point)
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_idx = idx
        
        return nearest_idx
    
    def _distance(
        self,
        p1: Tuple[float, float, float],
        p2: Tuple[float, float, float]
    ) -> float:
        """Euclidean distance between two points"""
        return math.sqrt(sum((a - b)**2 for a, b in zip(p1, p2)))
    
    def _steer(
        self,
        from_point: Tuple[float, float, float],
        to_point: Tuple[float, float, float],
        max_step: float
    ) -> Tuple[float, float, float]:
        """Steer from one point toward another"""
        dx = to_point[0] - from_point[0]
        dy = to_point[1] - from_point[1]
        dz = to_point[2] - from_point[2]
        
        distance = math.sqrt(dx**2 + dy**2 + dz**2)
        
        if distance <= max_step:
            return to_point
        
        ratio = max_step / distance
        return (
            from_point[0] + dx * ratio,
            from_point[1] + dy * ratio,
            from_point[2] + dz * ratio
        )
    
    def _apply_orthogonal_bias(
        self,
        from_point: Tuple[float, float, float],
        to_point: Tuple[float, float, float]
    ) -> Tuple[float, float, float]:
        """Bias movement toward orthogonal directions"""
        # Allow diagonal movement sometimes
        if random.random() > self.ORTHOGONAL_BIAS:
            return to_point
        
        # Snap to dominant axis
        dx = abs(to_point[0] - from_point[0])
        dy = abs(to_point[1] - from_point[1])
        dz = abs(to_point[2] - from_point[2])
        
        if dx >= dy and dx >= dz:
            return (to_point[0], from_point[1], from_point[2])
        elif dy >= dx and dy >= dz:
            return (from_point[0], to_point[1], from_point[2])
        else:
            return (from_point[0], from_point[1], to_point[2])
    
    def _extract_path(
        self,
        tree: Dict[int, Dict[str, Any]],
        goal_idx: int,
        goal_point: Tuple[float, float, float]
    ) -> List[Tuple[float, float, float]]:
        """Extract path by backtracking from goal"""
        path = [goal_point]
        current_idx = goal_idx
        
        while tree[current_idx]['parent'] is not None:
            current_idx = tree[current_idx]['parent']
            path.append(tree[current_idx]['point'])
        
        return list(reversed(path))
    
    def _simplify_path(
        self,
        path: List[Tuple[float, float, float]],
        obstacles: List[Tuple[float, float, float, float, float, float]],
        clearance: float
    ) -> List[Tuple[float, float, float]]:
        """Remove redundant waypoints using line-of-sight"""
        if len(path) <= 2:
            return path
        
        simplified = [path[0]]
        i = 0
        
        while i < len(path) - 1:
            # Try to skip ahead as far as possible
            j = len(path) - 1
            while j > i + 1:
                if self._is_path_clear(path[i], path[j], obstacles, clearance):
                    simplified.append(path[j])
                    i = j
                    break
                j -= 1
            else:
                # Couldn't skip, take next waypoint
                simplified.append(path[i + 1])
                i += 1
        
        return simplified


class IFCGeometryGenerator:
    """
    Generate IFC geometry for routed conduits
    Extracted from RouteMEPConduit operator
    """
    
    def generate_conduit(
        self,
        ifc_file,
        waypoints: List[Tuple[float, float, float]],
        diameter: float
    ) -> bool:
        """
        Generate IFC conduit segments and fittings
        
        Args:
            ifc_file: Active IFC file handle
            waypoints: Route waypoints in meters
            diameter: Conduit diameter in mm
        
        Returns:
            True if successful, False otherwise
        """
        import ifcopenshell.api
        from mathutils import Vector
        
        print(f"\n📦 Generating IFC geometry:")
        print(f"   Waypoints: {len(waypoints)}")
        print(f"   Diameter: {diameter}mm")
        
        # Detect IFC schema
        schema = ifc_file.schema
        print(f"   Schema: {schema}")
        
        # Create or get electrical system
        electrical_system = self._get_or_create_system(ifc_file)
        
        created_elements = []
        
        # Create segments between waypoints
        for i in range(len(waypoints) - 1):
            start_pt = Vector(waypoints[i])
            end_pt = Vector(waypoints[i + 1])
            
            # Calculate segment properties
            direction = end_pt - start_pt
            length = direction.length
            
            if length < 0.01:  # Skip very short segments
                continue
            
            segment = self._create_segment(
                ifc_file, schema, start_pt, end_pt, length,
                diameter, i+1, electrical_system
            )
            
            if segment:
                created_elements.append(segment)
                print(f"  Segment: {start_pt} → {end_pt} (length: {length:.2f}m)")
        
        # Create fittings at waypoints (except start/end)
        if len(waypoints) > 2:
            for i in range(1, len(waypoints) - 1):
                fitting = self._create_fitting(
                    ifc_file, schema, waypoints[i],
                    diameter, i, electrical_system
                )
                if fitting:
                    created_elements.append(fitting)
                    print(f"  ✓ Created elbow {i} at waypoint {waypoints[i]}")
        
        success = len(created_elements) > 0
        if success:
            print(f"\n✓ Generated {len(created_elements)} IFC elements successfully")
        else:
            print(f"\n✗ No IFC elements created")
        
        return success
    
    def _get_or_create_system(self, ifc_file):
        """Get or create electrical distribution system"""
        import ifcopenshell.api
        
        # Check if system exists
        for system in ifc_file.by_type("IfcSystem"):
            if system.Name == "Electrical Distribution":
                print(f"   ✓ Using existing system: {system.Name}")
                return system
        
        # Create new system
        system = ifcopenshell.api.run(
            "system.add_system",
            ifc_file,
            ifc_class="IfcDistributionSystem"
        )
        system.Name = "Electrical Distribution"
        system.Description = "Auto-generated electrical conduit system"
        system.ObjectType = "ELECTRICAL"
        
        print(f"   ✓ Created new system: {system.Name}")
        return system
    
    def _create_segment(self, ifc_file, schema, start_pt, end_pt, length, 
                       diameter, index, system):
        """Create cable carrier segment between two points"""
        import ifcopenshell.api
        
        try:
            # Create segment (schema-aware)
            if schema == "IFC2X3":
                # IFC2X3: Use IfcFlowSegment
                segment = ifcopenshell.api.run(
                    "root.create_entity",
                    ifc_file,
                    ifc_class="IfcFlowSegment"
                )
                segment.Name = f"Cable Tray Segment {index}"
                segment.ObjectType = "CABLETRAY"
            else:
                # IFC4+: Use IfcCableCarrierSegment
                segment = ifcopenshell.api.run(
                    "root.create_entity",
                    ifc_file,
                    ifc_class="IfcCableCarrierSegment"
                )
                segment.Name = f"Cable Tray Segment {index}"
                segment.PredefinedType = "CABLETRAYSEGMENT"
            
            # Validate entity created
            if not segment or not hasattr(segment, 'GlobalId'):
                print(f"  ✗ Failed to create segment {index}")
                return None
            
            print(f"  ✓ Created segment {index}: {segment.is_a()} (GlobalId: {segment.GlobalId})")
            
            # Assign to electrical system
            try:
                ifcopenshell.api.run(
                    "system.assign_system",
                    ifc_file,
                    products=[segment],
                    system=system
                )
            except Exception as e:
                print(f"  ⚠ Warning: Could not assign segment to system: {e}")
            
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
                print(f"  ⚠ Warning: Could not add properties: {e}")
            
            return segment
            
        except Exception as e:
            print(f"  ✗ Error creating segment {index}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _create_fitting(self, ifc_file, schema, location, diameter, index, system):
        """Create cable carrier fitting (elbow) at waypoint"""
        import ifcopenshell.api
        
        try:
            if schema == "IFC2X3":
                fitting = ifcopenshell.api.run(
                    "root.create_entity",
                    ifc_file,
                    ifc_class="IfcFlowFitting"
                )
                fitting.Name = f"Cable Tray Elbow {index}"
                fitting.ObjectType = "CABLETRAY_BEND"
            else:
                fitting = ifcopenshell.api.run(
                    "root.create_entity",
                    ifc_file,
                    ifc_class="IfcCableCarrierFitting"
                )
                fitting.Name = f"Cable Tray Elbow {index}"
                fitting.PredefinedType = "BEND"
            
            # Validate entity
            if not fitting or not hasattr(fitting, 'GlobalId'):
                return None
            
            # Assign to electrical system
            try:
                ifcopenshell.api.run(
                    "system.assign_system",
                    ifc_file,
                    products=[fitting],
                    system=system
                )
            except Exception as e:
                print(f"  ⚠ Warning: Could not assign fitting to system: {e}")
            
            return fitting
            
        except Exception as e:
            print(f"  ✗ Error creating fitting {index}: {e}")
            return None