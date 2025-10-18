# ============================================================================
# FILE: src/bonsai/bonsai/bim/module/mep_engineering/visualization.py
# PURPOSE: Visual debugging tools for MEP conduit routing
# ============================================================================
# 
# This module creates temporary Blender objects to visualize:
# - Start/end points (colored spheres)
# - Obstacle bounding boxes (semi-transparent cubes)
# - Clearance zones (wireframe boxes)
# 
# All objects are prefixed with "MEP Debug" for easy cleanup.
# Does not modify Bonsai core - uses standard Blender API only.
# ============================================================================

import bpy
import bmesh
from mathutils import Vector
from typing import List, Tuple, Optional


def create_debug_sphere(
    location: Tuple[float, float, float],
    radius: float,
    color: Tuple[float, float, float, float],
    name: str
) -> bpy.types.Object:
    """
    Create a colored sphere at specified location
    
    Args:
        location: (x, y, z) position in meters
        radius: Sphere radius in meters
        color: (r, g, b, a) color with alpha (0-1 range)
        name: Object name (will be prefixed with "MEP Debug ")
        
    Returns:
        Created Blender object
    """
    # Create sphere mesh
    mesh = bpy.data.meshes.new(f"MEP Debug {name}")
    obj = bpy.data.objects.new(f"MEP Debug {name}", mesh)
    
    # Link to scene
    bpy.context.scene.collection.objects.link(obj)
    
    # Generate sphere geometry using bmesh
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=16, v_segments=8, radius=radius)
    bm.to_mesh(mesh)
    bm.free()
    
    # Set location
    obj.location = location
    
    # Create and assign material
    mat = bpy.data.materials.new(name=f"MEP Debug Material {name}")
    mat.diffuse_color = color
    mat.use_nodes = False  # Simple material
    
    if color[3] < 1.0:  # Has transparency
        mat.blend_method = 'OPAQUE'
        mat.show_transparent_back = False
    
    obj.data.materials.append(mat)
    
    return obj


def create_obstacle_box(
    bbox: Tuple[float, float, float, float, float, float],
    color: Tuple[float, float, float, float],
    name: str,
    wireframe: bool = False
) -> bpy.types.Object:
    """
    Create a box representing an obstacle's bounding box
    
    Args:
        bbox: (min_x, min_y, min_z, max_x, max_y, max_z) in meters
        color: (r, g, b, a) color with alpha
        name: Object name (will be prefixed with "MEP Debug ")
        wireframe: If True, display as wireframe only
        
    Returns:
        Created Blender object
    """
    min_x, min_y, min_z, max_x, max_y, max_z = bbox
    
    # Calculate dimensions and center
    width = max_x - min_x
    depth = max_y - min_y
    height = max_z - min_z
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    center_z = (min_z + max_z) / 2
    
    # Create cube mesh
    mesh = bpy.data.meshes.new(f"MEP Debug {name}")
    obj = bpy.data.objects.new(f"MEP Debug {name}", mesh)
    
    # Link to scene
    bpy.context.scene.collection.objects.link(obj)
    
    # Generate cube geometry
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bm.to_mesh(mesh)
    bm.free()
    
    # Position and scale
    obj.location = (center_x, center_y, center_z)
    obj.scale = (width / 2, depth / 2, height / 2)
    
    # Create and assign material
    # Create and assign material with emission
    mat = bpy.data.materials.new(name=f"MEP Debug Material {name}")
    mat.use_nodes = True  # Enable nodes for emission
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Clear default nodes
    nodes.clear()

    # Add shader nodes
    output = nodes.new('ShaderNodeOutputMaterial')
    emission = nodes.new('ShaderNodeEmission')

    # Set emission color and strength
    emission.inputs['Color'].default_value = color
    emission.inputs['Strength'].default_value = 5.0  # Bright glow

    # Connect emission to output
    links.new(emission.outputs['Emission'], output.inputs['Surface'])
    
    if color[3] < 1.0 or wireframe:  # Transparent or wireframe
        mat.blend_method = 'OPAQUE'
        mat.show_transparent_back = False
    
    obj.data.materials.append(mat)
    
    # Set display mode
    if wireframe:
        obj.display_type = 'WIRE'
    
    return obj


def create_clearance_zone(
    bbox: Tuple[float, float, float, float, float, float],
    clearance: float,
    name: str
) -> bpy.types.Object:
    """
    Create a wireframe box showing clearance zone around obstacle
    
    Args:
        bbox: Original obstacle bbox (min_x, min_y, min_z, max_x, max_y, max_z)
        clearance: Buffer distance in meters
        name: Object name (will be prefixed with "MEP Debug Clearance ")
        
    Returns:
        Created Blender object
    """
    # Expand bbox by clearance
    expanded_bbox = (
        bbox[0] - clearance,  # min_x
        bbox[1] - clearance,  # min_y
        bbox[2] - clearance,  # min_z
        bbox[3] + clearance,  # max_x
        bbox[4] + clearance,  # max_y
        bbox[5] + clearance   # max_z
    )
    
    # Yellow wireframe for clearance zones
    yellow_color = (1.0, 1.0, 0.0, 0.3)
    
    return create_obstacle_box(
        expanded_bbox,
        yellow_color,
        f"Clearance {name}",
        wireframe=True
    )


def create_corridor_visualization(
    start: Tuple[float, float, float],
    end: Tuple[float, float, float],
    buffer: float
) -> bpy.types.Object:
    """
    Create a box showing the routing corridor search space
    
    Args:
        start: Start point (x, y, z)
        end: End point (x, y, z)
        buffer: Corridor buffer distance in meters
        
    Returns:
        Created Blender object
    """
    # Calculate corridor bounding box
    min_x = min(start[0], end[0]) - buffer
    max_x = max(start[0], end[0]) + buffer
    min_y = min(start[1], end[1]) - buffer
    max_y = max(start[1], end[1]) + buffer
    min_z = min(start[2], end[2]) - buffer
    max_z = max(start[2], end[2]) + buffer
    
    corridor_bbox = (min_x, min_y, min_z, max_x, max_y, max_z)
    
    # Cyan wireframe for corridor
    cyan_color = (0.0, 1.0, 1.0, 0.2)
    
    return create_obstacle_box(
        corridor_bbox,
        cyan_color,
        "Corridor",
        wireframe=True
    )


def create_path_line(
    waypoints: List[Tuple[float, float, float]],
    name: str = "Path"
) -> bpy.types.Object:
    """
    Create a line/curve showing the routing path
    
    Args:
        waypoints: List of (x, y, z) points forming the path
        name: Object name (will be prefixed with "MEP Debug ")
        
    Returns:
        Created Blender curve object
    """
    if len(waypoints) < 2:
        return None
    
    # Create curve data
    curve_data = bpy.data.curves.new(f"MEP Debug {name}", type='CURVE')
    curve_data.dimensions = '3D'
    curve_data.resolution_u = 2
    curve_data.bevel_depth = 0.05  # Line thickness
    
    # Create spline
    polyline = curve_data.splines.new('POLY')
    polyline.points.add(len(waypoints) - 1)
    
    for i, point in enumerate(waypoints):
        x, y, z = point
        polyline.points[i].co = (x, y, z, 1)
    
    # Create object
    obj = bpy.data.objects.new(f"MEP Debug {name}", curve_data)
    bpy.context.scene.collection.objects.link(obj)
    
    # Green material for path
    mat = bpy.data.materials.new(name=f"MEP Debug Material {name}")
    mat.diffuse_color = (0.0, 1.0, 0.0, 1.0)  # Bright green
    mat.use_nodes = False
    obj.data.materials.append(mat)
    
    return obj


def clear_debug_objects():
    """
    Remove all MEP debug visualization objects from the scene
    """
    objects_to_remove = []
    
    # Find all objects with "MEP Debug" prefix
    for obj in bpy.data.objects:
        if obj.name.startswith("MEP Debug"):
            objects_to_remove.append(obj)
    
    # Remove objects
    for obj in objects_to_remove:
        bpy.data.objects.remove(obj, do_unlink=True)
    
    # Clean up orphaned data
    for mesh in bpy.data.meshes:
        if mesh.name.startswith("MEP Debug") and mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    
    for material in bpy.data.materials:
        if material.name.startswith("MEP Debug") and material.users == 0:
            bpy.data.materials.remove(material)
    
    for curve in bpy.data.curves:
        if curve.name.startswith("MEP Debug") and curve.users == 0:
            bpy.data.curves.remove(curve)
    
    print(f"✓ Cleared {len(objects_to_remove)} debug objects")


def visualize_routing_scenario(
    start: Tuple[float, float, float],
    end: Tuple[float, float, float],
    obstacles: List[Tuple[float, float, float, float, float, float]],
    clearance: float,
    waypoints: Optional[List[Tuple[float, float, float]]] = None,
    show_clearance_zones: bool = True,
    show_corridor: bool = True
) -> dict:
    """
    Complete visualization of a routing scenario
    
    Args:
        start: Start point (x, y, z) in IFC coordinates
        end: End point (x, y, z) in IFC coordinates
        obstacles: List of obstacle bboxes (min_x, min_y, min_z, max_x, max_y, max_z)
        clearance: Clearance distance in meters
        waypoints: Optional list of route waypoints
        show_clearance_zones: Show clearance zones around obstacles
        show_corridor: Show routing corridor
        
    Returns:
        Dictionary of created Blender objects
    """
    # IFC coordinates match Blender coordinates - no offset needed
    # Calculate coordinate offset from first building object
    import bpy
    offset_x = 0
    offset_y = 0
    offset_z = 0
    
    # Find any IFC building object to determine offset
    for obj in bpy.data.objects:
        if obj.type == 'MESH' and 'Ifc' in obj.name:
            # Building is around (120, -20, 10), IFC coords around (-50387, 34273, 6)
            # Estimate offset from object location vs expected IFC coords
            # Use your start point as reference
            offset_x = start[0] - 120  # Adjust to building X center
            offset_y = start[1] + 20   # Adjust to building Y center  
            offset_z = start[2] - 6    # Adjust to building Z
            print(f"📍 Detected offset from building: ({offset_x:.1f}, {offset_y:.1f}, {offset_z:.1f})")
            break
    
    if offset_x == 0:
        print(f"⚠️  No building objects found - using raw IFC coordinates")
    
    # Apply offset to all coordinates (currently zero)
    start = (start[0] - offset_x, start[1] - offset_y, start[2] - offset_z)
    end = (end[0] - offset_x, end[1] - offset_y, end[2] - offset_z)
    
    # Apply offset to obstacles
    offset_obstacles = []
    for bbox in obstacles:
        offset_bbox = (
            bbox[0] - offset_x, bbox[1] - offset_y, bbox[2] - offset_z,
            bbox[3] - offset_x, bbox[4] - offset_y, bbox[5] - offset_z
        )
        offset_obstacles.append(offset_bbox)
    
    obstacles = offset_obstacles
    
    # Rest of function continues as before...
    created_objects = {}
    
    # Start point (green sphere)
    created_objects['start'] = create_debug_sphere(
        start,
        radius=0.5,
        color=(0.0, 1.0, 0.0, 1.0),  # Green
        name="Start Point"
    )
    
    # End point (red sphere)
    created_objects['end'] = create_debug_sphere(
        end,
        radius=0.5,
        color=(1.0, 0.0, 0.0, 1.0),  # Red
        name="End Point"
    )
    
    # Corridor (optional)
    if show_corridor:
        buffer = clearance * 3  # Corridor is wider than clearance
        created_objects['corridor'] = create_corridor_visualization(
            start, end, buffer
        )
    
    # Obstacles and clearance zones
    created_objects['obstacles'] = []
    created_objects['clearances'] = []
    
    for i, bbox in enumerate(obstacles):
        # Obstacle box (semi-transparent red)
        obs_obj = create_obstacle_box(
            bbox,
            color=(1.0, 0.0, 0.0, 0.3),  # Red, 30% opacity
            name=f"Obstacle {i+1}",
            wireframe=True
        )
        created_objects['obstacles'].append(obs_obj)
        
        # Clearance zone (optional, yellow wireframe)
        if show_clearance_zones:
            clear_obj = create_clearance_zone(
                bbox,
                clearance,
                name=f"{i+1}"
            )
            created_objects['clearances'].append(clear_obj)
    
    # Create path line if waypoints provided
    if waypoints and len(waypoints) >= 2:
        # Apply same coordinate offset to waypoints
        offset_waypoints = [
            (wp[0] - offset_x, wp[1] - offset_y, wp[2] - offset_z)
            for wp in waypoints
        ]
        created_objects['path'] = create_path_line(
            offset_waypoints,
            name="Conduit Route"
        )
    
    print(f"✓ Visualization created:")
    print(f"  - Start/End points: 2 spheres")
    if waypoints:
        print(f"  - Route path: {len(waypoints)} waypoints (green curve)")
    print(f"  - Obstacles: {len(obstacles)} boxes")
    if show_clearance_zones:
        print(f"  - Clearance zones: {len(obstacles)} wireframes")
    if show_corridor:
        print(f"  - Corridor: 1 wireframe box")
    
    return created_objects

def navigate_to_view(
    target_location: Tuple[float, float, float],
    view_distance: float = 15.0
) -> bool:
    """
    Smoothly animate viewport to focus on target location with X-ray mode
    Uses Blender's native view_selected for smooth zoom animation
    
    Args:
        target_location: (x, y, z) point to center on in Blender coordinates
        view_distance: Distance to zoom (unused - Blender calculates automatically)
        
    Returns:
        True if viewport adjusted successfully, False otherwise
    """
    import bpy
    
    success = False
    
    # Find 3D viewport and adjust view
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    # Save current X-ray state before enabling
                    if not space.shading.show_xray:
                        # Store original alpha (or flag that X-ray was off)
                        bpy.context.scene["MEP_saved_xray_alpha"] = -1.0  # -1 means "was off"
                    else:
                        # Store current alpha value
                        bpy.context.scene["MEP_saved_xray_alpha"] = space.shading.xray_alpha
                    
                    # Enable X-ray mode for see-through
                    space.shading.show_xray = True
                    space.shading.xray_alpha = 0.3  # 30% building opacity
                    
                    print(f"✓ X-ray mode: ON (30% opacity)")
                    print(f"  Saved previous state: {bpy.context.scene['MEP_saved_xray_alpha']}")
                    
                    success = True
                    break
            break
    
    # Now trigger smooth zoom using Blender's native frame selected
    # This requires that debug objects are already selected
    # Now trigger smooth zoom using Blender's native frame selected
    # Need to override context to provide proper 3D view region
    if success:
        try:
            # Find the area and region we just modified
            for area in bpy.context.screen.areas:
                if area.type == 'VIEW_3D':
                    for region in area.regions:
                        if region.type == 'WINDOW':
                            override = {'area': area, 'region': region}
                            with bpy.context.temp_override(**override):
                                bpy.ops.view3d.view_selected(use_all_regions=False)
                            print(f"✓ Smooth zoom animation to conduit location")
                            break
                    break
        except Exception as e:
            print(f"⚠  Could not animate zoom: {e}")
    
    if not success:
        print("⚠  Could not find 3D viewport to adjust")
    
    return success

# ============================================================================
# USAGE EXAMPLE (for testing)
# ============================================================================

if __name__ == "__main__":
    # Example: Visualize a simple routing scenario
    
    # Clear previous debug objects
    clear_debug_objects()
    
    # Define scenario
    start = (-50428.0, 34202.0, 6.0)
    end = (-50434.0, 34210.0, 6.0)
    
    # Sample obstacles (from your Terminal 1 data)
    obstacles = [
        (-50428.2, 34201.8, 5.95, -50427.8, 34202.2, 6.05),  # Near start
        (-50430.0, 34205.0, 5.9, -50429.6, 34205.4, 6.1),    # Mid corridor
        (-50433.8, 34209.8, 5.95, -50433.4, 34210.2, 6.05),  # Near end
    ]
    
    clearance = 0.5
    
    # Create visualization
    objects = visualize_routing_scenario(
        start, end, obstacles, clearance,
        show_clearance_zones=True,
        show_corridor=True
    )
    
    # Navigate camera to view
    midpoint = (
        (start[0] + end[0]) / 2,
        (start[1] + end[1]) / 2,
        (start[2] + end[2]) / 2
    )
    navigate_to_view(midpoint, distance=20.0)
    
    print("\n✓ Visualization complete!")
    print("Run clear_debug_objects() to remove all debug objects")