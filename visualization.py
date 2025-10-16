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
        mat.blend_method = 'BLEND'
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
    mat = bpy.data.materials.new(name=f"MEP Debug Material {name}")
    mat.diffuse_color = color
    mat.use_nodes = False
    
    if color[3] < 1.0 or wireframe:  # Transparent or wireframe
        mat.blend_method = 'BLEND'
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
    show_clearance_zones: bool = True,
    show_corridor: bool = True
) -> dict:
    """
    Complete visualization of a routing scenario
    """
    # Get Blender coordinate offset from georeferencing
    import bpy
    geo_props = bpy.context.scene.BIMGeoreferenceProperties
    
    # Initialize offsets
    offset_x = 0.0
    offset_y = 0.0
    offset_z = 0.0
    
    # Apply offset if georeferencing is active
    if geo_props.has_blender_offset:
        try:
            # CRITICAL FIX: Offset properties are in MILLIMETERS, convert to METERS
            offset_x = float(geo_props.blender_offset_x) / 1000.0
            offset_y = float(geo_props.blender_offset_y) / 1000.0
            offset_z = float(geo_props.blender_offset_z) / 1000.0
            
            print(f"📍 Applying Blender offset: ({offset_x:.2f}, {offset_y:.2f}, {offset_z:.2f})m")
            print(f"   (From: {geo_props.blender_offset_x:.1f}mm, {geo_props.blender_offset_y:.1f}mm, {geo_props.blender_offset_z:.1f}mm)")
        except (ValueError, AttributeError) as e:
            print(f"⚠️ Could not read georeference offset: {e}")
            # Continue with zero offset
    else:
        print("ℹ️ No Blender offset detected, using coordinates as-is")
    
    # Apply offset to all coordinates
    start = (start[0] - offset_x, start[1] - offset_y, start[2] - offset_z)
    end = (end[0] - offset_x, end[1] - offset_y, end[2] - offset_z)
    
    print(f"   Start after offset: ({start[0]:.2f}, {start[1]:.2f}, {start[2]:.2f})m")
    print(f"   End after offset: ({end[0]:.2f}, {end[1]:.2f}, {end[2]:.2f})m")
    
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
        radius=0.2,
        color=(0.0, 1.0, 0.0, 1.0),  # Green
        name="Start Point"
    )
    
    # End point (red sphere)
    created_objects['end'] = create_debug_sphere(
        end,
        radius=0.2,
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
            name=f"Obstacle {i+1}"
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
    
    print(f"✓ Visualization created:")
    print(f"  - Start/End points: 2 spheres")
    print(f"  - Obstacles: {len(obstacles)} boxes")
    if show_clearance_zones:
        print(f"  - Clearance zones: {len(obstacles)} wireframes")
    if show_corridor:
        print(f"  - Corridor: 1 wireframe box")
    
    return created_objects

def navigate_to_view(
    target_location: Tuple[float, float, float],
    distance: float = 15.0
) -> Optional[bpy.types.Object]:
    """
    Position viewport camera to look at target location
    
    Args:
        target_location: (x, y, z) point to look at
        distance: Camera distance from target in meters
        
    Returns:
        Camera object or None if viewport not found
    """
    # Get or create camera
    camera = bpy.data.objects.get("MEP Debug Camera")
    if not camera:
        camera_data = bpy.data.cameras.new("MEP Debug Camera")
        camera = bpy.data.objects.new("MEP Debug Camera", camera_data)
        bpy.context.scene.collection.objects.link(camera)
    
    # Position camera (offset above and to the side for good view)
    camera.location = (
        target_location[0] + distance * 0.5,
        target_location[1] - distance * 0.7,
        target_location[2] + distance * 0.5
    )
    
    # Point at target
    direction = Vector(target_location) - Vector(camera.location)
    camera.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
    
    # Set as active camera
    bpy.context.scene.camera = camera
    
    # Try to switch viewport to camera view
    try:
        for area in bpy.context.screen.areas:
            if area.type == 'VIEW_3D':
                for space in area.spaces:
                    if space.type == 'VIEW_3D':
                        space.region_3d.view_perspective = 'CAMERA'
                        space.shading.show_xray = True  # See through walls
                        break
                break
    except Exception as e:
        print(f"⚠ Could not switch to camera view: {e}")
    
    return camera


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