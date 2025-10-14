# ============================================================================
# FILE: ui.py
# PURPOSE: Define UI panels that appear in Blender interface
# ============================================================================

import bpy
from bpy.types import Panel

class BIM_PT_mep_engineering(Panel):
    """MEP Engineering panel in Blender UI"""
    bl_label = "MEP Engineering"
    bl_idname = "BIM_PT_mep_engineering"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "scene"
    bl_parent_id = "BIM_PT_tab_quality_control"  # ← Check this line

    # bl_parent_id = "BIM_PT_tab_geometric_relationships"  # Commented out for testing
    
    def draw(self, context):
        """Draw the panel UI"""
        layout = self.layout
        props = context.scene.BIMmepEngineeringProperties
        
        # Add a header
        layout.label(text="MEP Analysis Tools - DEVELOPMENT MODE", icon='MODIFIER')
        
        # Project settings section
        box = layout.box()
        box.label(text="Project Settings")
        box.prop(props, "project_number")
        
        # Calculation settings section
        box = layout.box()
        box.label(text="Calculation Settings")
        box.prop(props, "enable_clash_detection")
        box.prop(props, "min_clearance_mm")
        
        # Test button
        layout.separator()
        layout.operator("bim.test_mep_operator", icon='PLAY')
        
        # ================================================================
        # ROUTING SECTION (Phase 1)
        # ================================================================
        layout.separator()
        layout.separator()
        
        box = layout.box()
        box.label(text="Conduit Routing", icon='CURVE_PATH')
        
        # Start Point
        row = box.row(align=True)
        row.label(text="Start Point:")
        row.operator("bim.set_route_start_point", text="Set from Cursor", icon='CURSOR')
        
        row = box.row()
        row.prop(props, "route_start_point", text="")
        
        # End Point
        row = box.row(align=True)
        row.label(text="End Point:")
        row.operator("bim.set_route_end_point", text="Set from Cursor", icon='CURSOR')
        
        row = box.row()
        row.prop(props, "route_end_point", text="")
        
        # Settings
        box.separator()
        row = box.row()
        row.prop(props, "clearance_distance")
        
        row = box.row()
        row.prop(props, "conduit_diameter")
        
        row = box.row()
        row.prop(props, "target_disciplines")
        
        # Route Button
        box.separator()
        row = box.row()
        row.scale_y = 1.5
        
        # Check if federation loaded to enable/disable button
        fed_props = context.scene.BIMFederationProperties
        row.enabled = fed_props.index_loaded
        row.operator("bim.route_mep_conduit", text="Route Conduit", icon='ANIM')
        
        if not fed_props.index_loaded:
            box.label(text="⚠ Load Federation Index first", icon='ERROR')

        # ================================================================
        # VALIDATION SECTION (Phase 2B)
        # ================================================================
        layout.separator()
        
        box = layout.box()
        box.label(text="Clash Validation", icon='CHECKMARK')
        
        row = box.row()
        row.prop(props, "export_bcf")
        
        row = box.row()
        row.prop(props, "show_clash_details")
        
        # Validate button
        row = box.row()
        row.scale_y = 1.5
        row.enabled = fed_props.index_loaded
        row.operator("bim.validate_conduit_route", text="Validate Route", icon='CHECKMARK')
        
        if not fed_props.index_loaded:
            box.label(text="⚠ Load Federation Index first", icon='ERROR')