# Bonsai - OpenBIM Blender Add-on
# Copyright (C) 2025 Your Engineering Firm
#
# This file is part of Bonsai.
#
# Bonsai is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import bpy
from . import ui, prop, operator

# Expose classes so main __init__.py can find them
classes = (
    prop.BIMmepEngineeringProperties,
    operator.TestMEPOperator,
    operator.RouteMEPConduit,
    operator.SetRouteStartPoint,
    operator.SetRouteEndPoint,
    ui.BIM_PT_mep_engineering,
)

def register():
    """Called when addon is enabled"""
    # Attach properties to Blender's Scene
    bpy.types.Scene.BIMmepEngineeringProperties = bpy.props.PointerProperty(
        type=prop.BIMmepEngineeringProperties
    )

def unregister():
    """Called when addon is disabled - cleanup"""
    # Remove properties from Scene
    del bpy.types.Scene.BIMmepEngineeringProperties