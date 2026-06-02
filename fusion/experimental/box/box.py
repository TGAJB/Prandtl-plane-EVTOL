"""
Fusion 360 – Change User Parameters Script
============================================
HOW TO USE:
  1. Open your Fusion 360 design.
  2. Go to Tools > Add-Ins > Scripts and Add-Ins (Shift+S).
  3. Click the green "+" next to "My Scripts", browse to this file.
  4. Select the script and click "Run".

Edit the PARAMETERS dict below to match your design's parameter names and
the new values you want to apply. Expressions (strings) are also supported.
"""

import adsk.core
import adsk.fusion
import traceback

def run(context):
    ui = None
    try:
        app = adsk.core.Application.get()
        ui = app.userInterface

        doc = app.activeDocument

        design = adsk.fusion.Design.cast(doc.products.itemByProductType("DesignProductType"))
        if not design:
            ui.messageBox("No active Fusion design", "Error")
            return

        userparams = design.userParameters

        userparams.itemByName("length").expression = f"{1000} mm"

    except:
        raise ValueError