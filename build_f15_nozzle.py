import math
import os
import sys
import bpy
import bmesh
from mathutils import Vector, Matrix, Euler

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from f100_geom_primitives import NozzleSpec, build_clay_materials, rgba
from f100_engine_module import (
    build_forward_turbofan_core_and_fan,
    build_static_base_and_interior,
    build_armature_and_bind_parts,
)
from f100_stand_module import (
    build_maintenance_stand_and_trailer,
    refresh_stand_only_in_current_scene,
)
from f100_materials_module import apply_procedural_pbr_to_current_scene


UI_PANEL_SCRIPT_SOURCE = '''import bpy

def _update_nozzle_props(self, context):
    if context and context.view_layer:
        context.view_layer.update()

def _srgb_to_lin_ui(c):
    return (c / 12.92) if c <= 0.04045 else (((c + 0.055) / 1.055) ** 2.4)

def _hex_rgba_ui(h):
    return (
        _srgb_to_lin_ui(((h >> 16) & 0xFF) / 255.0),
        _srgb_to_lin_ui(((h >> 8) & 0xFF) / 255.0),
        _srgb_to_lin_ui((h & 0xFF) / 255.0),
        1.0,
    )

def _update_color_id_mode(self, context):
    use_cid = getattr(self, "f15_color_id_mode", True)
    fac = 1.0 if use_cid else 0.0
    for mat in bpy.data.materials:
        if "f15_id_hex" in mat:
            if mat.node_tree:
                mx = mat.node_tree.nodes.get("ColorID_Mode_Mix")
                if mx is not None:
                    mx.inputs[0].default_value = fac
            mat.diffuse_color = _hex_rgba_ui(int(mat["f15_id_hex"])) if use_cid else (0.86, 0.87, 0.89, 1.0)
    if context and context.screen:
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                for sp in area.spaces:
                    if sp.type == 'VIEW_3D':
                        sp.shading.color_type = 'MATERIAL' if use_cid else 'OBJECT'
    if context and context.view_layer:
        context.view_layer.update()

def _update_shroud_vis(self, context):
    show = getattr(self, "f15_show_shroud", True)
    sh_obj = bpy.data.objects.get("Static_Nozzle_Outer_Shroud")
    if sh_obj is not None:
        sh_obj.hide_viewport = not show
        sh_obj.hide_render = not show
    show_st = getattr(self, "f15_show_stand", True)
    show_tr = getattr(self, "f15_show_trailer", True)
    for o in bpy.data.objects:
        if o.name.startswith("Stand_Cradle_"):
            o.hide_viewport = not show_st
            o.hide_render = not show_st
        elif o.name.startswith("Stand_Trailer_"):
            vis_t = show_st and show_tr
            o.hide_viewport = not vis_t
            o.hide_render = not vis_t
    if context and context.view_layer:
        context.view_layer.update()

class F15_OT_SetNozzlePreset(bpy.types.Operator):
    bl_idname = "f15_nozzle.set_preset"
    bl_label = "Set Nozzle Aperture Preset"
    bl_description = "Apply a predefined nozzle throat and exit area configuration"
    bl_options = {'REGISTER', 'UNDO'}

    target_open: bpy.props.FloatProperty(default=0.0, min=0.0, max=1.0)
    target_slider: bpy.props.FloatProperty(default=0.0, min=-1.0, max=1.0)

    def execute(self, context):
        scn = context.scene
        if getattr(scn, "f15_auto_demo", False):
            scn.f15_auto_demo = False
        scn.f15_nozzle_open = self.target_open
        scn.f15_slider_mode = self.target_slider
        return {'FINISHED'}

class F15_OT_ToggleGroundTestDemo(bpy.types.Operator):
    bl_idname = "f15_nozzle.toggle_demo"
    bl_label = "Toggle Actuation Cycle"
    bl_description = "Start or stop timeline playback of the nozzle actuation test cycle"

    def execute(self, context):
        scn = context.scene
        scn.f15_auto_demo = not scn.f15_auto_demo
        act = bpy.data.actions.get("Action_F15_GroundTest")
        if scn.f15_auto_demo:
            if act is not None:
                if scn.animation_data is None:
                    scn.animation_data_create()
                scn.animation_data.action = act
                if hasattr(scn.animation_data, "action_slot") and hasattr(act, "slots") and len(act.slots) > 0:
                    try:
                        scn.animation_data.action_slot = act.slots[0]
                    except Exception:
                        pass
            if context.screen and not context.screen.is_animation_playing:
                bpy.ops.screen.animation_play()
        else:
            if context.screen and context.screen.is_animation_playing:
                bpy.ops.screen.animation_play()
            if scn.animation_data is not None:
                scn.animation_data.action = None
        return {'FINISHED'}

class F15_PT_NozzleControlPanel(bpy.types.Panel):
    bl_label = "F100-PW-220 Control"
    bl_idname = "F15_PT_nozzle_control"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "F100 Nozzle"

    def draw(self, context):
        layout = self.layout
        scn = context.scene

        box_cid = layout.box()
        box_cid.label(text="Shading Display", icon='MATERIAL')
        row_cid = box_cid.row()
        row_cid.scale_y = 1.25
        is_cid = getattr(scn, "f15_color_id_mode", True)
        cid_txt = "Component ID Shading" if is_cid else "Neutral Clay Shading"
        row_cid.prop(scn, "f15_color_id_mode", text=cid_txt, toggle=True, icon='COLOR' if is_cid else 'SHADING_SOLID')

        box_main = layout.box()
        box_main.label(text="Nozzle Kinematics", icon='ARMATURE_DATA')
        col = box_main.column(align=True)
        col.scale_y = 1.25
        col.prop(scn, "f15_nozzle_open", text="Nozzle Aperture (A8)", slider=True)
        col.prop(scn, "f15_slider_mode", text="Exit Area Offset (A9)", slider=True)

        box_pre = layout.box()
        box_pre.label(text="Aperture Presets", icon='PRESET')
        row = box_pre.row(align=True)
        row.scale_y = 1.20
        op0 = row.operator("f15_nozzle.set_preset", text="Closed")
        op0.target_open = 0.0
        op0.target_slider = 0.0
        op1 = row.operator("f15_nozzle.set_preset", text="Intermediate")
        op1.target_open = 0.5
        op1.target_slider = 0.0
        op2 = row.operator("f15_nozzle.set_preset", text="Full Open")
        op2.target_open = 1.0
        op2.target_slider = 0.0

        box_rot = layout.box()
        box_rot.label(text="Rotor Assembly", icon='FORCE_VORTEX')
        box_rot.prop(scn, "f15_turbine_spin", text="Drive Rotor on Playback", toggle=True)
        box_rot.prop(scn, "f15_rotor_manual", text="Manual Rotor Angle (deg)", slider=True)

        box_demo = layout.box()
        box_demo.label(text="Actuation Cycle", icon='TIME')
        is_demo = getattr(scn, "f15_auto_demo", False)
        btn_txt = "Stop Actuation Cycle" if is_demo else "Run Actuation Cycle"
        btn_ico = 'PAUSE' if is_demo else 'PLAY'
        row_d = box_demo.row()
        row_d.scale_y = 1.25
        row_d.operator("f15_nozzle.toggle_demo", text=btn_txt, icon=btn_ico, depress=is_demo)

        box_ab = layout.box()
        box_ab.label(text="Visibility & Inspection", icon='LIGHT_SUN')
        box_ab.prop(scn, "f15_show_shroud", text="Outer Shroud", toggle=True, icon='MOD_SOLIDIFY')
        box_ab.prop(scn, "f15_show_stand", text="Maintenance Stand", toggle=True, icon='OBJECT_DATAMODE')
        row_tr = box_ab.row(align=True)
        row_tr.enabled = getattr(scn, "f15_show_stand", True)
        row_tr.prop(scn, "f15_show_trailer", text="Transport Trailer", toggle=True, icon='AUTO')
        box_ab.prop(scn, "f15_ab_glow", text="Internal Inspection Light", toggle=True, icon='SHADING_SOLID')
        sub = box_ab.row(align=True)
        sub.enabled = getattr(scn, "f15_ab_glow", False)
        sub.prop(scn, "f15_ab_glow_blue", text="Cool Inspection Tint", toggle=True)

_CLASSES = (
    F15_OT_SetNozzlePreset,
    F15_OT_ToggleGroundTestDemo,
    F15_PT_NozzleControlPanel,
)

def register_f15_ui():
    bpy.types.Scene.f15_color_id_mode = bpy.props.BoolProperty(
        name="Component ID Shading",
        description="Toggle between component-group color ID shading and neutral clay shading",
        default=True,
        update=_update_color_id_mode,
    )
    bpy.types.Scene.f15_nozzle_open = bpy.props.FloatProperty(
        name="Nozzle Aperture (A8)",
        description="Continuous throat aperture control from minimum (0.0) to maximum (1.0)",
        default=0.0, min=0.0, max=1.0, precision=3, subtype='FACTOR',
        update=_update_nozzle_props,
    )
    bpy.types.Scene.f15_slider_mode = bpy.props.FloatProperty(
        name="Exit Area Offset (A9)",
        description="Secondary carriage offset along the divergent guide rails",
        default=0.0, min=-1.0, max=1.0, precision=3,
        update=_update_nozzle_props,
    )
    bpy.types.Scene.f15_turbine_spin = bpy.props.BoolProperty(
        name="Drive Rotor on Playback",
        description="Rotate the low-pressure turbine rotor and spline shaft during timeline playback",
        default=True, update=_update_nozzle_props,
    )
    bpy.types.Scene.f15_rotor_manual = bpy.props.FloatProperty(
        name="Manual Rotor Angle",
        description="Manual angular offset for static rotor inspection (0 to 360 deg)",
        default=0.0, min=0.0, max=360.0, precision=1,
        update=_update_nozzle_props,
    )
    bpy.types.Scene.f15_auto_demo = bpy.props.BoolProperty(
        name="Actuation Cycle Active", default=False,
    )
    bpy.types.Scene.f15_show_shroud = bpy.props.BoolProperty(
        name="Outer Shroud",
        description="Toggle visibility of the outer fairing shroud",
        default=True, update=_update_shroud_vis,
    )
    bpy.types.Scene.f15_show_stand = bpy.props.BoolProperty(
        name="Maintenance Stand",
        description="Toggle visibility of the rollover maintenance stand and guide rails",
        default=True, update=_update_shroud_vis,
    )
    bpy.types.Scene.f15_show_trailer = bpy.props.BoolProperty(
        name="Transport Trailer",
        description="Toggle visibility of the four-wheel steerable transport chassis",
        default=True, update=_update_shroud_vis,
    )
    bpy.types.Scene.f15_ab_glow = bpy.props.BoolProperty(
        name="Internal Inspection Light",
        description="Toggle internal cavity illumination for interior inspection",
        default=True, update=_update_nozzle_props,
    )
    bpy.types.Scene.f15_ab_glow_blue = bpy.props.BoolProperty(
        name="Cool Inspection Tint",
        description="Switch internal illumination from neutral warm to cool blue",
        default=False, update=_update_nozzle_props,
    )
    for cls in _CLASSES:
        try:
            bpy.utils.unregister_class(cls)
        except Exception:
            pass
        bpy.utils.register_class(cls)

if __name__ == "__main__":
    register_f15_ui()
'''


def setup_ui_action_and_clay_lighting(mats: dict, coll_lights):
    txt_name = "f15_nozzle_ui.py"
    if txt_name in bpy.data.texts:
        bpy.data.texts.remove(bpy.data.texts[txt_name])
    txt = bpy.data.texts.new(txt_name)
    txt.write(UI_PANEL_SCRIPT_SOURCE)
    txt.use_module = True

    ns = {}
    exec(compile(UI_PANEL_SCRIPT_SOURCE, txt_name, 'exec'), ns)
    ns["register_f15_ui"]()

    scn = bpy.context.scene
    scn.frame_start = 1
    scn.frame_end = 240
    scn.render.fps = 30

    scn.f15_nozzle_open = 0.0
    scn.f15_slider_mode = 0.0
    for fr, u_v, m_v in [
        (1,   0.00,  0.00),
        (24,  0.00,  0.00),
        (50,  0.50,  0.00),
        (72,  0.50,  0.00),
        (98,  1.00,  0.00),
        (124, 1.00, +1.00),
        (150, 1.00, -0.75),
        (172, 1.00,  0.00),
        (202, 0.00,  0.00),
        (240, 0.00,  0.00),
    ]:
        scn.frame_set(fr)
        scn.f15_nozzle_open = u_v
        scn.f15_slider_mode = m_v
        scn.keyframe_insert(data_path="f15_nozzle_open", frame=fr)
        scn.keyframe_insert(data_path="f15_slider_mode", frame=fr)

    if scn.animation_data and scn.animation_data.action:
        act = scn.animation_data.action
        act.name = "Action_F15_GroundTest"
        act.use_fake_user = True
        scn.animation_data.action = None
    scn.frame_set(1)
    scn.f15_nozzle_open = 0.0
    scn.f15_slider_mode = 0.0
    scn.f15_turbine_spin = True
    scn.f15_rotor_manual = 0.0
    scn.f15_ab_glow = True
    scn.f15_ab_glow_blue = False
    scn.f15_color_id_mode = True
    for win in bpy.context.window_manager.windows:
        for area in win.screen.areas:
            if area.type == 'VIEW_3D':
                for sp in area.spaces:
                    if sp.type == 'VIEW_3D':
                        sp.shading.color_type = 'MATERIAL'


    def _add_ab_light(name: str, loc: tuple, base_energy: float, radius: float):
        ld = bpy.data.lights.new(name, type='POINT')
        ld.energy = base_energy
        ld.shadow_soft_size = radius
        ld.color = (1.0, 0.94, 0.88)
        lo = bpy.data.objects.new(name, ld)
        lo.location = loc
        coll_lights.objects.link(lo)

        fc_e = ld.driver_add("energy")
        d_e = fc_e.driver
        d_e.type = 'SCRIPTED'
        v_g = d_e.variables.new()
        v_g.name = "glow_on"
        v_g.type = 'SINGLE_PROP'
        v_g.targets[0].id_type = 'SCENE'
        v_g.targets[0].id = scn
        v_g.targets[0].data_path = "f15_ab_glow"
        d_e.expression = f"glow_on * {base_energy:.1f}"

        rgb_pairs = [(1.0, 0.45), (0.94, 0.72), (0.86, 1.0)]
        for idx, (c_warm, c_blue) in enumerate(rgb_pairs):
            fc_c = ld.driver_add("color", idx)
            dc = fc_c.driver
            dc.type = 'SCRIPTED'
            vb = dc.variables.new()
            vb.name = "is_blue"
            vb.type = 'SINGLE_PROP'
            vb.targets[0].id_type = 'SCENE'
            vb.targets[0].id = scn
            vb.targets[0].data_path = "f15_ab_glow_blue"
            dc.expression = f"{c_warm:.4f} + is_blue * ({c_blue - c_warm:.4f})"
        return lo

    _add_ab_light("Light_AB_Nozzle_Throat", (0.14, 0.28, 0.14),  6.5, 0.16)
    _add_ab_light("Light_AB_TailCone_Core", (0.08, -0.44, 0.08), 8.5, 0.12)
    _add_ab_light("Light_AB_Turbine_Deep",  (-0.08, -0.90, 0.06), 14.0, 0.18)

    glow_bsdf = mats["ab_glow_ring"].node_tree.nodes.get("AB_Glow_BSDF")
    if glow_bsdf and "Emission Strength" in glow_bsdf.inputs:
        fc_em = glow_bsdf.inputs["Emission Strength"].driver_add("default_value")
        dem = fc_em.driver
        dem.type = 'SCRIPTED'
        vg = dem.variables.new()
        vg.name = "glow_on"
        vg.type = 'SINGLE_PROP'
        vg.targets[0].id_type = 'SCENE'
        vg.targets[0].id = scn
        vg.targets[0].data_path = "f15_ab_glow"
        dem.expression = "glow_on * 0.12"


    world = bpy.data.worlds.new("World_Clay_Studio")
    scn.world = world
    wnt = world.node_tree
    wnt.nodes.clear()
    w_out = wnt.nodes.new("ShaderNodeOutputWorld")
    w_bg = wnt.nodes.new("ShaderNodeBackground")
    w_bg.inputs["Color"].default_value = rgba(0x101318)
    w_bg.inputs["Strength"].default_value = 0.32
    wnt.links.new(w_bg.outputs["Background"], w_out.inputs["Surface"])

    studio_lights = [
        ("Clay_Key_TopRight",   (2.35, 1.05, 2.35),   (math.radians(-44), math.radians(38), math.radians(16)),  310.0, 2.2, (1.0, 0.98, 0.96)),
        ("Clay_Fill_CoolLeft",  (-2.60, 0.20, 0.85),  (math.radians(-16), math.radians(-68), 0.0),              155.0, 2.6, (0.88, 0.93, 1.0)),
        ("Clay_Rim_BackBottom", (0.50, 2.65, -1.45),  (math.radians(118), math.radians(10), 0.0),               175.0, 1.9, (0.96, 0.98, 1.0)),
        ("Clay_Cavity_Rear",    (0.62, 3.10, 0.48),   (math.radians(82), math.radians(11), math.radians(169)),  58.0,  1.3, (0.98, 0.98, 1.0)),
        ("Clay_Front_Key",      (-2.35, -3.15, 0.85), (math.radians(72.0), 0.0, math.radians(-58.0)),           165.0, 2.6, (1.0, 0.98, 0.96)),
        ("Clay_Front_Fill",     (2.55, -3.45, 0.65),  (math.radians(76.0), 0.0, math.radians(68.0)),            175.0, 2.8, (0.92, 0.96, 1.0)),
        ("Clay_StbdFwd_Low",    (-2.45, -4.35, -0.45),(math.radians(96.0), 0.0, math.radians(-54.0)),           115.0, 2.4, (0.94, 0.96, 1.0)),
        ("Clay_PortFwd_Low",    (2.35, -3.95, -0.55), (math.radians(98.0), 0.0, math.radians(62.0)),            105.0, 2.4, (0.92, 0.95, 1.0)),
        ("Clay_Inlet_Axial",    (0.0, -6.40, 0.35),   (math.radians(84.0), 0.0, 0.0),                            95.0, 2.4, (0.98, 0.98, 1.0)),
    ]
    for lname, loc, rot_xyz, energy, sz, col in studio_lights:
        la = bpy.data.lights.new(lname, type='AREA')
        la.energy = energy
        la.size = sz
        la.color = col
        lo = bpy.data.objects.new(lname, la)
        lo.location = loc
        lo.rotation_euler = Euler(rot_xyz, 'XYZ')
        coll_lights.objects.link(lo)


def add_tracked_camera(name: str, loc: tuple, target_loc: tuple, lens: float, coll_cam):
    tgt = bpy.data.objects.new(f"{name}_Target", None)
    tgt.empty_display_size = 0.04
    tgt.location = target_loc
    coll_cam.objects.link(tgt)

    cd = bpy.data.cameras.new(name)
    cd.lens = lens
    cd.clip_start = 0.02
    cd.clip_end = 50.0
    co = bpy.data.objects.new(name, cd)
    co.location = loc
    coll_cam.objects.link(co)

    tr = co.constraints.new('TRACK_TO')
    tr.target = tgt
    tr.track_axis = 'TRACK_NEGATIVE_Z'
    tr.up_axis = 'UP_Y'
    return co


def report_polygon_statistics_and_verify_kinematics(arm_obj, spec: NozzleSpec):
    tot_v, tot_f, tot_t = 0, 0, 0
    print("\n================ [Geometry Summary] ================")
    for col in bpy.context.scene.collection.children:
        cv, cf, ct, co = 0, 0, 0, 0
        for obj in col.all_objects:
            if obj.type == 'MESH':
                me = obj.data
                me.calc_loop_triangles()
                cv += len(me.vertices)
                cf += len(me.polygons)
                ct += len(me.loop_triangles)
                co += 1
        if co > 0:
            print(f"  {col.name:<36} | Objects={co:3d} | Verts={cv:10,d} | Faces={cf:10,d} | Tris={ct:10,d}")
        tot_v += cv
        tot_f += cf
        tot_t += ct

    print(f"  {'[TOTAL]':<36} | Verts={tot_v:10,d} | Faces={tot_f:10,d} | Tris={tot_t:10,d}")
    assert tot_f >= 1_000_000, f"Polygon count below expected threshold: {tot_f:,}"

    tot_quads, tot_ngons, tot_tris_poly, tot_wn_objs, tot_mesh_objs, tot_sharp_edges = 0, 0, 0, 0, 0, 0
    seen_meshes = set()
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            tot_mesh_objs += 1
            if any(m.type == 'WEIGHTED_NORMAL' and m.keep_sharp for m in obj.modifiers):
                tot_wn_objs += 1
            me = obj.data
            for p in me.polygons:
                nv = p.loop_total
                if nv == 4:
                    tot_quads += 1
                elif nv > 4:
                    tot_ngons += 1
                else:
                    tot_tris_poly += 1
            if me.name not in seen_meshes:
                seen_meshes.add(me.name)
                sh_attr = me.attributes.get("sharp_edge")
                if sh_attr is not None:
                    tot_sharp_edges += sum(1 for d in sh_attr.data if d.value)
    quad_pct = 100.0 * tot_quads / max(1, tot_f)
    print(f"[Topology Check] Quads={quad_pct:.3f}% ({tot_quads:,} Quads / {tot_tris_poly:,} Tris / {tot_ngons:,} N-gons) | "
          f"WeightedNormal={tot_wn_objs}/{tot_mesh_objs} | SharpEdges={tot_sharp_edges:,}")
    assert tot_tris_poly == 0 and tot_ngons == 0, (
        f"Non-quad topology detected: Tris={tot_tris_poly}, Ngons={tot_ngons}"
    )
    assert tot_wn_objs == tot_mesh_objs, (
        f"Missing WeightedNormal modifier on mesh objects: {tot_wn_objs}/{tot_mesh_objs}"
    )

    scn = bpy.context.scene
    pb = arm_obj.pose.bones
    max_err_mm = 0.0
    print("\n================ [Kinematics Check] ================")
    test_states = [
        (0.00, -1.00),
        (0.00,  0.00),
        (0.28, -1.00),
        (0.25,  0.00),
        (0.50, -0.60),
        (0.50,  0.80),
        (0.75,  0.00),
        (1.00,  0.00),
        (1.00,  1.00),
    ]
    state_records = {}
    for u, sm in test_states:
        scn.f15_nozzle_open = u
        scn.f15_slider_mode = sm
        bpy.context.view_layer.update()
        e1 = (pb["B_ShortLink_00"].tail - pb["B_SyncLug_00"].head).length * 1000.0
        e2 = (pb["B_PullRod_00"].tail   - pb["B_BellOutTgt_00"].head).length * 1000.0
        e3 = (pb["B_DiagStrut_00"].tail - pb["B_BracketPS_00"].head).length * 1000.0
        e_boom_a = (pb["B_TransRodA_00"].tail - pb["B_SqBlkA_00"].head).length * 1000.0
        e_boom_b = (pb["B_TransRodB_00"].tail - pb["B_SqBlkB_15"].head).length * 1000.0
        ring_span_mm = (pb["B_RingStrapR_00"].head - pb["B_RingHubS_00"].head).length * 1000.0
        err_now = max(e1, e2, e3)
        max_err_mm = max(max_err_mm, err_now)
        sl_mm = pb["B_SliderCarriage_00"].location[1] * 1000.0
        state_records[(u, sm)] = {
            "sl_mm": sl_mm,
            "sl_rol_deg": math.degrees(pb["B_SliderRollerF_00"].rotation_euler[2]),
            "sy_rol_deg": math.degrees(pb["B_SyncRollerF_00"].rotation_euler[2]),
            "bal_ang_deg": math.degrees(pb["B_BalanceFlap_00"].rotation_euler[2]),
            "bal_rol_deg": math.degrees(pb["B_BalanceRoller_00"].rotation_euler[2]),
            "boom_deg": math.degrees(pb["B_Boomerang_00"].rotation_euler[2]),
            "e_boom_a": e_boom_a,
            "e_boom_b": e_boom_b,
            "ring_span_mm": ring_span_mm,
        }
        print(f"  u={u:.2f}, A9={sm:+.2f} | ThroatR={pb['B_ConvFlap_00'].tail.z:.4f}m | "
              f"ExitR={pb['B_DivFlap_00'].tail.z:.4f}m | Carriage={sl_mm:+6.2f}mm | Bellcrank={state_records[(u, sm)]['boom_deg']:+6.2f}deg | "
              f"RingSpan={ring_span_mm:6.2f}mm | PinErr={err_now:.5f}mm | RodErr={e_boom_a:.3f}/{e_boom_b:.3f}mm")

    auto_stroke_mm = state_records[(1.00, 0.00)]["sl_mm"] - state_records[(0.00, 0.00)]["sl_mm"]
    full_stroke_mm = max(v["sl_mm"] for v in state_records.values()) - min(v["sl_mm"] for v in state_records.values())
    sl_rol_sweep   = state_records[(1.00, 1.00)]["sl_rol_deg"] - state_records[(0.28, -1.00)]["sl_rol_deg"]
    sy_rol_sweep   = state_records[(1.00, 0.00)]["sy_rol_deg"] - state_records[(0.00, 0.00)]["sy_rol_deg"]
    bal_flap_sweep = state_records[(1.00, 0.00)]["bal_ang_deg"] - state_records[(0.00, 0.00)]["bal_ang_deg"]
    bal_rol_sweep  = state_records[(1.00, 0.00)]["bal_rol_deg"] - state_records[(0.00, 0.00)]["bal_rol_deg"]
    boom_sweep     = state_records[(1.00, 0.00)]["boom_deg"] - state_records[(0.00, 0.00)]["boom_deg"]
    max_boom_err   = max(max(v["e_boom_a"], v["e_boom_b"]) for v in state_records.values())
    min_ring_span  = min(v["ring_span_mm"] for v in state_records.values())
    max_ring_span  = max(v["ring_span_mm"] for v in state_records.values())

    scn.frame_set(1)
    bpy.context.view_layer.update()
    rot_f1 = math.degrees(pb["B_LPT_Rotor"].rotation_euler[1])
    scn.frame_set(11)
    bpy.context.view_layer.update()
    rot_f11 = math.degrees(pb["B_LPT_Rotor"].rotation_euler[1])
    lpt_spin_10f = rot_f11 - rot_f1

    scn.frame_set(1)
    scn.f15_nozzle_open = 0.0
    scn.f15_slider_mode = 0.0
    bpy.context.view_layer.update()

    print(f"  [Subsystem 1] Track carriage auto stroke = {auto_stroke_mm:.2f} mm | Total A9 range = {full_stroke_mm:.2f} mm | Roller sweep = {sl_rol_sweep:.1f} deg")
    print(f"  [Subsystem 2] Balance flap cam sweep = {bal_flap_sweep:+.2f} deg | Cam roller sweep = {bal_rol_sweep:.1f} deg")
    print(f"  [Subsystem 3] Sync ring guide roller sweep = {sy_rol_sweep:.1f} deg")
    print(f"  [Subsystem 4] LPT rotor 10-frame increment = {lpt_spin_10f:.2f} deg")
    print(f"  [Subsystem 5] Convergent bellcrank sweep = {boom_sweep:+.2f} deg | Max tie-rod error = {max_boom_err:.3f} mm")
    print(f"  [Subsystem 6] Structural ring strap span = {min_ring_span:.2f} to {max_ring_span:.2f} mm")
    assert max_err_mm < 0.05, f"Primary linkage closure error out of tolerance: {max_err_mm:.4f} mm"
    assert max_boom_err < 0.25, f"Bellcrank tie-rod error out of tolerance: {max_boom_err:.3f} mm"
    assert (max_ring_span - 21.5) < 105.5, f"Structural ring strap out of engagement range: {min_ring_span}..{max_ring_span}"
    assert abs(auto_stroke_mm - 32.0) < 0.2 and full_stroke_mm > 54.0, f"Carriage stroke out of tolerance: auto={auto_stroke_mm}, full={full_stroke_mm}"
    assert abs(lpt_spin_10f - 180.0) < 0.5, f"LPT rotor rotation increment out of tolerance: {lpt_spin_10f}"
    print(f"[Pass] Max primary pin error = {max_err_mm:.5f} mm; kinematic checks passed.\n")


def configure_render_and_viewport_clay(samples: int = 56):
    scn = bpy.context.scene
    scn.render.engine = 'CYCLES'
    scn.cycles.samples = samples
    scn.cycles.use_denoising = True
    try:
        prefs = bpy.context.preferences.addons['cycles'].preferences
        for dev_type in ('OPTIX', 'CUDA', 'HIP', 'ONEAPI'):
            try:
                prefs.compute_device_type = dev_type
                prefs.get_devices()
                gpus = [d for d in prefs.devices if d.type != 'CPU']
                if gpus:
                    for d in prefs.devices:
                        d.use = (d.type != 'CPU')
                    scn.cycles.device = 'GPU'
                    print(f"[Cycles] GPU backend enabled: {dev_type} ({gpus[0].name})")
                    break
            except Exception:
                continue
    except Exception:
        pass

    scn.render.resolution_x = 1440
    scn.render.resolution_y = 1000
    scn.render.resolution_percentage = 100
    scn.view_settings.view_transform = 'AgX'
    try:
        scn.view_settings.look = 'AgX - Medium High Contrast'
    except Exception:
        pass

    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == 'VIEW_3D':
                for space in area.spaces:
                    if space.type == 'VIEW_3D':
                        space.shading.type = 'SOLID'
                        space.shading.light = 'STUDIO'
                        space.shading.color_type = 'SINGLE'
                        space.shading.single_color = (0.85, 0.86, 0.88)
                        space.shading.show_cavity = True
                        space.shading.cavity_type = 'BOTH'
                        space.shading.cavity_ridge_factor = 0.90
                        space.shading.cavity_valley_factor = 0.90


def build_complete_f15_nozzle_scene(save_blend_path: str = None):
    bpy.ops.wm.read_factory_settings(use_empty=True)

    root_coll = bpy.context.scene.collection
    def _sub_coll(cname: str):
        c = bpy.data.collections.new(cname)
        root_coll.children.link(c)
        return c

    coll_fwd      = _sub_coll("00_Forward_Turbofan_Core_And_Fan")
    coll_static   = _sub_coll("01_Static_Base_Casing")
    coll_interior = _sub_coll("02_Afterburner_Interior")
    coll_moving   = _sub_coll("03_Articulated_Nozzle_And_Linkages")
    coll_rig      = _sub_coll("04_Kinematics_Rig")
    coll_lights   = _sub_coll("05_Studio_And_Cameras")
    coll_stand    = _sub_coll("06_Maintenance_Stand_And_Trailer")

    spec = NozzleSpec()
    mats = build_clay_materials()

    build_forward_turbofan_core_and_fan(spec, mats, coll_fwd)
    build_maintenance_stand_and_trailer(mats, coll_stand)
    build_static_base_and_interior(spec, mats, coll_static, coll_interior)
    setup_ui_action_and_clay_lighting(mats, coll_lights)
    arm_obj = build_armature_and_bind_parts(spec, mats, coll_moving, coll_rig)

    add_tracked_camera("Cam_01_Rear_IrisCheck",             (0.0, 3.25, 0.0),      (0.0, -0.35, 0.0),    56.0, coll_lights)
    add_tracked_camera("Cam_02_Side_LinkageCheck",          (-9.60, -2.22, 0.0),   (0.0, -2.22, 0.0),    50.0, coll_lights)
    cam_obli = add_tracked_camera("Cam_03_Oblique_Main",    (4.65, 2.85, 1.45),    (0.0, -1.85, 0.0),    42.0, coll_lights)
    add_tracked_camera("Cam_04_Macro_Plumbing_Lockwire",    (1.22, -0.95, 0.88),   (0.45, -1.35, 0.33),  66.0, coll_lights)
    add_tracked_camera("Cam_05_Macro_Track_FingerSeal",     (1.04, 0.94, 0.74),    (0.31, 0.44, 0.31),   68.0, coll_lights)
    add_tracked_camera("Cam_06_Macro_Balance_SealFlaps",    (1.16, 0.44, 0.80),    (0.42, 0.22, 0.29),   54.0, coll_lights)
    add_tracked_camera("Cam_07_Macro_BoatTail_Hoist",       (1.52, -1.25, 1.08),   (0.45, -1.80, 0.34),  48.0, coll_lights)
    add_tracked_camera("Cam_08_Macro_Inner_ThermalTiles",   (0.76, 1.56, 0.46),    (-0.24, 0.52, -0.22), 56.0, coll_lights)
    add_tracked_camera("Cam_09_Macro_Deep_Augmentor_Spray", (0.28, 1.38, 0.24),    (-0.22, -1.55, -0.22), 82.0, coll_lights)
    add_tracked_camera("Cam_10_Front_Inlet_Fan",            (-2.15, -6.95, 0.45),  (0.0, -4.30, 0.0),    48.0, coll_lights)
    bpy.context.scene.camera = cam_obli

    report_polygon_statistics_and_verify_kinematics(arm_obj, spec)
    configure_render_and_viewport_clay(samples=64)
    apply_procedural_pbr_to_current_scene(save_blend_path=None)

    bpy.context.scene.camera = cam_obli
    bpy.context.scene.f15_nozzle_open = 0.22
    bpy.context.scene.f15_ab_glow = True
    bpy.context.view_layer.update()

    if save_blend_path:
        bpy.ops.wm.save_as_mainfile(filepath=save_blend_path)
        print(f"[Saved] {save_blend_path}")


if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))
    blend_file = os.path.join(out_dir, "f15_exhaust_nozzle.blend")
    if "--refresh-materials" in sys.argv:
        apply_procedural_pbr_to_current_scene(save_blend_path=blend_file)
    elif "--refresh-stand" in sys.argv:
        refresh_stand_only_in_current_scene(save_blend_path=None)
        apply_procedural_pbr_to_current_scene(save_blend_path=blend_file)
    else:
        build_complete_f15_nozzle_scene(save_blend_path=blend_file)
