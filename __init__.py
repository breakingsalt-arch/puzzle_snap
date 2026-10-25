# SPDX-License-Identifier: GPL-3.0-or-later
"""
Puzzle Snap - move a piece near another mesh and it clicks into place.

Turn the tool ON in the sidebar (N > Puzzle). While it is ON, select a piece and
press G: the piece follows the mouse and snaps vertex-to-vertex / edge-to-edge onto
other meshes, like a puzzle. Place it with a click and (optionally) the pieces are
joined and the seam welded.

Keys while moving
  S               snapping on/off
  X / Y / Z       lock to a world axis
  R / Shift+R / Ctrl+R   rotate 90 degrees around Z / X / Y
  Middle mouse, wheel, numpad   navigate as usual
  LMB / Enter     place        RMB / Esc   cancel
"""

import math
from collections import Counter

import bmesh
import bpy
import gpu
import numpy as np
from bpy.props import (BoolProperty, CollectionProperty, EnumProperty, FloatProperty,
                       IntProperty, PointerProperty)
from bpy_extras import view3d_utils
from gpu_extras.batch import batch_for_shader
from mathutils import Matrix, Vector, kdtree

MAX_PIECE_POINTS = 600      # cap on moving points, keeps the live search fast


# ---------------------------------------------------------------------------
# Core helpers (no UI)
# ---------------------------------------------------------------------------

def mat_np(m):
    return np.array([[m[i][j] for j in range(4)] for i in range(4)], dtype=np.float64)


def to_world(local_pts, m4):
    """local_pts (N,3) -> world (N,3) using a 4x4 numpy matrix."""
    if len(local_pts) == 0:
        return np.empty((0, 3))
    return local_pts @ m4[:3, :3].T + m4[:3, 3]


def boundary_vertex_indices(me):
    """Indices of vertices on open edges (one face) or loose edges."""
    bm = bmesh.new()
    bm.from_mesh(me)
    found = set()
    for e in bm.edges:
        if e.is_boundary or e.is_wire:
            found.add(e.verts[0].index)
            found.add(e.verts[1].index)
    bm.free()
    return sorted(found)


def mesh_points(obj, open_edges_only):
    """Local-space vertex positions (N,3). With open_edges_only, just the vertices on
    open edges - unless the mesh is closed, in which case all vertices are used."""
    me = obj.data
    n = len(me.vertices)
    if n == 0:
        return np.empty((0, 3))
    co = np.empty(n * 3, dtype=np.float32)
    me.vertices.foreach_get("co", co)
    co = co.reshape(n, 3).astype(np.float64)
    if open_edges_only:
        idx = boundary_vertex_indices(me)
        if idx:
            return co[idx]
    return co


def thin(arr, cap):
    if len(arr) <= cap:
        return arr
    return arr[np.linspace(0, len(arr) - 1, cap).astype(int)]


def build_kd(points):
    kd = kdtree.KDTree(len(points))
    for i, p in enumerate(points.tolist()):
        kd.insert(p, i)
    kd.balance()
    return kd


class TargetSet:
    """World-space snap points of every mesh the piece can attach to."""

    def __init__(self, objects, open_edges_only):
        self.names = [o.name for o in objects]
        pts, owner = [], []
        for i, o in enumerate(objects):
            w = to_world(mesh_points(o, open_edges_only), mat_np(o.matrix_world))
            if len(w):
                pts.append(w)
                owner.append(np.full(len(w), i, dtype=np.int64))
        self.pts = np.concatenate(pts) if pts else np.empty((0, 3))
        self.owner = np.concatenate(owner) if owner else np.empty((0,), dtype=np.int64)
        self.kd = build_kd(self.pts) if len(self.pts) else None


def find_snap(piece_pts, target_pts, kd, radius, tol, min_matches,
              max_candidates=12, max_score_pts=250):
    """Translation that lines up the most piece points with target points.

    Returns (delta (3,), [matched target indices]) or None.
    """
    m = len(piece_pts)
    if m == 0 or kd is None:
        return None

    near = []
    for i, p in enumerate(piece_pts.tolist()):
        _co, idx, dist = kd.find(p)
        if dist <= radius:
            near.append((dist, i, idx))
    if not near:
        return None
    near.sort()

    step = max(tol * 0.5, 1e-6)
    seen, deltas = set(), []
    for _d, i, idx in near:
        delta = target_pts[idx] - piece_pts[i]
        key = tuple(np.round(delta / step).astype(np.int64).tolist())
        if key in seen:
            continue
        seen.add(key)
        deltas.append(delta)
        if len(deltas) >= max_candidates:
            break

    sample = piece_pts if m <= max_score_pts else piece_pts[np.linspace(0, m - 1, max_score_pts).astype(int)]
    need = min(min_matches, len(sample))
    best = None
    for delta in deltas:
        hits = set()
        for q in (sample + delta).tolist():
            _co, idx, dist = kd.find(q)
            if dist <= tol:
                hits.add(idx)
        score = (len(hits), -float(np.linalg.norm(delta)))
        if best is None or score > best[0]:
            best = (score, delta, hits)

    if best is None or best[0][0] < need:
        return None
    return best[1], sorted(best[2])


def fit_similarity(p, q):
    """Least-squares uniform scale s and translation t so that s*p + t ~ q."""
    pc, qc = p.mean(axis=0), q.mean(axis=0)
    p0, q0 = p - pc, q - qc
    denom = float((p0 * p0).sum())
    if denom < 1e-12:
        return None
    s = float((p0 * q0).sum()) / denom
    if s <= 0.0:
        return None
    return s, qc - s * pc


def find_snap_scale(piece_pts, target_pts, kd, radius, tol, min_matches, max_scale):
    """Fallback when a plain move can't line things up: also scale the piece (uniformly,
    about the world origin) so its points land on the target's.

    Returns (scale, translation (3,), [matched target indices]) or None.
    Applied as  x' = scale * x + translation.
    """
    m = len(piece_pts)
    if m < 2 or kd is None:
        return None
    need = max(2, min(min_matches, m))

    cur = piece_pts
    s, t = 1.0, np.zeros(3)
    for _ in range(6):
        pairs = []
        for i, p in enumerate(cur.tolist()):
            _co, idx, dist = kd.find(p)
            if dist <= radius:
                pairs.append((dist, i, idx))
        pairs.sort()
        used, pi, qi = set(), [], []
        for _d, i, idx in pairs:
            if idx in used:
                continue
            used.add(idx)
            pi.append(i)
            qi.append(idx)
        if len(pi) < 2:
            return None
        fit = fit_similarity(piece_pts[pi], target_pts[qi])
        if fit is None:
            return None
        s, t = fit
        if not (1.0 / max_scale <= s <= max_scale):
            return None
        cur = piece_pts * s + t

    hits = set()
    for q in cur.tolist():
        _co, idx, dist = kd.find(q)
        if dist <= tol:
            hits.add(idx)
    if len(hits) < need:
        return None
    return s, t, sorted(hits)


def weld_seam(obj, piece_world, target_world, tol):
    """Merge vertices that sit within `tol` of both the piece and the target."""
    kd_a = build_kd(piece_world)
    kd_b = build_kd(target_world)
    me = obj.data
    bm = bmesh.new()
    bm.from_mesh(me)
    mw = obj.matrix_world
    cand = []
    for v in bm.verts:
        w = mw @ v.co
        if kd_a.find(w)[2] <= tol and kd_b.find(w)[2] <= tol:
            cand.append(v)
    merged = 0
    if cand:
        scale = max(sum(mw.to_scale()) / 3.0, 1e-9)
        before = len(bm.verts)
        bmesh.ops.remove_doubles(bm, verts=cand, dist=tol / scale)
        merged = before - len(bm.verts)
    bm.to_mesh(me)
    bm.free()
    me.update()
    return merged


def connect_pieces(context, pieces, target, tol, weld):
    """Join `pieces` into `target`; optionally weld the seam. Returns merged vertex count."""
    piece_world = target_world = None
    if weld:
        piece_world = np.concatenate([
            to_world(mesh_points(o, False), mat_np(o.matrix_world)) for o in pieces
        ])
        target_world = to_world(mesh_points(target, False), mat_np(target.matrix_world))

    if target.data.users > 1:
        target.data = target.data.copy()

    bpy.ops.object.select_all(action='DESELECT')
    for o in pieces:
        o.select_set(True)
    target.select_set(True)
    context.view_layer.objects.active = target
    bpy.ops.object.join()

    if weld:
        return weld_seam(target, piece_world, target_world, tol)
    return 0


def _unique(objs):
    seen, out = set(), []
    for o in objs:
        if o.name not in seen:
            seen.add(o.name)
            out.append(o)
    return out


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

class PuzzleSnapObjRef(bpy.types.PropertyGroup):
    obj: PointerProperty(type=bpy.types.Object)


POINT_ITEMS = [
    ('OPEN', "Open Edges", "Use only vertices on open edges (the connector ends). Closed meshes use all vertices"),
    ('ALL', "All Vertices", "Use every vertex"),
]


class PuzzleSnapSettings(bpy.types.PropertyGroup):
    # --- source / target selection
    use_custom_source: BoolProperty(
        name="Custom Source",
        description="Move exactly the objects in the Source list (no need to select them). "
                    "When off, the selected meshes are the source",
        default=False)
    use_custom_target: BoolProperty(
        name="Custom Target",
        description="Snap only to the objects in the Target list. When off, snap to every other visible mesh",
        default=False)
    sources: CollectionProperty(type=PuzzleSnapObjRef)
    targets: CollectionProperty(type=PuzzleSnapObjRef)

    # --- snapping
    snap_distance: FloatProperty(
        name="Snap Distance",
        description="How close (in scene units) the piece's points must get to another mesh before it snaps",
        default=0.5, min=0.001, soft_max=10.0, subtype='DISTANCE', unit='LENGTH')
    tolerance: FloatProperty(
        name="Match Tolerance",
        description="How exactly two vertices must line up to count as a match, and how close they must be to get welded",
        default=0.05, min=0.0001, soft_max=1.0, subtype='DISTANCE', unit='LENGTH')
    min_matches: IntProperty(
        name="Min Matching Vertices",
        description="Vertices that must line up at once before it snaps. 1 = single corner, 2+ = a whole edge",
        default=2, min=1, max=64)
    source_points: EnumProperty(
        name="Source Vertices", description="Which vertices of the moving piece do the snapping",
        items=POINT_ITEMS, default='OPEN')
    target_points: EnumProperty(
        name="Target Vertices", description="Which vertices of the other meshes can be snapped to",
        items=POINT_ITEMS, default='OPEN')

    # --- scale
    scale_to_fit: BoolProperty(
        name="Scale To Fit",
        description="If a plain move can't line the vertices up, also scale the piece (uniformly) so its "
                    "vertices land on the target's",
        default=False)
    max_scale: FloatProperty(
        name="Max Scale Change",
        description="Largest allowed scale change. 2 means the piece may grow up to 2x or shrink down to 0.5x",
        default=2.0, min=1.01, soft_max=5.0, max=20.0)

    # --- connect
    connect_mode: EnumProperty(
        name="Connect Mode", description="What happens when you place a snapped piece",
        items=[
            ('SNAP', "Snap Only", "Leave the pieces as separate objects"),
            ('JOIN', "Join", "Join the piece into the object it snapped to"),
            ('WELD', "Join + Weld", "Join, then merge the vertices along the seam so the edges are truly connected"),
        ],
        default='WELD')


# ---------------------------------------------------------------------------
# Viewport feedback
# ---------------------------------------------------------------------------

_draw_state = {"pts": [], "handle": None}


def _draw_cb():
    pts = _draw_state["pts"]
    if not pts:
        return
    try:
        shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        batch = batch_for_shader(shader, 'POINTS', {"pos": pts})
        gpu.state.depth_test_set('NONE')
        gpu.state.point_size_set(11.0)
        shader.bind()
        shader.uniform_float("color", (0.1, 1.0, 0.35, 1.0))
        batch.draw(shader)
        gpu.state.point_size_set(1.0)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Assign source / target
# ---------------------------------------------------------------------------

class PUZZLESNAP_OT_assign(bpy.types.Operator):
    bl_idname = "puzzlesnap.assign"
    bl_label = "Assign Source / Target"
    bl_description = "Store the selected mesh objects as the Source or Target list"
    bl_options = {'INTERNAL'}

    role: EnumProperty(items=[('SOURCE', "Source", ""), ('TARGET', "Target", "")])
    action: EnumProperty(items=[('SET', "Set", ""), ('CLEAR', "Clear", "")], default='SET')

    @classmethod
    def description(cls, context, properties):
        if properties.action == 'CLEAR':
            return f"Clear the {properties.role.lower()} list and go back to the automatic behaviour"
        if properties.role == 'SOURCE':
            return "Use the selected meshes as the Source: the objects that move and do the snapping"
        return "Use the selected meshes as the Target: the objects the source snaps onto"

    def execute(self, context):
        s = context.scene.puzzle_snap
        is_source = self.role == 'SOURCE'
        coll = s.sources if is_source else s.targets
        flag = "use_custom_source" if is_source else "use_custom_target"
        label = "Source" if is_source else "Target"

        if self.action == 'CLEAR':
            coll.clear()
            setattr(s, flag, False)
            self.report({'INFO'}, f"{label} list cleared")
            return {'FINISHED'}

        objs = _unique([o for o in context.selected_objects if o.type == 'MESH'])
        if not objs:
            self.report({'ERROR'}, "Select one or more mesh objects first")
            return {'CANCELLED'}
        coll.clear()
        for o in objs:
            coll.add().obj = o
        setattr(s, flag, True)
        self.report({'INFO'}, f"{label}: {len(objs)} object(s) set")
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# The modal tool
# ---------------------------------------------------------------------------

class PUZZLESNAP_OT_move(bpy.types.Operator):
    bl_idname = "puzzlesnap.move"
    bl_label = "Puzzle Move"
    bl_description = "Move the source piece(s); they snap onto the target meshes edge-to-edge like a puzzle"
    bl_options = {'REGISTER', 'UNDO'}

    # ---- setup -----------------------------------------------------------
    def invoke(self, context, event):
        passthru = {'CANCELLED', 'PASS_THROUGH'}     # lets the normal Grab run instead
        area = context.area
        if area is None or area.type != 'VIEW_3D':
            self.report({'ERROR'}, "Run this from the 3D Viewport")
            return {'CANCELLED'}
        if context.mode != 'OBJECT':
            return passthru

        cfg = context.scene.puzzle_snap
        layer_objs = context.view_layer.objects

        def usable(o):
            return o is not None and o.type == 'MESH' and o.name in layer_objs and o.visible_get()

        if cfg.use_custom_source:
            moving = _unique([i.obj for i in cfg.sources if usable(i.obj)])
            if not moving:
                self.report({'WARNING'}, "Custom Source is on, but no visible source object is set")
                return passthru
        else:
            moving = _unique([o for o in context.selected_objects if usable(o)])
            if not moving:
                return passthru

        moving_names = {o.name for o in moving}
        if cfg.use_custom_target:
            targets = _unique([i.obj for i in cfg.targets if usable(i.obj) and i.obj.name not in moving_names])
            if not targets:
                self.report({'WARNING'}, "Custom Target is on, but no visible target object is set")
                return passthru
        else:
            targets = [o for o in layer_objs if usable(o) and o.name not in moving_names]
            if not targets:
                return passthru

        self.targets = TargetSet(targets, cfg.target_points == 'OPEN')
        if self.targets.kd is None:
            self.report({'WARNING'}, "The target meshes have no vertices")
            return passthru

        per_obj_cap = max(50, MAX_PIECE_POINTS // len(moving))
        self.local, self.base, self.orig = {}, {}, {}
        for o in moving:
            pts = thin(mesh_points(o, cfg.source_points == 'OPEN'), per_obj_cap)
            if len(pts) == 0:
                continue
            self.local[o.name] = pts
            self.base[o.name] = o.matrix_world.copy()
            self.orig[o.name] = o.matrix_world.copy()
        if not self.local:
            return passthru

        self.region = next((r for r in area.regions if r.type == 'WINDOW'), None)
        self.rv3d = area.spaces.active.region_3d
        if self.region is None or self.rv3d is None:
            self.report({'ERROR'}, "Could not find the 3D view")
            return {'CANCELLED'}

        self.free = Vector((0.0, 0.0, 0.0))          # where the mouse dragged the piece
        self.snap_delta = Vector((0.0, 0.0, 0.0))
        pts = self.world_points()
        self.depth_point = Vector(((pts.min(axis=0) + pts.max(axis=0)) / 2.0).tolist())
        self.axis = None
        self.snap_on = True
        self.snapped = False
        self.matched = 0
        self.snap_scale = 1.0
        self.matched_owner = None
        self.prev_coord = self.mouse_coord(event)
        self._cleaned = False

        _draw_state["pts"] = []
        _draw_state["handle"] = bpy.types.SpaceView3D.draw_handler_add(_draw_cb, (), 'WINDOW', 'POST_VIEW')
        self.update_status(context)
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    # ---- helpers ---------------------------------------------------------
    def mouse_coord(self, event):
        return Vector((event.mouse_x - self.region.x, event.mouse_y - self.region.y))

    def world_points(self):
        t = Matrix.Translation(self.free)
        return np.concatenate([to_world(local, mat_np(t @ self.base[name]))
                               for name, local in self.local.items()])

    def update_status(self, context):
        if not self.snap_on:
            state = "SNAP OFF"
        elif self.snapped:
            state = f"SNAPPED - {self.matched} vertices matched"
            if abs(self.snap_scale - 1.0) > 1e-4:
                state += f", scaled x{self.snap_scale:.2f}"
        else:
            state = "searching..."
        axis = f"  Axis: {self.axis}" if self.axis else ""
        context.workspace.status_text_set(
            "Puzzle Snap | LMB/Enter: place   RMB/Esc: cancel   S: snapping   "
            f"X/Y/Z: axis   R / Shift+R / Ctrl+R: rotate 90\u00b0 | {state}{axis}")

    def apply(self, context):
        cfg = context.scene.puzzle_snap
        pts = self.world_points()
        total = Matrix.Translation(self.free)
        tidx = None
        self.snap_scale = 1.0

        if self.snap_on:
            result = find_snap(pts, self.targets.pts, self.targets.kd,
                               cfg.snap_distance, cfg.tolerance, cfg.min_matches)
            if result:
                delta, tidx = result
                total = Matrix.Translation(self.free + Vector(delta.tolist()))
            elif cfg.scale_to_fit:
                res = find_snap_scale(pts, self.targets.pts, self.targets.kd, cfg.snap_distance,
                                      cfg.tolerance, cfg.min_matches, cfg.max_scale)
                if res:
                    s, t, tidx = res
                    self.snap_scale = s
                    total = (Matrix.Translation(Vector(t.tolist()))
                             @ Matrix.Scale(s, 4)
                             @ Matrix.Translation(self.free))

        if tidx:
            self.snapped = True
            self.matched = len(tidx)
            _draw_state["pts"] = [tuple(self.targets.pts[i]) for i in tidx]
            owners = Counter(int(self.targets.owner[i]) for i in tidx)
            self.matched_owner = owners.most_common(1)[0][0]
        else:
            self.snapped = False
            self.matched = 0
            self.matched_owner = None
            _draw_state["pts"] = []

        for name, base in self.base.items():
            ob = bpy.data.objects.get(name)
            if ob:
                ob.matrix_world = total @ base
        self.update_status(context)
        context.area.tag_redraw()

    def drag(self, event):
        coord = self.mouse_coord(event)
        depth = self.depth_point + self.free
        a = view3d_utils.region_2d_to_location_3d(self.region, self.rv3d, self.prev_coord, depth)
        b = view3d_utils.region_2d_to_location_3d(self.region, self.rv3d, coord, depth)
        delta = b - a
        if self.axis:
            keep = {'X': 0, 'Y': 1, 'Z': 2}[self.axis]
            delta = Vector((delta[0] if keep == 0 else 0.0,
                            delta[1] if keep == 1 else 0.0,
                            delta[2] if keep == 2 else 0.0))
        self.free += delta
        self.prev_coord = coord

    def rotate(self, axis):
        pivot = self.depth_point + self.free
        rot = (Matrix.Translation(pivot)
               @ Matrix.Rotation(math.radians(90.0), 4, axis)
               @ Matrix.Translation(-pivot))
        t = Matrix.Translation(self.free)
        for name in self.base:
            self.base[name] = rot @ t @ self.base[name]
        self.depth_point = pivot
        self.free = Vector((0.0, 0.0, 0.0))

    # ---- finishing -------------------------------------------------------
    def cleanup(self, context):
        if getattr(self, "_cleaned", True):
            return
        self._cleaned = True
        if _draw_state["handle"] is not None:
            bpy.types.SpaceView3D.draw_handler_remove(_draw_state["handle"], 'WINDOW')
            _draw_state["handle"] = None
        _draw_state["pts"] = []
        try:
            context.workspace.status_text_set(None)
            if context.area:
                context.area.tag_redraw()
        except Exception:
            pass

    def restore(self):
        for name, mat in self.orig.items():
            ob = bpy.data.objects.get(name)
            if ob:
                ob.matrix_world = mat

    def confirm(self, context):
        cfg = context.scene.puzzle_snap
        snapped, owner, matched = self.snapped, self.matched_owner, self.matched
        self.cleanup(context)

        if snapped and cfg.connect_mode in {'JOIN', 'WELD'} and owner is not None:
            target = bpy.data.objects.get(self.targets.names[owner])
            pieces = [bpy.data.objects[n] for n in self.base if n in bpy.data.objects]
            if target and pieces:
                weld = cfg.connect_mode == 'WELD'
                merged = connect_pieces(context, pieces, target, cfg.tolerance, weld)
                if weld:
                    self.report({'INFO'}, f"Connected to '{target.name}': {matched} vertices matched, {merged} welded")
                else:
                    self.report({'INFO'}, f"Joined into '{target.name}'")
                return {'FINISHED'}
        if snapped:
            self.report({'INFO'}, f"Snapped: {matched} vertices matched")
        return {'FINISHED'}

    def cancel(self, context):
        self.cleanup(context)

    # ---- events ----------------------------------------------------------
    def modal(self, context, event):
        pressed = event.value == 'PRESS'

        if pressed and event.type in {'LEFTMOUSE', 'RET', 'NUMPAD_ENTER'}:
            return self.confirm(context)
        if pressed and event.type in {'RIGHTMOUSE', 'ESC'}:
            self.restore()
            self.cleanup(context)
            return {'CANCELLED'}

        if (event.type in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE', 'TRACKPADPAN',
                           'TRACKPADZOOM', 'MOUSEROTATE'}
                or event.type.startswith('NUMPAD') or event.type.startswith('NDOF')):
            return {'PASS_THROUGH'}

        if event.type == 'MOUSEMOVE':
            self.drag(event)
            self.apply(context)
        elif pressed and event.type in {'X', 'Y', 'Z'}:
            self.axis = None if self.axis == event.type else event.type
            self.update_status(context)
        elif pressed and event.type == 'S':
            self.snap_on = not self.snap_on
            self.apply(context)
        elif pressed and event.type == 'R':
            self.rotate('X' if event.shift else 'Y' if event.ctrl else 'Z')
            self.apply(context)

        return {'RUNNING_MODAL'}


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def _draw_object_list(layout, coll):
    names = [i.obj.name for i in coll if i.obj]
    if not names:
        layout.label(text="Nothing set", icon='INFO')
        return
    for n in names[:5]:
        layout.label(text=n, icon='MESH_DATA')
    if len(names) > 5:
        layout.label(text=f"... and {len(names) - 5} more")


class PUZZLESNAP_PT_panel(bpy.types.Panel):
    bl_label = "Puzzle Snap"
    bl_idname = "PUZZLESNAP_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Puzzle"

    def draw(self, context):
        layout = self.layout
        wm = context.window_manager
        s = context.scene.puzzle_snap
        on = wm.puzzle_snap_enabled

        col = layout.column()
        col.scale_y = 1.6
        col.prop(wm, "puzzle_snap_enabled",
                 text="Puzzle Snap: ON" if on else "Puzzle Snap: OFF",
                 toggle=True, icon='SNAP_ON' if on else 'SNAP_OFF')
        if on:
            layout.label(text="Select a piece and press G", icon='INFO')

        # --- source
        box = layout.box()
        row = box.row(align=True)
        row.prop(s, "use_custom_source", text="Source (what moves)")
        row = box.row(align=True)
        op = row.operator("puzzlesnap.assign", text="Set Source from Selection", icon='RESTRICT_SELECT_OFF')
        op.role, op.action = 'SOURCE', 'SET'
        op = row.operator("puzzlesnap.assign", text="", icon='X')
        op.role, op.action = 'SOURCE', 'CLEAR'
        if s.use_custom_source:
            _draw_object_list(box.column(align=True), s.sources)
        else:
            box.label(text="Using: selected objects")

        # --- target
        box = layout.box()
        row = box.row(align=True)
        row.prop(s, "use_custom_target", text="Target (snaps onto)")
        row = box.row(align=True)
        op = row.operator("puzzlesnap.assign", text="Set Target from Selection", icon='RESTRICT_SELECT_OFF')
        op.role, op.action = 'TARGET', 'SET'
        op = row.operator("puzzlesnap.assign", text="", icon='X')
        op.role, op.action = 'TARGET', 'CLEAR'
        if s.use_custom_target:
            _draw_object_list(box.column(align=True), s.targets)
        else:
            box.label(text="Using: all other visible meshes")

        # --- snapping
        col = layout.column(align=True)
        col.label(text="Snapping")
        col.prop(s, "snap_distance")
        col.prop(s, "tolerance")
        col.prop(s, "min_matches")
        col.prop(s, "source_points")
        col.prop(s, "target_points")

        # --- scale
        col = layout.column(align=True)
        col.prop(s, "scale_to_fit")
        sub = col.column(align=True)
        sub.active = s.scale_to_fit
        sub.prop(s, "max_scale")

        # --- connect
        col = layout.column(align=True)
        col.label(text="When placed")
        col.prop(s, "connect_mode", text="")

        box = layout.box()
        box.label(text="While moving:")
        for line in ("LMB / Enter  place", "RMB / Esc  cancel", "S  snapping on/off",
                     "X / Y / Z  lock axis", "R, Shift+R, Ctrl+R  rotate 90\u00b0"):
            box.label(text=line)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

classes = (PuzzleSnapObjRef, PuzzleSnapSettings, PUZZLESNAP_OT_assign,
           PUZZLESNAP_OT_move, PUZZLESNAP_PT_panel)
_keymaps = []          # (keymap, item) pairs; the G override is only active while the tool is ON


def _on_toggle(self, context):
    for _km, kmi in _keymaps:
        kmi.active = self.puzzle_snap_enabled


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.puzzle_snap = PointerProperty(type=PuzzleSnapSettings)
    bpy.types.WindowManager.puzzle_snap_enabled = BoolProperty(
        name="Puzzle Snap",
        description="While on, pressing G in Object Mode moves the selected piece with puzzle snapping",
        default=False, update=_on_toggle)

    kc = bpy.context.window_manager.keyconfigs.addon
    if kc:
        km = kc.keymaps.new(name='Object Mode', space_type='EMPTY')
        kmi = km.keymap_items.new(PUZZLESNAP_OT_move.bl_idname, 'G', 'PRESS')
        kmi.active = False
        _keymaps.append((km, kmi))


def unregister():
    for km, kmi in _keymaps:
        km.keymap_items.remove(kmi)
    _keymaps.clear()
    if _draw_state["handle"] is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_draw_state["handle"], 'WINDOW')
        _draw_state["handle"] = None
    del bpy.types.WindowManager.puzzle_snap_enabled
    del bpy.types.Scene.puzzle_snap
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
