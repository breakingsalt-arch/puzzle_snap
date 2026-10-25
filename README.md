# Puzzle Snap

Move a mesh piece near another mesh and it clicks into place, vertex-to-vertex and
edge-to-edge, like a puzzle. Optionally join the pieces and weld the seam so the
edges are truly connected. Great for modular kits (sewers, corridors, pipes, walls).

Requires Blender 4.2 or newer.

## Use
1. Open the sidebar (`N`) > **Puzzle** tab and switch **Puzzle Snap: ON**.
2. Select a piece and press **G**. Move the mouse; when its open edges come close to
   another mesh it snaps and green dots show the matched vertices.
3. Left click / Enter to place it, right click / Esc to cancel.

While moving: `S` snapping on/off, `X` `Y` `Z` lock axis, `R` / `Shift+R` / `Ctrl+R`
rotate 90 degrees around Z / X / Y. Middle mouse, wheel and numpad still navigate.

With the tool OFF, `G` behaves exactly like normal Blender. If nothing meshy is
selected, `G` also falls back to the normal grab.

## Source and Target
* **Source** = what moves. Default: the selected meshes. Press *Set Source from
  Selection* to lock it to specific objects (no need to select them afterwards).
* **Target** = what it snaps onto. Default: every other visible mesh. Press *Set Target
  from Selection* to restrict snapping to specific objects.
* The X buttons clear a list and return to the automatic behaviour.
* *Source Vertices* / *Target Vertices* choose between open-edge vertices only (the
  connector ends) and all vertices.

## Scale To Fit
When a plain move cannot line the vertices up (for example two pipe openings with
different diameters), the piece is also scaled uniformly so its vertices land on the
target's. *Max Scale Change* limits how far it may scale.

## Settings
* **Snap Distance** - how close the piece must get before it snaps.
* **Match Tolerance** - how exactly vertices must line up, and the weld distance.
* **Min Matching Vertices** - 1 = single corner, 2+ = a whole edge must fit.
* **When placed** - Snap Only, Join, or Join + Weld.

## Limits
Uses each object's base mesh (modifiers and shape keys are ignored). Very dense
pieces are sampled to stay responsive. Join + Weld edits the target mesh; use undo to
revert.

## License
GPL-3.0-or-later.
