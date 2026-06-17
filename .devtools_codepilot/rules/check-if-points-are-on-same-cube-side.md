---
globs: "**/*.py"
description: When working with a cube where each face lies at a fixed coordinate
  (e.g., ±42), two points are on the same side if one of their coordinates
  matches exactly (same value and sign). This check should be performed along
  the axis where the cube face is aligned.
---

To determine if two points lie on the same side of a cube, verify that one of their coordinates (x, y, or z) has the same absolute value (e.g., 42) and the same sign, indicating they are on the same face. Compare the coordinate that is constant across cube faces.