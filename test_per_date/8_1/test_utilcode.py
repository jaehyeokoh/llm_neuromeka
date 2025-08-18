# Visual debug + final collision function
# - analytic_collision(): final function returning True/False
# - debug_random_scene(): generate 5 random obstacles, draw scene, print verdicts
#
# Conventions
# - start, end: [x, y]
# - obstacles: [[x, y, area], ...]
# - agent_radius is added to obstacle radius (Minkowski sum).
#
# Coordinate ranges are set to x∈[0.1,0.7], y∈[0.0,0.4] by default for debug.
#
# You can run debug_random_scene() multiple times to eyeball robustness.
# When you're satisfied, just use analytic_collision() with your own inputs.


import math
from typing import List, Tuple, Optional
import random

import numpy as np

import matplotlib.pyplot as plt

# ---------- Core utilities ----------

def area_to_radius(area: float) -> float:
    if area < 0:
        raise ValueError("area must be non-negative")
    return math.sqrt(area / math.pi)

def _segment_circle_intersect(p0, p1, c, r, eps=1e-9) -> bool:
    """Nearest-point method: O(1), continuous, robust."""
    x0, y0 = p0; x1, y1 = p1; cx, cy = c
    vx, vy = x1 - x0, y1 - y0
    wx, wy = cx - x0, cy - y0
    vv = vx*vx + vy*vy
    if vv <= eps:  # degenerate: point
        dx, dy = x0 - cx, y0 - cy
        return (dx*dx + dy*dy) <= (r*r + eps)
    t = (wx*vx + wy*vy) / vv
    if t < 0.0: t = 0.0
    elif t > 1.0: t = 1.0
    qx, qy = x0 + t*vx, y0 + t*vy
    dx, dy = qx - cx, qy - cy
    return (dx*dx + dy*dy) <= (r*r + eps)

# ---------- Final function ----------

def analytic_collision(start: List[float], end: List[float], obstacles: List[List[float]], agent_radius: float = 0.0) -> bool:
    """
    Returns True if the segment start->end collides with any circular obstacle.
    Obstacles are [x, y, area]. agent_radius is added to each obstacle radius.
    """
    p0 = (float(start[0]), float(start[1]))
    p1 = (float(end[0]), float(end[1]))
    for (ox, oy, area) in obstacles:
        r_eff = area_to_radius(float(area)) + float(agent_radius)
        if _segment_circle_intersect(p0, p1, (float(ox), float(oy)), r_eff):
            return True
    return False

# ---------- Visual debug ----------

def debug_random_scene(
    n_obstacles: int = 5,
    fixed_obstacle_radius: float = 0.04,
    agent_radius: float = 0.03,
    x_range: Tuple[float, float] = (0.1, 0.7),
    y_range: Tuple[float, float] = (0.0, 0.4),
    seed: Optional[int] = None
):
    """
    Random scene generator and visualizer.
    Draws obstacles (actual r and effective r+r_agent), the segment, and prints collision verdict.
    """
    rng = random.Random(seed)
    def randf(a,b): return rng.uniform(a,b)

    start = [randf(*x_range), randf(*y_range)]
    end   = [randf(*x_range), randf(*y_range)]
    area  = math.pi * (fixed_obstacle_radius ** 2)
    obstacles = [[randf(*x_range), randf(*y_range), area] for _ in range(n_obstacles)]

    collided = analytic_collision(start, end, obstacles, agent_radius=agent_radius)

    # Plot
    fig = plt.figure(figsize=(6,5))
    ax = plt.gca()
    # segment
    ax.plot([start[0], end[0]], [start[1], end[1]], label="segment start→end")
    ax.scatter([start[0], end[0]], [start[1], end[1]], s=25)

    theta = np.linspace(0, 2*np.pi, 256)
    for i,(ox,oy,area_i) in enumerate(obstacles):
        r = area_to_radius(area_i)
        # actual obstacle
        cx = ox + r*np.cos(theta)
        cy = oy + r*np.sin(theta)
        ax.plot(cx, cy, label=f"obs {i} (r={r:.3f})")
        # effective radius for collision (r + agent_radius)
        reff = r + agent_radius
        cxe = ox + reff*np.cos(theta)
        cye = oy + reff*np.sin(theta)
        ax.plot(cxe, cye, linestyle="--", label=f"obs {i} effective r={reff:.3f}")

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(x_range[0]-0.02, x_range[1]+0.02)
    ax.set_ylim(y_range[0]-0.02, y_range[1]+0.02)
    ax.set_title(f"Visual debug | collided = {collided}")
    ax.legend(loc="best")
    plt.show()

    print("Start:", start)
    print("End:", end)
    print("Obstacles (x, y, area):", obstacles)
    print("Agent radius:", agent_radius)
    print("Collision verdict (analytic):", collided)

# --------- Run one debug scene ---------
debug_random_scene(seed=1234)
