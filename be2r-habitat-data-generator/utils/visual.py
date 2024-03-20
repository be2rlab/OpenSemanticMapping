import numpy as np
from matplotlib import pyplot as plt

from PIL import Image

from habitat_sim.utils.common import d3_40_colors_rgb

d3_150_colors_rgb = np.random.default_rng(200).integers(0, 256, (150, 3), dtype=np.uint8)

def display_sample(rgb_obs, semantic_obs=None, depth_obs=None):
    rgb_img = Image.fromarray(rgb_obs, mode="RGBA")

    arr = [rgb_img]
    titles = ["rgb"]
    if semantic_obs is not None:
        semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
        semantic_img.putpalette(d3_150_colors_rgb.flatten())
        semantic_img.putdata((semantic_obs.flatten() % 150).astype(np.uint8))
        semantic_img = semantic_img.convert("RGBA")
        arr.append(semantic_img)
        titles.append("semantic")

    if depth_obs is not None:
        depth_img = Image.fromarray((depth_obs / 10 * 255).astype(np.uint8), mode="L")
        arr.append(depth_img)
        titles.append("depth")

    plt.figure(figsize=(12, 8))
    for i, data in enumerate(arr):
        ax = plt.subplot(1, 3, i + 1)
        ax.axis("off")
        ax.set_title(titles[i])
        plt.imshow(data)
    plt.show(block=False)


def convert_points_to_topdown(pathfinder, points, meters_per_pixel):
    points_topdown = []
    bounds = pathfinder.get_bounds()
    for point in points:
        # convert 3D x,z to topdown x,y
        px = (point[0] - bounds[0][0]) / meters_per_pixel
        py = (point[2] - bounds[0][2]) / meters_per_pixel
        points_topdown.append(np.array([px, py]))
    return points_topdown


# display a topdown map with matplotlib
def display_map(topdown_map, agent_position=None, key_points=None):
    plt.figure(figsize=(12, 8))
    ax = plt.subplot(1, 1, 1)
    ax.axis("off")
    ax.imshow(topdown_map)
    # plot points on map

    if agent_position is not None:
        ax.plot(agent_position[0], agent_position[1], marker='*', color='red', markersize=10, alpha=0.8, ls='', label='Start point')

    if key_points is not None:
        for i, point in enumerate(key_points):
            ax.plot(point[0], point[1], marker="o", markersize=10, alpha=0.8, ls='', label=f'Goal {i+1}')

    ax.legend()
    plt.show()