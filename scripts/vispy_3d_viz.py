import argparse

import numpy as np
from vispy import app, scene

parser = argparse.ArgumentParser()
parser.add_argument("--data", help="Path to the data file", required=True)


def visualize(data, cmap="inferno"):
    canvas = scene.SceneCanvas(keys="interactive", show=True)
    view = canvas.central_widget.add_view()
    volume = scene.visuals.Volume(data, parent=view.scene, threshold=0, cmap=cmap)
    view.camera = scene.cameras.TurntableCamera(
        parent=view.scene, fov=60, name="Turntable"
    )
    axis = scene.visuals.XYZAxis(parent=view.scene)
    app.run()


def main(args):
    if "heat" in args.data:
        cmap = "turbo"
    else:
        cmap = "inferno"
    data = np.load(args.data)
    print(f"Data shape: {data.shape}")
    visualize(data, cmap)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
