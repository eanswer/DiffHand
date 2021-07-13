import os
import shutil
from pathlib import Path

class SimRenderer:
    @staticmethod
    def replay(sim, record = False, record_path = None):
        if record:
            record_folder = os.path.join(Path(record_path).parent, 'tmp')
            os.makedirs(record_folder, exist_ok = True)
            sim.viewer_options.record = True
            sim.viewer_options.record_folder = record_folder
            loop = sim.viewer_options.loop
            infinite = sim.viewer_options.infinite
            sim.viewer_options.loop = False
            sim.viewer_options.infinite = False
        
        sim.replay()

        if record:
            images_path = os.path.join(record_folder, r"%d.png")
            os.system("ffmpeg -i {} -vf palettegen palette.png".format(images_path))
            os.system("ffmpeg -i {} -i palette.png -lavfi paletteuse {}".format(images_path, record_path))
            os.remove("palette.png")

            shutil.rmtree(record_folder)

            sim.viewer_options.record = False
            sim.viewer_options.loop = loop
            sim.viewer_options.infinite = infinite
            
