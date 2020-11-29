from pathlib import Path

from khumeia.data.item import SatelliteImage
from khumeia.roi.tile import LabelledTile, Tile
from khumeia.utils import io_utils

__all__ = ["ItemTileDumper", "ImageItemTileDumper", "NpArrayTileDumper"]


class ItemTileDumper:
    def __init__(self, item: SatelliteImage):
        self.item = item
        self.image = item.image

    def __call__(self, tile: Tile):
        return self.dump_tile(tile)

    def dump_tile(self, tile: Tile):
        raise NotImplementedError


class ImageItemTileDumper(ItemTileDumper):
    def __init__(self, item, output_dir: Path, save_format="jpg"):
        super(ImageItemTileDumper, self).__init__(item=item)
        self.output_dir = Path(output_dir)
        self.save_format = save_format

    def dump_tile(self, tile: LabelledTile):
        if tile.item_id == self.item.key:
            (self.output_dir / tile.label).mkdir(exist_ok=True)

            tile_data = tile.get_data(self.image)
            tile_basename = "{}_{}.{}".format(self.item.key, tile.key, self.save_format)
            io_utils.imsave(self.output_dir / tile.label / tile_basename, tile_data)

            return self.output_dir / tile.label / tile_basename, tile.label
        else:
            return None


class NpArrayTileDumper(ItemTileDumper):
    def dump_tile(self, tile: LabelledTile):
        if tile.item_id == self.item.key:
            tile_data = tile.get_data(self.image)
            tile_label = 0 if tile.label == "background" else 1
            return tile_data, tile_label
        else:
            return None
