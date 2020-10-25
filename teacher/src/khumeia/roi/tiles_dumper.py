import os

from khumeia.data.item import SatelliteImage
from khumeia.roi.tile import Tile, LabelledTile
from khumeia.utils import io_utils


class ItemTileDumper:
    def __init__(self, item: SatelliteImage):
        self.item = item
        self.image = item.image

    def __call__(self, tile: Tile):
        return self.dump_tiles_for_item(tile)

    def dump_tiles_for_item(self, tile: Tile):
        raise NotImplementedError


class ImageItemTileDumper(ItemTileDumper):
    def __init__(self, item, output_dir, save_format="jpg"):
        super(ImageItemTileDumper, self).__init__(item=item)
        self.output_dir = output_dir
        self.save_format = save_format

    def dump_tiles_for_item(self, tile: LabelledTile):
        if tile.item_id == self.item.key:
            os.makedirs(os.path.join(self.output_dir, tile.label), exist_ok=True)

            tile_data = tile.get_data(self.image)
            tile_basename = "{}_{}.{}".format(self.item.key, tile.key, self.save_format)
            io_utils.imsave(os.path.join(self.output_dir, tile.label, tile_basename), tile_data)

            return os.path.join(self.output_dir, tile.label, tile_basename), tile.label
        else:
            return None


class NpArrayTileDumper(ItemTileDumper):
    def dump_tiles_for_item(self, tile: LabelledTile):
        if tile.item_id == self.item.key:
            tile_data = tile.get_data(self.image)
            tile_label = 0 if tile.label == "background" else 1
            return tile_data, tile_label
        else:
            return None
