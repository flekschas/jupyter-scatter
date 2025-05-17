import json
import pathlib
from typing import Literal

import numpy as np
import pandas as pd


class Font:
    """
    A class for font information
    """

    face: str
    style: Literal['normal', 'italic']
    weight: Literal['normal', 'bold']
    line_height: float
    num_glyphs: int
    kernings: np.ndarray
    xadvances: np.ndarray

    def __init__(
        self,
        spec_file_path: str,
        face: str,
        weight: Literal['normal', 'bold'] = 'normal',
        style: Literal['normal', 'italic'] = 'normal',
    ):
        with open(pathlib.Path(__file__).parent / spec_file_path, 'r') as f:
            spec = json.load(f)

        scale = 1.0 / spec['info']['size']

        self.face = face
        self.style = style
        self.weight = weight

        self.line_height = scale * spec['common']['lineHeight']

        self.glyphs = pd.DataFrame(
            data=spec['chars'],
            columns=['id', 'width', 'height', 'xoffset', 'yoffset', 'xadvance'],
        )
        self.glyphs.set_index('id', inplace=True)

        self.glyphs.xoffset *= scale
        self.glyphs.yoffset *= scale
        self.glyphs.width *= scale
        self.glyphs.height *= scale
        self.glyphs.xadvance *= scale

        self.min_glyph_id = min(glyph['id'] for glyph in spec['chars'])
        self.max_glyph_id = max(glyph['id'] for glyph in spec['chars'])
        self.glyph_id_extent = self.max_glyph_id - self.min_glyph_id

        self.kernings = np.zeros(self.glyph_id_extent**2)

        for kerning in spec['kernings']:
            first_idx = kerning['first'] - self.min_glyph_id
            second_idx = kerning['second'] - self.min_glyph_id
            kerning_idx = first_idx * self.glyph_id_extent + second_idx
            self.kernings[kerning_idx] = scale * kerning['amount']

        self.xadvances = np.zeros(self.max_glyph_id + 1)

        for glyph in spec['chars']:
            self.xadvances[glyph['id']] = scale * glyph['xadvance']

    def kerning(self, first_glyph_id: int, second_glyph_id: int):
        first_idx = first_glyph_id - self.min_glyph_id
        second_idx = second_glyph_id - self.min_glyph_id
        kerning_idx = first_idx * self.glyph_id_extent + second_idx
        return self.kernings[kerning_idx]
