"""Feature extractor: first 256 bytes of ELF .text."""

from __future__ import annotations

import numpy as np
from elftools.elf.elffile import ELFFile

from maldet.types import Sample


class Text256Extractor:
    output_shape = (256,)
    dtype = "uint8"

    def __init__(self, size: int = 256, pad_value: int = 0) -> None:
        self.size = size
        self.pad_value = pad_value

    def extract(self, sample: Sample) -> np.ndarray:
        with open(sample.path, "rb") as f:
            try:
                elf = ELFFile(f)
                section = elf.get_section_by_name(".text")
                if section is None:
                    raise ValueError(".text section missing")
                data = section.data()[: self.size]
            except Exception as exc:
                raise ValueError(f"ELF parse failed: {exc}") from exc
        if len(data) < self.size:
            data = data + bytes([self.pad_value] * (self.size - len(data)))
        return np.frombuffer(data, dtype=np.uint8).copy()
