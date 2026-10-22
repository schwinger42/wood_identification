"""Parsing and loading of the Raman wood dataset.

Hierarchy: species -> tree -> block (cube) -> face (C/R/T) -> spot.
The statistical unit for any species-level claim is the TREE, not the spectrum.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "dataset" / "raw"

# Folder / filename spellings (typos kept on disk) -> canonical scientific name.
# "Chamaecyparis obyusa" is ASSUMED to be Japanese hinoki (C. obtusa); confirm with Prof. Tsao.
SPECIES_CANONICAL: dict[str, str] = {
    "Acacia confusa": "Acacia confusa",
    "Acacia mangium": "Acacia mangium",
    "Chamaecyparis formosensis": "Chamaecyparis formosensis",
    "Chamaecyparis obtusa var. formosana": "Chamaecyparis obtusa var. formosana",
    "Chamaecyparis obyusa": "Chamaecyparis obtusa",
    "Fokienia hodginsii": "Fokienia hodginsii",
    "Machilus kusanoi": "Machilus kusanoi",
    "Swietenia mahagoni": "Swietenia mahagoni",
    "Swietenia marophylla": "Swietenia macrophylla",
    "Swietenia marcrophylla": "Swietenia macrophylla",
    "Swietenia macrophylla": "Swietenia macrophylla",
}

# Three filename variants seen in the data:
#   24mW-<date>-<sp>-<tree>-<block>-<face>-<spot>.txt
#   24mW-<date>-<sp>-<block>-<face>-<spot>.txt
#   <date>-<sp>-<block>-<face>-<spot>.txt          (laser power not recorded)
FILENAME_RE = re.compile(
    r"^(?:(?P<power>\d+)mW-)?"
    r"(?P<date>\d{8})-"
    r"(?P<species>[A-Za-z. ]+?)-"
    r"(?:(?P<tree>\d+)-)?"
    r"(?P<block>\d+)-"
    r"(?P<face>[CRT])-"
    r"(?P<spot>\d+)\.txt$"
)


@dataclass(frozen=True)
class SpectrumMeta:
    path: str
    species: str
    tree: int
    block: int
    face: str
    spot: int
    date: str
    power_mw: int | None

    @property
    def tree_id(self) -> str:
        return f"{self.species}|T{self.tree}"

    @property
    def block_id(self) -> str:
        return f"{self.tree_id}|B{self.block}"


def parse_filename(name: str) -> SpectrumMeta:
    """Parse one spectrum filename (basename) into metadata.

    Species without a tree token are treated as tree 1 (one tree per species).
    """
    m = FILENAME_RE.match(name)
    if m is None:
        raise ValueError(f"Unrecognised filename: {name!r}")
    raw_species = m["species"].strip()
    if raw_species not in SPECIES_CANONICAL:
        raise ValueError(f"Unknown species token {raw_species!r} in {name!r}")
    return SpectrumMeta(
        path=name,
        species=SPECIES_CANONICAL[raw_species],
        tree=int(m["tree"]) if m["tree"] else 1,
        block=int(m["block"]),
        face=m["face"],
        spot=int(m["spot"]),
        date=m["date"],
        power_mw=int(m["power"]) if m["power"] else None,
    )


def read_spectrum(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read one `;`-separated export: (raman_shift_cm1, intensity). Third column is all zeros."""
    arr = np.loadtxt(path, delimiter=";", dtype=np.float64)
    return arr[:, 0], arr[:, 1]


def load_dataset(root: Path = RAW_DIR) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load every spectrum under `root`.

    Returns X (n_spectra, n_points), wavenumbers (n_points,), meta DataFrame (n_spectra rows)
    with columns species, tree, block, face, spot, date, power_mw, tree_id, block_id, session, path.
    Raises if the Raman-shift axis differs between files.
    """
    files = sorted(root.rglob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt spectra under {root}")
    rows: list[dict] = []
    spectra: list[np.ndarray] = []
    axis: np.ndarray | None = None
    for f in files:
        meta = parse_filename(f.name)
        wn, y = read_spectrum(f)
        if axis is None:
            axis = wn
        elif wn.shape != axis.shape or not np.allclose(wn, axis):
            raise ValueError(f"Raman-shift axis differs in {f}")
        spectra.append(y)
        rows.append(
            {
                "species": meta.species,
                "tree": meta.tree,
                "block": meta.block,
                "face": meta.face,
                "spot": meta.spot,
                "date": meta.date,
                "power_mw": meta.power_mw,
                "tree_id": meta.tree_id,
                "block_id": meta.block_id,
                "session": meta.date,
                "path": str(f.relative_to(root)),
            }
        )
    assert axis is not None
    return np.vstack(spectra), axis, pd.DataFrame(rows)
