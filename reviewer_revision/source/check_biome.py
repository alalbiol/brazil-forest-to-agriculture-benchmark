"""Check complete 440-m stable-agriculture chips against the IBGE 2019 biome map.

Download https://geoftp.ibge.gov.br/informacoes_ambientais/estudos_ambientais/biomas/vetores/Biomas_250mil.zip
to data/raw/ibge/Biomas_250mil.zip before running.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import geopandas as gpd
from pyproj import Transformer
from shapely.geometry import Point, box
from shapely.ops import transform

BENCH = Path(__file__).resolve().parent.parent

ARCHIVE = BENCH / "data/raw/ibge/Biomas_250mil.zip"
EXPECTED_SHA256 = "1d8a4d860d9c203b10ef41dd0701ced2e389ad86442e23a815a7cccdc98ca0c5"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path,
                        help="Extracted reproduction package; writes under rerun/")
    args = parser.parse_args()
    bundle = args.bundle.resolve() if args.bundle else None
    archive = bundle / "sources/Biomas_250mil.zip" if bundle else ARCHIVE
    reference = (bundle / "reviewer_revision/results/stable_agriculture_reference.csv"
                 if bundle else BENCH / "results/reviewer_revision/stable_agriculture_reference.csv")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError(f"Unexpected IBGE archive checksum: {digest}")
    polygons = gpd.read_file("zip://" + str(archive.resolve()))
    assert polygons.crs.to_epsg() == 4674
    amazon = polygons.loc[polygons.Bioma == "Amazônia", "geometry"].iloc[0]
    to_map_crs = Transformer.from_crs(4326, polygons.crs, always_xy=True).transform
    with reference.open() as stream:
        rows = list(csv.DictReader(stream))
    outside_centres, outside_chips = [], []
    for row in rows:
        lon, lat = float(row["lon"]), float(row["lat"])
        delta_lat = 440 / (2 * 111_320)
        delta_lon = 440 / (2 * 111_320 * math.cos(math.radians(lat)))
        if not amazon.covers(transform(to_map_crs, Point(lon, lat))):
            outside_centres.append(row["window_id"])
        footprint = box(lon - delta_lon, lat - delta_lat,
                        lon + delta_lon, lat + delta_lat)
        if not amazon.covers(transform(to_map_crs, footprint)):
            outside_chips.append(row["window_id"])
    result = {"source": "IBGE Biomas 1:250,000 (2019)", "archive_sha256": digest,
              "crs": "EPSG:4674", "n": len(rows),
              "outside_centres": outside_centres, "outside_440m_chips": outside_chips}
    target = bundle / "rerun/biome_check.json" if bundle else BENCH / "results/reviewer_revision/biome_check.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
