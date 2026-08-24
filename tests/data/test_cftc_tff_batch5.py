from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path

from ftmoquant.data.cftc_tff_batch5 import LINEAGE_ID, normalize
from ftmoquant.research.alpha_lab.batch5_cftc_availability import (
    EXPECTED_AMENDMENT_SEMANTIC_SHA256,
)
from ftmoquant.research.alpha_lab.batch5_preregistration import (
    EXPECTED_PREREGISTRATION_SEMANTIC_SHA256,
)


def test_normalization_verifies_dual_hashes_and_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "cftc"
    raw = root / "raw"
    raw.mkdir(parents=True)
    archive = raw / "fut_fin_txt_2022.zip"
    source_fields = [
        "Market_and_Exchange_Names",
        "Report_Date_as_YYYY-MM-DD",
        "CFTC_Contract_Market_Code",
        "Open_Interest_All",
        "Dealer_Positions_Long_All",
        "Dealer_Positions_Short_All",
    ]
    # Build with DictWriter to preserve the comma in the fixture name safely.
    import io

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=source_fields)
    writer.writeheader()
    for code in ("090741", "092741", "096742", "097741", "099741", "112741", "232741"):
        writer.writerow(
            {
                "Market_and_Exchange_Names": "TEST - CME",
                "Report_Date_as_YYYY-MM-DD": "2022-11-22",
                "CFTC_Contract_Market_Code": code,
                "Open_Interest_All": "100",
                "Dealer_Positions_Long_All": "30",
                "Dealer_Positions_Short_All": "20",
            }
        )
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("FinFut22.txt", buffer.getvalue())
    import hashlib

    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    (root / "cftc_tff_raw_manifest.json").write_text(
        json.dumps(
            {
                "lineage_id": LINEAGE_ID,
                "availability_amendment_semantic_sha256": (
                    EXPECTED_AMENDMENT_SEMANTIC_SHA256
                ),
                "files": [
                    {"year": 2022, "path": "raw/fut_fin_txt_2022.zip", "sha256": digest}
                ],
            }
        ),
        encoding="utf-8",
    )
    target, readiness = normalize(root)
    rows = list(csv.DictReader(target.open(encoding="utf-8")))
    assert len(rows) == 7
    assert {row["availability_status"] for row in rows} == {"UNRESOLVED"}
    assert {row["availability_timestamp"] for row in rows} == {""}
    assert {row["original_preregistration_semantic_sha256"] for row in rows} == {
        EXPECTED_PREREGISTRATION_SEMANTIC_SHA256
    }
    assert json.loads(readiness.read_text())["research_ready"] is False
