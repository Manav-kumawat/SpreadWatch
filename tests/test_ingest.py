"""L1 ingest: header mapping, wide/long curve layouts, and refusal of unregistered files."""
import pandas as pd
import pytest

import ingest


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    for name, val in {"ROOT": tmp_path, "RAW_DIR": tmp_path / "data" / "raw",
                      "REAL_DIR": tmp_path / "data" / "real", "OUTPUT_DIR": tmp_path / "outputs",
                      "SOURCES_CSV": tmp_path / "sources.csv"}.items():
        monkeypatch.setattr(ingest, name, val)
    for sub in ("trades", "gsec", "ratings"):
        (tmp_path / "data" / "raw" / sub).mkdir(parents=True)
    pd.DataFrame({"Trade Date": ["03-07-2024"], "ISIN": ["INE000X01"], "Issuer Name": ["Abc Ltd"],
                  "Coupon Rate (%)": [8.5], "Maturity Date": ["31-03-2029"],
                  "Weighted Average Price": [99.5], "Weighted Average Yield": [8.63],
                  "Trade Value (Rs. Cr)": [25]}).to_csv(tmp_path / "data/raw/trades/nse.csv", index=False)
    pd.DataFrame({"Date": ["03-07-2024"], "3M": [6.8], "1Y": [6.9], "5Y": [7.0], "10Y": [7.1]}) \
        .to_csv(tmp_path / "data/raw/gsec/fbil.csv", index=False)
    pd.DataFrame({"issuer": ["Abc Ltd"], "agency": ["CRISIL"], "action_date": ["01-04-2023"],
                  "rating_to": ["CRISIL AA/Stable"]}).to_csv(tmp_path / "data/raw/ratings/r.csv", index=False)
    return tmp_path


def register(root, files):
    pd.DataFrame([{"source_id": f"S{i}", "file": f, "publisher": "p", "url": "u", "date_pulled": "2026-10-18",
                   "description": "d", "synthetic": "N"} for i, f in enumerate(files)]) \
        .to_csv(root / "sources.csv", index=False)


def test_unregistered_file_is_refused(sandbox):
    register(sandbox, ["data/raw/trades/nse.csv", "data/raw/gsec/fbil.csv"])   # ratings file not registered
    with pytest.raises(ingest.ProvenanceError, match="r.csv"):
        ingest.ingest_real()


def test_registered_files_are_mapped_to_canonical_columns(sandbox):
    register(sandbox, ["data/raw/trades/nse.csv", "data/raw/gsec/fbil.csv", "data/raw/ratings/r.csv"])
    log = ingest.ingest_real()
    assert len(log) == 3 and log["sha256"].str.len().eq(64).all()
    t = pd.read_csv(sandbox / "data/real/trades.csv")
    assert {"trade_date", "isin", "issuer", "coupon_pct", "maturity_date", "price", "printed_yield_pct",
            "value_cr", "source_id"} <= set(t.columns)
    assert t.loc[0, "trade_date"] == "2024-07-03"          # day-first date parsed correctly
    c = pd.read_csv(sandbox / "data/real/gsec_curve.csv")
    assert sorted(c["tenor_years"]) == [0.25, 1.0, 5.0, 10.0]


def test_tenor_labels():
    assert ingest.parse_tenor("3M") == 0.25
    assert ingest.parse_tenor("10 Yr") == 10.0
    assert ingest.parse_tenor("0.5") == 0.5
    with pytest.raises(ValueError):
        ingest.parse_tenor("Date")


def test_bond_master_fills_missing_terms_and_fixes_issuer_name(sandbox):
    # NSE-style trade file with no coupon/maturity and a different issuer label
    pd.DataFrame({"Trade Date": ["03-07-2024"], "ISIN": ["INE000X01"], "Description": ["ABC LTD NCD"],
                  "Price": [99.5], "Yield": [8.63]}).to_csv(sandbox / "data/raw/trades/nse.csv", index=False)
    (sandbox / "data/raw/master").mkdir()
    pd.DataFrame({"isin": ["INE000X01"], "issuer": ["Abc Ltd"], "coupon_pct": [8.5],
                  "maturity_date": ["31-03-2029"], "coupon_freq": ["Annual"]}) \
        .to_csv(sandbox / "data/raw/master/bonds.csv", index=False)
    register(sandbox, ["data/raw/trades/nse.csv", "data/raw/gsec/fbil.csv", "data/raw/ratings/r.csv",
                       "data/raw/master/bonds.csv"])
    ingest.ingest_real()
    t = pd.read_csv(sandbox / "data/real/trades.csv")
    assert t.loc[0, "issuer"] == "Abc Ltd"
    assert t.loc[0, "coupon_pct"] == 8.5 and t.loc[0, "maturity_date"] == "2029-03-31"
    assert t.loc[0, "coupon_freq"] == 1


def test_prepare_curve_combines_tenor_files(tmp_path):
    import prepare_curve
    for name, ys in [("India 1-Year Bond Yield Historical Data.csv", [6.5, 6.6]),
                     ("India 10-Year Bond Yield Historical Data.csv", [7.4, 7.5]),
                     ("India 3-Month Bond Yield Historical Data.csv", [6.2, 6.3])]:
        pd.DataFrame({"Date": ["03/29/2022", "03/30/2022"], "Price": ys, "Open": ys}).to_csv(tmp_path / name, index=False)
    w = prepare_curve.combine(tmp_path)
    assert list(w.columns) == ["date", "0.25", "1", "10"]
    assert str(w.loc[0, "date"]) == "2022-03-29"            # month-first date read correctly
    assert w.loc[1, "10"] == 7.5
    # and ingest understands the combined layout
    c = ingest.canonical_curve(w)
    assert sorted(c["tenor_years"].unique()) == [0.25, 1.0, 10.0]
