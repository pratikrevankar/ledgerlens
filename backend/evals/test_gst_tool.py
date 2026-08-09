"""Unit tests for the deterministic GST calculator — run with `pytest`, no infra.

These are the 'the math must never drift' guardrails. The agent's numbers come
from this function, not the LLM, so they're pinned here.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.tools import GstInput, compute_gst  # noqa: E402


def test_interstate_is_igst():
    r = compute_gst(GstInput(taxable_value=50000, rate_percent=18,
                             supplier_state="KA", place_of_supply_state="MH", supply_type="services"))
    assert r.interstate is True
    assert r.igst == 9000.0
    assert r.cgst == 0.0 and r.sgst == 0.0
    assert r.total_tax == 9000.0
    assert r.invoice_total == 59000.0


def test_intrastate_splits_cgst_sgst():
    r = compute_gst(GstInput(taxable_value=10000, rate_percent=18,
                             supplier_state="MH", place_of_supply_state="MH", supply_type="goods"))
    assert r.interstate is False
    assert r.igst == 0.0
    assert r.cgst == 900.0 and r.sgst == 900.0
    assert r.total_tax == 1800.0


def test_cgst_sgst_never_lose_a_paisa_on_odd_tax():
    # 5% of 100.10 = 5.005 → 5.01 tax; split must still sum exactly to tax.
    r = compute_gst(GstInput(taxable_value=100.10, rate_percent=5,
                             supplier_state="DL", place_of_supply_state="DL"))
    assert round(r.cgst + r.sgst, 2) == r.total_tax


def test_state_codes_are_case_insensitive():
    a = compute_gst(GstInput(taxable_value=1000, rate_percent=12, supplier_state="ka", place_of_supply_state="KA"))
    assert a.interstate is False
