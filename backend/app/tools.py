"""GST computation tool.

A clean-room Python reimplementation of the GST split logic (NOT imported from
any production codebase). Deterministic and unit-tested, so the agent's numbers
are trustworthy — the LLM decides WHEN to compute and with what inputs; the math
itself never touches the model.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class GstInput(BaseModel):
    """Inputs for a GST computation on a single supply."""
    taxable_value: float = Field(..., description="Value of the supply in INR, before GST")
    rate_percent: float = Field(18.0, description="Applicable GST rate, e.g. 18 for 18%")
    supplier_state: str = Field(..., description="Supplier's state (name or 2-letter code)")
    place_of_supply_state: str = Field(..., description="Place-of-supply state (name or code)")
    supply_type: Literal["goods", "services"] = Field("goods")


class GstResult(BaseModel):
    taxable_value: float
    rate_percent: float
    interstate: bool
    igst: float
    cgst: float
    sgst: float
    total_tax: float
    invoice_total: float
    explanation: str


def _round2(x: float) -> float:
    return round(x + 1e-9, 2)


def compute_gst(inp: GstInput) -> GstResult:
    interstate = inp.supplier_state.strip().upper() != inp.place_of_supply_state.strip().upper()
    tax = _round2(inp.taxable_value * inp.rate_percent / 100.0)

    if interstate:
        igst, cgst, sgst = tax, 0.0, 0.0
        head = f"Inter-State supply (IGST Act s.7) → IGST @ {inp.rate_percent:g}%."
    else:
        half = _round2(tax / 2.0)
        cgst = sgst = half
        igst = 0.0
        # keep CGST+SGST == tax even when tax is odd in the last paisa
        cgst = _round2(tax - sgst)
        head = f"Intra-State supply (IGST Act s.8) → CGST @ {inp.rate_percent/2:g}% + SGST @ {inp.rate_percent/2:g}%."

    total_tax = _round2(igst + cgst + sgst)
    pos_rule = "s.10 (goods)" if inp.supply_type == "goods" else "s.12 (services)"
    explanation = (
        f"{head} Taxable value Rs {inp.taxable_value:,.2f} (CGST Act s.15); place of supply per IGST Act {pos_rule}. "
        f"Tax = Rs {total_tax:,.2f}; invoice total Rs {_round2(inp.taxable_value + total_tax):,.2f}."
    )
    return GstResult(
        taxable_value=inp.taxable_value, rate_percent=inp.rate_percent, interstate=interstate,
        igst=igst, cgst=cgst, sgst=sgst, total_tax=total_tax,
        invoice_total=_round2(inp.taxable_value + total_tax), explanation=explanation,
    )


# LangChain tool wrapper — lets the LLM invoke the calc via function-calling.
def make_gst_tool():
    from langchain_core.tools import StructuredTool

    def _run(taxable_value: float, supplier_state: str, place_of_supply_state: str,
             rate_percent: float = 18.0, supply_type: str = "goods") -> dict:
        res = compute_gst(GstInput(
            taxable_value=taxable_value, rate_percent=rate_percent,
            supplier_state=supplier_state, place_of_supply_state=place_of_supply_state,
            supply_type=supply_type,  # type: ignore[arg-type]
        ))
        return res.model_dump()

    return StructuredTool.from_function(
        func=_run,
        name="compute_gst",
        description="Compute the GST (IGST or CGST+SGST split) on a supply given its taxable value, "
                    "rate, supplier state and place-of-supply state. Use for any 'how much GST' question.",
        args_schema=GstInput,
    )
