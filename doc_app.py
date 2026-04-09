import streamlit as st
import requests
import json
import re

# ── CONFIG ──
DEEPSEEK_BASE_URL = "https://genailab.tcs.in"
DEEPSEEK_MODEL = "azure_ai/genailab-maas-DeepSeek-V3-0324"

# ── SAMPLE DOCUMENTS ──
SAMPLES = {
    "invoice": (
        "invoice",
        """INVOICE

Invoice Number: INV-2024-1087
Invoice Date: November 15, 2024
Due Date: December 15, 2024
Payment Terms: NET 30

BILL FROM:
Apex Solutions Ltd.
123 Business Park, Suite 400
Mumbai, Maharashtra 400001
GST: 27AABCA1234F1Z5
Contact: accounts@apexsolutions.in

BILL TO:
TechCore Enterprises Pvt. Ltd.
56 Cyber Tower, Andheri East
Mumbai, Maharashtra 400093
Attention: Mr. Rajiv Sharma
PO Reference: PO-2024-0542

LINE ITEMS:
1. Cloud Infrastructure Setup & Configuration
   Qty: 1 | Unit Price: ₹2,50,000 | Amount: ₹2,50,000

2. Software License (Annual) - DocuTrack Pro
   Qty: 5 seats | Unit Price: ₹18,000 | Amount: ₹90,000

3. Data Migration Services (Phase 1)
   Qty: 40 hrs | Unit Price: ₹3,500/hr | Amount: ₹1,40,000

4. Technical Support SLA - Gold Tier (3 months)
   Qty: 1 | Unit Price: ₹45,000 | Amount: ₹45,000

Subtotal: ₹5,25,000
GST (18%): ₹94,500
TOTAL DUE: ₹6,19,500

PAYMENT INSTRUCTIONS:
Bank: HDFC Bank
Account Name: Apex Solutions Ltd.
Account No: 50100123456789
IFSC: HDFC0001234

Late payment penalty: 1.5% per month after due date.
For queries contact: billing@apexsolutions.in | +91 98765 43210""",
    ),
    "contract": (
        "contract",
        """SERVICE AGREEMENT

Agreement Number: SA-2024-0389
Effective Date: October 1, 2024
Expiry Date: September 30, 2025

PARTIES:
Service Provider: DataBridge Technologies Pvt. Ltd.
Client: NexGen Retail Solutions Ltd.

SCOPE OF SERVICES:
1. Cloud data integration and ETL pipeline management
2. Monthly reporting and analytics dashboards
3. 24x7 technical support (Gold SLA)

FINANCIALS:
Monthly Fee: ₹3,50,000 + GST
Annual Value: ₹42,00,000 + GST
Payment: Due by 5th of each month
Late penalty: 2% per month

GOVERNING LAW: This agreement shall be governed by the laws of India.
Disputes resolved via arbitration in Mumbai.

IMPORTANT CLAUSES:
- Force majeure applies for acts of God, government actions, pandemics
- Liability cap: 3 months of service fees (₹10,50,000)
- Indemnification clause applies for data breaches caused by Service Provider negligence""",
    ),
    "report": (
        "report",
        """INTERNAL QUARTERLY REPORT — Q3 FY2024
Prepared by: Finance & Operations Division
Date: October 10, 2024
Classification: CONFIDENTIAL

EXECUTIVE SUMMARY:
Q3 FY2024 showed mixed performance. Revenue reached ₹42.7 Cr against a target of ₹40 Cr (6.75% above target).
Operational costs rose 12% due to workforce expansion and infrastructure investments.
Net profit margin declined to 14.3% from 18.1% in Q2.

FINANCIAL HIGHLIGHTS:
- Total Revenue: ₹42.7 Cr (↑ 8% YoY)
- Operating Costs: ₹28.3 Cr (↑ 12% YoY)
- EBITDA: ₹14.4 Cr (↓ 3% QoQ)
- Net Profit: ₹6.1 Cr (margin: 14.3%)
- Cash & Equivalents: ₹18.2 Cr
- Outstanding Receivables: ₹9.4 Cr (60+ days: ₹3.1 Cr — flagged)

KEY RISKS:
1. Outstanding receivables from 3 clients (₹3.1 Cr, >60 days)
2. Vendor concentration risk: 40% procurement from single vendor
3. GST filing deadline: November 20, 2024
4. Attrition: 14% in Q3

ACTION ITEMS FOR Q4:
- Collect overdue receivables by November 30, 2024
- Diversify vendor base by December 31, 2024
- Complete annual audit preparation by November 15, 2024""",
    ),
    "po": (
        "purchase_order",
        """PURCHASE ORDER

PO Number: PO-2024-0891
PO Date: November 5, 2024
Expected Delivery: December 1, 2024

BUYER:
Meridian Manufacturing Pvt. Ltd.
Industrial Area Phase II, Chandigarh - 160002
Authorized By: Mr. Deepak Verma, Procurement Head

VENDOR:
SupplyMax India Ltd.
Sector 62, Noida - 201309
Vendor Code: VM-2234

ORDERED ITEMS:
1. CNC Precision Bearings (Grade A) - 500 units @ ₹850 = ₹4,25,000
2. Stainless Steel Fasteners M8x40 - 2000 units @ ₹45 = ₹90,000
3. Industrial Lubricant (20L drums) - 25 drums @ ₹2,200 = ₹55,000
4. Safety Gloves (Large, Grade II) - 200 pairs @ ₹320 = ₹64,000

Subtotal: ₹6,34,000
GST (12%): ₹76,080
GRAND TOTAL: ₹7,10,080

DELIVERY TERMS: DAP Chandigarh facility
PAYMENT TERMS: 50% advance on PO confirmation; 50% within 15 days of delivery acceptance.
PENALTIES: Delivery delay beyond 3 days: 0.5% penalty per day. Max: 5% of PO value.""",
    ),
}

# ── API CALL ──
def call_deepseek(api_key: str, system_prompt: str, user_prompt: str) -> str:
    url = f"{DEEPSEEK_BASE_URL}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": 4096,
        "temperature": 0.2,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    data = resp.json()
    if "error" in data:
        raise ValueError(data["error"].get("message", str(data["error"])))
    return data["choices"][0]["message"]["content"]


def parse_json(raw: str) -> dict:
    clean = re.sub(r"```json|```", "", raw).strip()
    start = clean.find("{")
    end = clean.rfind("}")
    if start != -1 and end != -1:
        clean = clean[start : end + 1]
    return json.loads(clean)


# ── STREAMLIT APP ──
st.set_page_config(page_title="DocSense — Back Office Summarization Agent", layout="wide", page_icon="📋")

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .stTabs [data-baseweb="tab"] { font-size: 13px; }
    .metric-card {
        background: #0f0f1a; border: 1px solid #1e1e2e;
        border-radius: 8px; padding: 14px; text-align: center;
    }
    .risk-high { border-left: 4px solid #ef4444; background: rgba(239,68,68,0.06); padding: 12px; border-radius: 6px; margin-bottom: 8px; }
    .risk-med  { border-left: 4px solid #f59e0b; background: rgba(245,158,11,0.06); padding: 12px; border-radius: 6px; margin-bottom: 8px; }
    .risk-low  { border-left: 4px solid #10b981; background: rgba(16,185,129,0.06); padding: 12px; border-radius: 6px; margin-bottom: 8px; }
</style>
""", unsafe_allow_html=True)

# ── HEADER ──
col_logo, col_title = st.columns([1, 11])
with col_logo:
    st.markdown("## 📋")
with col_title:
    st.markdown("## DocSense")
    st.caption("Back Office Summarization Agent · DeepSeek-V3-0324 via TCS GenAI Lab")

st.divider()

# ── SIDEBAR ──
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    api_key = st.text_input("TCS GenAI Lab API Key", type="password", placeholder="sk-...", value="sk-Cx_LohcL6L6cavC7c5mriw")

    doc_type = st.selectbox("Document Type", [
        "auto", "invoice", "contract", "report", "purchase_order", "policy", "memo"
    ], format_func=lambda x: {
        "auto": "Auto-detect", "invoice": "Invoice", "contract": "Contract / Agreement",
        "report": "Internal Report", "purchase_order": "Purchase Order",
        "policy": "Policy Document", "memo": "Memo / Correspondence"
    }[x])

    st.markdown("### 🎯 Extraction Focus")
    focus_options = ["💰 Amounts", "📅 Dates", "⚖️ Obligations", "⚠️ Risks", "👥 Parties", "📌 Action Items"]
    focus = st.multiselect("Select focus areas", focus_options, default=["💰 Amounts", "📅 Dates", "⚖️ Obligations", "⚠️ Risks"])

    st.markdown("### 📂 Load Sample Document")
    sample_cols = st.columns(2)
    with sample_cols[0]:
        if st.button("🧾 Invoice"):     st.session_state["sample"] = "invoice"
        if st.button("📊 Report"):      st.session_state["sample"] = "report"
    with sample_cols[1]:
        if st.button("📝 Contract"):    st.session_state["sample"] = "contract"
        if st.button("🛒 Purchase Order"): st.session_state["sample"] = "po"

# ── MAIN INPUT ──
uploaded_file = st.file_uploader("Upload Document", type=["txt", "md", "csv"])

# Load sample if triggered
if "sample" in st.session_state:
    key = st.session_state.pop("sample")
    dtype, sample_text = SAMPLES[key]
    st.session_state["doc_text"] = sample_text
    st.session_state["doc_type_override"] = dtype

default_text = st.session_state.get("doc_text", "")
doc_text_area = st.text_area(
    "Or paste document text here",
    value=default_text,
    height=180,
    placeholder="Paste invoice, contract, or report text here…",
)

# File upload overrides text area
if uploaded_file:
    doc_text_area = uploaded_file.read().decode("utf-8", errors="ignore")[:12000]
    st.success(f"📎 Loaded: {uploaded_file.name}")

run = st.button("🔍 Analyze & Summarize", type="primary", use_container_width=True)

# ── RUN ──
if run:
    if not api_key:
        st.error("Please enter your TCS GenAI Lab API key.")
        st.stop()
    if not doc_text_area.strip():
        st.error("Please paste or upload a document first.")
        st.stop()

    safe_doc = doc_text_area[:8000]
    effective_type = st.session_state.pop("doc_type_override", doc_type)

    results = {}
    progress = st.progress(0, text="Starting analysis…")

    # Step 1 — Summary
    progress.progress(10, text="[1/4] Classifying document and generating executive summary…")
    try:
        raw1 = call_deepseek(
            api_key,
            "You are an expert back-office document analyst. Analyze business documents (invoices, contracts, reports, purchase orders) and extract structured intelligence. Always respond with valid JSON only, no markdown.",
            f"""Analyze this {'business' if effective_type == 'auto' else effective_type} document and return ONLY valid JSON:

DOCUMENT:
{safe_doc}

Return exactly:
{{
  "doc_type": "<Invoice|Contract|Report|Purchase Order|Policy|Memo|Other>",
  "doc_title": "<inferred title or reference number>",
  "doc_date": "<primary date found or 'Not specified'>",
  "doc_reference": "<invoice/PO/agreement number or 'N/A'>",
  "overview": "<2-3 sentence executive summary>",
  "purpose": "<the business purpose of this document>",
  "total_amount": "<primary monetary total if any, else 'N/A'>",
  "currency": "<currency symbol/code>",
  "urgency": "<High|Medium|Low>",
  "urgency_reason": "<why this urgency level>",
  "due_date": "<most critical deadline or 'N/A'>",
  "key_metric_1_label": "<e.g. Total Value>",
  "key_metric_1_value": "<value>",
  "key_metric_2_label": "<e.g. Payment Terms>",
  "key_metric_2_value": "<value>",
  "key_metric_3_label": "<e.g. Parties Involved>",
  "key_metric_3_value": "<value>"
}}"""
        )
        results["summary"] = parse_json(raw1)
        progress.progress(30, text=f"[1/4] ✓ Classified as {results['summary'].get('doc_type', '?')}")
    except Exception as e:
        st.error(f"Step 1 failed: {e}")
        st.stop()

    # Step 2 — Key Points
    progress.progress(35, text="[2/4] Extracting key points, amounts, dates, and obligations…")
    try:
        raw2 = call_deepseek(
            api_key,
            "You are a back-office intelligence agent. Extract and categorize critical information from business documents. Return ONLY valid JSON, no markdown.",
            f"""Extract key points from this {results['summary'].get('doc_type','business')} document. Focus on: {', '.join(focus)}.

DOCUMENT:
{safe_doc}

Return exactly:
{{
  "key_points": [
    {{
      "text": "<the key information point>",
      "category": "<Amounts|Dates|Obligations|Risks|Action Items|Parties|Info>",
      "icon": "<single emoji>",
      "priority": "<High|Medium|Low>"
    }}
  ]
}}

Extract 8-12 key points covering the most actionable and critical information."""
        )
        results["key_points"] = parse_json(raw2).get("key_points", [])
        progress.progress(55, text=f"[2/4] ✓ {len(results['key_points'])} key points extracted")
    except Exception as e:
        st.warning(f"Step 2 failed: {e}")
        results["key_points"] = []

    # Step 3 — Entities & Timeline
    progress.progress(60, text="[3/4] Mapping entities and building date timeline…")
    try:
        raw3 = call_deepseek(
            api_key,
            "You are a document intelligence agent specializing in named entity recognition and timeline extraction for business documents. Return ONLY valid JSON, no markdown.",
            f"""Extract entities and timeline from this document:

DOCUMENT:
{safe_doc}

Return exactly:
{{
  "parties": [], "organizations": [], "amounts": [],
  "references": [], "locations": [], "contacts": [],
  "timeline": [
    {{
      "date": "<date string>",
      "event": "<what happens>",
      "detail": "<additional context>",
      "icon": "<single emoji>",
      "type": "<deadline|milestone|payment|start|end|report|other>"
    }}
  ]
}}

Include all dates. Sort timeline chronologically."""
        )
        results["entities"] = parse_json(raw3)
        tl = results["entities"].get("timeline", [])
        pts = len(results["entities"].get("parties", []))
        progress.progress(78, text=f"[3/4] ✓ {len(tl)} dates mapped, {pts} parties identified")
    except Exception as e:
        st.warning(f"Step 3 failed: {e}")
        results["entities"] = {}

    # Step 4 — Risks
    progress.progress(80, text="[4/4] Running risk and compliance analysis…")
    try:
        raw4 = call_deepseek(
            api_key,
            "You are a risk and compliance analyst specializing in business document review. Return ONLY valid JSON, no markdown.",
            f"""Perform risk analysis on this {results['summary'].get('doc_type','business')} document:

DOCUMENT:
{safe_doc}

Return exactly:
{{
  "risks": [
    {{
      "title": "<risk title>",
      "description": "<what the risk is>",
      "level": "<High|Medium|Low>",
      "icon": "<single emoji>",
      "mitigation": "<recommended action>"
    }}
  ],
  "action_items": ["<action 1>", "<action 2>"],
  "compliance_notes": "<regulatory considerations>"
}}

Identify 4-8 risks. Include penalty clauses, deadline risks, obligation gaps."""
        )
        results["risks"] = parse_json(raw4)
        risk_count = len(results["risks"].get("risks", []))
        progress.progress(100, text=f"[4/4] ✓ {risk_count} risks identified — Analysis complete!")
    except Exception as e:
        st.warning(f"Step 4 failed: {e}")
        results["risks"] = {}

    st.session_state["results"] = results

# ── DISPLAY RESULTS ──
if "results" in st.session_state:
    results = st.session_state["results"]
    s = results.get("summary", {})

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📋 Summary", "🔑 Key Points", "🏷️ Entities", "📅 Timeline", "⚠️ Risks", "📝 Full Extract"
    ])

    # ── TAB 1: SUMMARY ──
    with tab1:
        urgency_color = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(s.get("urgency", "Low"), "🟢")

        st.markdown(f"### {s.get('doc_type','—')} · {s.get('doc_date','—')}")
        st.markdown(f"**{s.get('doc_title','Untitled Document')}**")
        st.markdown(s.get("overview", ""))
        st.caption(s.get("purpose", ""))

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric(s.get("key_metric_1_label", "Metric 1"), s.get("key_metric_1_value", "—"))
        with m2:
            st.metric(s.get("key_metric_2_label", "Metric 2"), s.get("key_metric_2_value", "—"))
        with m3:
            st.metric("Urgency", f"{urgency_color} {s.get('urgency','—')}")

        st.divider()
        details = {
            "Reference": s.get("doc_reference", "N/A"),
            "Due Date": s.get("due_date", "N/A"),
            "Total Amount": f"{s.get('currency','')} {s.get('total_amount','N/A')}" if s.get("total_amount") != "N/A" else "N/A",
            "Urgency Reason": s.get("urgency_reason", "—"),
            s.get("key_metric_3_label", "Metric 3"): s.get("key_metric_3_value", "—"),
            "Currency": s.get("currency", "—"),
        }
        cols = st.columns(3)
        for i, (k, v) in enumerate(details.items()):
            with cols[i % 3]:
                st.markdown(f"**{k}**")
                st.markdown(v)

    # ── TAB 2: KEY POINTS ──
    with tab2:
        kps = results.get("key_points", [])
        if not kps:
            st.info("No key points extracted.")
        else:
            cat_colors = {
                "Amounts": "🟢", "Dates": "🔵", "Obligations": "🟡",
                "Risks": "🔴", "Action Items": "🟣", "Parties": "⚪", "Info": "⚪"
            }
            for kp in kps:
                icon = kp.get("icon", "📌")
                cat = kp.get("category", "Info")
                dot = cat_colors.get(cat, "⚪")
                st.markdown(f"{icon} {kp.get('text','')} &nbsp; `{dot} {cat}`")
                st.divider()

    # ── TAB 3: ENTITIES ──
    with tab3:
        ent = results.get("entities", {})
        if not ent:
            st.info("No entity data.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**🏢 Parties & Organizations**")
                for item in [*ent.get("parties", []), *ent.get("organizations", [])]:
                    st.markdown(f"· {item}")
                st.markdown("**💰 Financial Amounts**")
                for item in ent.get("amounts", []):
                    st.markdown(f"· {item}")
            with c2:
                st.markdown("**🔖 References & IDs**")
                for item in ent.get("references", []):
                    st.markdown(f"· {item}")
                st.markdown("**📍 Locations & Contacts**")
                for item in [*ent.get("locations", []), *ent.get("contacts", [])]:
                    st.markdown(f"· {item}")

    # ── TAB 4: TIMELINE ──
    with tab4:
        timeline = results.get("entities", {}).get("timeline", [])
        if not timeline:
            st.info("No dates found in document.")
        else:
            for event in timeline:
                st.markdown(f"{event.get('icon','📅')} **{event.get('date','—')}** — {event.get('event','')}")
                if event.get("detail"):
                    st.caption(event["detail"])
                st.divider()

    # ── TAB 5: RISKS ──
    with tab5:
        risk_data = results.get("risks", {})
        risks = risk_data.get("risks", [])
        if not risks:
            st.info("No risks identified.")
        else:
            level_class = {"High": "risk-high", "Medium": "risk-med", "Low": "risk-low"}
            for r in risks:
                lvl = r.get("level", "Low")
                css = level_class.get(lvl, "risk-low")
                badge = {"High": "🔴 High", "Medium": "🟡 Medium", "Low": "🟢 Low"}.get(lvl, lvl)
                st.markdown(
                    f'<div class="{css}">'
                    f'{r.get("icon","⚠️")} <strong>{r.get("title","")}</strong> &nbsp; <code>{badge}</code><br>'
                    f'<span style="font-size:13px">{r.get("description","")}</span>'
                    + (f'<br><span style="color:#14b8a6">→ {r.get("mitigation","")}</span>' if r.get("mitigation") else "")
                    + "</div>",
                    unsafe_allow_html=True,
                )

        actions = risk_data.get("action_items", [])
        if actions:
            st.markdown("#### 📌 Recommended Actions")
            for a in actions:
                st.markdown(f"→ {a}")

        notes = risk_data.get("compliance_notes", "")
        if notes:
            st.markdown("#### 📜 Compliance Notes")
            st.info(notes)

    # ── TAB 6: RAW JSON ──
    with tab6:
        full_extract = {
            "summary": results.get("summary"),
            "key_points": results.get("key_points"),
            "entities": results.get("entities"),
            "risks": results.get("risks"),
        }
        st.download_button(
            "⬇️ Download JSON",
            data=json.dumps(full_extract, indent=2, ensure_ascii=False),
            file_name="docsense_extract.json",
            mime="application/json",
        )
        st.code(json.dumps(full_extract, indent=2, ensure_ascii=False), language="json")
