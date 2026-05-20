"""
Aletheia evaluation test set.
8 contracts, each with one or more known issues planted in them,
covering 6 themes.
"""

TEST_CONTRACTS = [
    # ────────────────────────────────────────────────────
    # TC-001: Wrong withholding rate (theme: withholding tax + cross-border)
    # ────────────────────────────────────────────────────
    {
        "id": "TC-001",
        "topic": "Withholding Tax Rate",
        "description": "Wrong withholding tax rate for cross-border services",
        "contract": """
Service Agreement between NZ Tech Ltd (New Zealand resident) and 
Silicon Valley Corp (United States resident).

Clause 1: NZ Tech Ltd shall pay Silicon Valley Corp USD 100,000 
quarterly for cloud software services.

Clause 2: Withholding Tax. NZ Tech Ltd shall withhold 5% tax on all 
royalty payments to Silicon Valley Corp.

Clause 3: Effective from January 2026, indefinite term.
        """,
        "expected_issues": [
            {
                "type": "RISK",
                "severity_min": "MEDIUM",
                "keywords": ["withholding", "tax", "rate", "royalt"],
                "description": "Withholding tax rate may not align with NZ-US treaty rates"
            }
        ],
    },

    # ────────────────────────────────────────────────────
    # TC-002: Double taxation (theme: double taxation)
    # ────────────────────────────────────────────────────
    {
        "id": "TC-002",
        "topic": "Double Taxation",
        "description": "No double taxation relief mechanism specified",
        "contract": """
Consulting Agreement between Auckland Advisors Ltd (NZ) and 
Texas Holdings Inc (USA).

Clause 1: Auckland Advisors provides financial consulting at 
USD 200,000 annual fee.

Clause 2: Both parties shall pay their respective income taxes 
in their home jurisdiction.

Clause 3: No provisions for tax credits or treaty benefits.

Clause 4: Termination by either party with 30 days notice.
        """,
        "expected_issues": [
            {
                "type": "GAP",
                "severity_min": "HIGH",
                "keywords": ["double", "tax", "credit", "relief", "treaty"],
                "description": "Missing double taxation relief provisions under NZ-US treaty"
            }
        ],
    },

    # ────────────────────────────────────────────────────
    # TC-003: Dispute resolution (theme: dispute resolution clauses)
    # ────────────────────────────────────────────────────
    {
        "id": "TC-003",
        "topic": "Dispute Resolution",
        "description": "Dispute resolution clause may conflict with treaty",
        "contract": """
Software License Agreement between Wellington Software Ltd (NZ) and 
California Cloud Inc (USA, California).

Clause 1: License fee USD 50,000 per year for SaaS platform.

Clause 2: Withholding tax: 15% on all license fees.

Clause 3: Dispute Resolution. All disputes shall be exclusively 
resolved in California state courts under California law.

Clause 4: No mention of arbitration or treaty-based dispute resolution.
        """,
        "expected_issues": [
            {
                "type": "RISK",
                "severity_min": "MEDIUM",
                "keywords": ["dispute", "jurisdiction", "California", "court"],
                "description": "Exclusive California jurisdiction may conflict with treaty mutual agreement procedure"
            }
        ],
    },

    # ────────────────────────────────────────────────────
    # TC-004: Contract termination (theme: termination provisions)
    # ────────────────────────────────────────────────────
    {
        "id": "TC-004",
        "topic": "Contract Termination",
        "description": "Indefinite term with no termination conditions",
        "contract": """
Distribution Agreement between Christchurch Distributors Ltd (NZ) and 
Dallas Goods LLC (USA).

Clause 1: NZ company distributes Dallas products in Oceania region.

Clause 2: Commission: 10% on gross sales.

Clause 3: Withholding tax: NZ Co withholds 10% on payments.

Clause 4: Term. This agreement is effective immediately and continues 
indefinitely. No termination clause provided.
        """,
        "expected_issues": [
            {
                "type": "GAP",
                "severity_min": "MEDIUM",
                "keywords": ["termination", "term", "duration", "notice"],
                "description": "Missing termination provisions and notice requirements"
            }
        ],
    },

    # ────────────────────────────────────────────────────
    # TC-005: Currency / FX (theme: currency / exchange rate)
    # ────────────────────────────────────────────────────
    {
        "id": "TC-005",
        "topic": "Currency / Exchange Rate",
        "description": "No currency conversion or exchange rate provisions",
        "contract": """
Manufacturing Services Agreement between Hamilton Manufacturing (NZ) and 
Detroit Industries (USA).

Clause 1: NZ company manufactures parts at agreed unit price of NZD 500 each.

Clause 2: Payment shall be made in USD at the time of delivery.

Clause 3: Withholding tax: 10%.

Clause 4: No mention of exchange rate determination, currency conversion 
date, or fluctuation handling.
        """,
        "expected_issues": [
            {
                "type": "GAP",
                "severity_min": "MEDIUM",
                "keywords": ["currency", "exchange", "conversion", "rate", "NZD", "USD"],
                "description": "Missing currency conversion and exchange rate provisions"
            }
        ],
    },

    # ────────────────────────────────────────────────────
    # TC-006: Comprehensive contract with multiple issues
    # ────────────────────────────────────────────────────
    {
        "id": "TC-006",
        "topic": "Multiple Issues (Comprehensive)",
        "description": "Multiple compliance issues across topics",
        "contract": """
Royalty Agreement between Queenstown IP Ltd (NZ) and 
Boston Tech Corp (USA).

Clause 1: NZ company grants exclusive license to Boston Tech for 
patented technology. License fee: USD 1,000,000 upfront + 8% royalty 
on net revenue.

Clause 2: Withholding tax: 2% on royalties.

Clause 3: All payments in USD. No exchange rate specified.

Clause 4: Disputes resolved exclusively in Boston federal court.

Clause 5: Indefinite term. No termination conditions.

Clause 6: No tax credit or double taxation relief mechanism.
        """,
        "expected_issues": [
            {
                "type": "RISK",
                "severity_min": "HIGH",
                "keywords": ["withholding", "royalt", "rate"],
                "description": "2% withholding rate likely violates treaty rate for royalties"
            },
            {
                "type": "GAP",
                "severity_min": "MEDIUM",
                "keywords": ["double", "tax", "credit", "relief"],
                "description": "Missing double taxation relief"
            },
            {
                "type": "GAP",
                "severity_min": "MEDIUM",
                "keywords": ["termination", "term"],
                "description": "Missing termination provisions"
            },
        ],
    },

    # ────────────────────────────────────────────────────
    # TC-007: Clean contract (control group — should yield almost no issues)
    # ────────────────────────────────────────────────────
    {
        "id": "TC-007",
        "topic": "Compliant Contract (Control)",
        "description": "Well-drafted contract — system should find minimal issues",
        "contract": """
Professional Services Agreement between Auckland Legal Ltd (NZ) and 
New York Counsel LLP (USA).

Clause 1: NZ firm provides legal services at USD 300/hour, payable 
within 30 days of invoice, in USD.

Clause 2: Withholding tax: 10% on service fees, in accordance with 
NZ-US Tax Convention.

Clause 3: Each party pays own income tax in its jurisdiction. Tax 
credits applicable under NZ-US treaty.

Clause 4: Disputes first attempted to be resolved through good faith 
negotiation. If unresolved within 60 days, either party may invoke 
mutual agreement procedure under Article 24 of the NZ-US Convention.

Clause 5: Term: 2 years from effective date, renewable by written 
agreement. Either party may terminate with 60 days written notice.

Clause 6: Exchange rate for any NZD/USD conversion determined by 
Reserve Bank of NZ mid-rate on payment date.
        """,
        "expected_issues": [
            # Intentionally empty — this is the control group; the system
            # should not find major issues. Many HIGH-risk findings here
            # would indicate false positives.
        ],
    },

    # ────────────────────────────────────────────────────
    # TC-008: Edge case (tests robustness on very short input)
    # ────────────────────────────────────────────────────
    {
        "id": "TC-008",
        "topic": "Edge Case (Brief Contract)",
        "description": "Very short contract — tests if system fabricates issues",
        "contract": """
Agreement: NZ Co pays US Co USD 10,000 for services. 
Effective today.
        """,
        "expected_issues": [
            {
                "type": "GAP",
                "severity_min": "MEDIUM",
                "keywords": ["withholding", "tax"],
                "description": "Missing withholding tax provisions"
            },
            {
                "type": "GAP",
                "severity_min": "MEDIUM",
                "keywords": ["termination", "term"],
                "description": "Missing termination/duration provisions"
            },
        ],
    },
]

# Smoke-check the test suite
if __name__ == "__main__":
    print(f"✅ Test Suite Loaded")
    print(f"   Total test cases: {len(TEST_CONTRACTS)}")
    print(f"\n📋 Test Coverage:")
    for tc in TEST_CONTRACTS:
        n_issues = len(tc["expected_issues"])
        print(f"   {tc['id']} | {tc['topic']:<40} | Expected issues: {n_issues}")

    total_issues = sum(len(tc["expected_issues"]) for tc in TEST_CONTRACTS)
    print(f"\n📊 Total expected issues across all tests: {total_issues}")