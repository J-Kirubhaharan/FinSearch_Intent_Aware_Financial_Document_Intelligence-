"""
Role definitions and document access rules.
Each role has:
  - allowed_categories : which PDF categories can be retrieved
  - allowed_docs       : specific document names allowed (None = all in category)
  - blocked_patterns   : question patterns that are refused for this role
  - preset_questions   : example questions shown in the sidebar
  - description        : shown on the landing page
"""

ROLES = {
    "customer": {
        "label"      : "👤 Customer",
        "description": "View your account, transactions, and disputes. Ask about your rights and policies.",
        "color"      : "#4C72B0",
        "allowed_categories": [
            "Consumer_Protection",  # CFPB, FDIC, FTC guides
            "Synthetic_Policies",   # Bank's own customer-facing policies
            "Regulatory",           # Public consumer laws: EFTA, Reg E/Z, FCBA
            "Payment_Industry",     # Public-facing docs e.g. PayPal User Agreement
        ],
        "allowed_docs": None,  # All docs in above categories allowed;
                               # internal topics blocked via blocked_patterns below
        "blocked_patterns": [
            # Internal fraud/investigation procedures
            "fraud detection", "fraud escalation", "investigation procedure",
            "unauthorized transaction investigation",
            "account takeover", "ato response",
            # Internal chargeback operations
            "chargeback evidence", "chargeback management", "chargeback filing",
            # Internal compliance/AML
            "aml", "anti money laundering", "kyb", "know your business",
            # Internal staff/agent operations
            "staff procedure", "agent workflow", "internal policy",
            "system override", "provisional credit",
            "regulatory deadline tracking", "merchant onboarding",
        ],
        "preset_questions": [
            "Why was my payment declined?",
            "How do I dispute a transaction?",
            "What is the refund policy?",
            "What are my rights for an unauthorized charge?",
            "How long does a dispute take to resolve?",
        ],
    },

    "support_agent": {
        "label"      : "🎧 Support Agent",
        "description": "Handle customer queries, look up policies, and resolve complaints.",
        "color"      : "#55A868",
        "allowed_categories": [
            "Consumer_Protection",
            "Regulatory",
            "Payment_Industry",
            "Synthetic_Policies",
        ],
        "allowed_docs": None,  # Access to all documents
        "blocked_patterns": [
            "aml investigation", "anti money laundering process",
            "account takeover forensics", "fraud system bypass",
        ],
        "preset_questions": [
            "Does Regulation E or Regulation Z apply to debit card disputes?",
            "What is the chargeback filing deadline?",
            "What steps are required for a do_not_honor decline?",
            "How do I process a fee waiver for a customer?",
            "What documentation is needed to escalate a dispute?",
        ],
    },

    "fraud_investigator": {
        "label"      : "🔍 Fraud Investigator",
        "description": "Investigate fraud cases, check liability thresholds, and escalate cases.",
        "color"      : "#C44E52",
        "allowed_categories": [
            "Regulatory",
            "Synthetic_Policies",
        ],
        "allowed_docs": [
            "fraud_detection_escalation_guide",
            "unauthorized_transaction_investigation_policy",
            "anti_money_laundering_aml_policy",
            "account_takeover_ato_response_policy",
            "regulatory_deadline_tracking_policy",
            "chargeback_filing_evidence_policy",
            "provisional_credit_policy",
        ],
        "blocked_patterns": [
            "customer personal data", "credit card terms",
            "fee waiver", "loan policy", "merchant onboarding",
        ],
        "preset_questions": [
            "What is the liability threshold for unauthorized transactions?",
            "What evidence is required to file a fraud chargeback?",
            "What are the AML red flags I should look for?",
            "When must a fraud case be escalated?",
            "What is the investigation procedure for account takeover?",
        ],
    },

    "merchant": {
        "label"      : "🏪 Merchant",
        "description": "Understand payment rules, compliance requirements, and transaction failures.",
        "color"      : "#DD8452",
        "allowed_categories": [
            "Payment_Industry",
            "Synthetic_Policies",
        ],
        "allowed_docs": [
            "eu_psd2_sca_policy",
            "payment_failure_error_code_guide",
            "merchant_onboarding_kyb_policy",
            "chargeback_management_policy",
            "chargeback_filing_evidence_policy",
        ],
        "blocked_patterns": [
            "customer account", "personal transaction", "individual dispute",
            "fraud investigation", "aml procedure", "customer complaint",
            "unauthorized transaction investigation", "provisional credit",
        ],
        "preset_questions": [
            "Why are my European payments failing?",
            "What is PSD2 Strong Customer Authentication?",
            "What compliance rules apply to cross-border payments?",
            "What are the chargeback limits I should know?",
            "What does the do_not_honor error code mean?",
        ],
    },

    "dispute_resolver": {
        "label"      : "⚖️ Dispute Resolver",
        "description": "Manage open disputes, track SLA deadlines, and apply resolution procedures.",
        "color"      : "#8172B2",
        "allowed_categories": [
            "Regulatory",
            "Consumer_Protection",
            "Synthetic_Policies",
        ],
        "allowed_docs": [
            "dispute_resolution_procedure",
            "regulatory_deadline_tracking_policy",
            "provisional_credit_policy",
            "chargeback_management_policy",
            "chargeback_filing_evidence_policy",
            "customer_complaint_handling_procedure",
        ],
        "blocked_patterns": [
            "fraud detection process", "aml", "merchant onboarding",
            "account takeover", "loan policy", "credit card terms",
            "fee waiver",
        ],
        "preset_questions": [
            "What steps must be completed before the Regulation E day-10 deadline?",
            "When must provisional credit be issued?",
            "What documentation is required to resolve a chargeback?",
            "What is the resolution process for unauthorized charges?",
            "How do I handle an overdue dispute?",
        ],
    },
}

ROLE_KEYS = list(ROLES.keys())
