# Curated use cases for the openjev playground.
# Shaped for Laya's question grammar:
#   choice: criteria = {option: description}
#   score:  criteria = [level0, level1, ...]  (ordinal)
#   noul:   instructions only (calibrated yes/no)
USECASES = [
    {
        "id": "support-triage",
        "title": "Support ticket triage",
        "icon": "\U0001F4E7",
        "summary": "Route to department, rate urgency, detect refund request.",
        "state": (
            "Subject: Duplicate charge on invoice #4411\n"
            "Hi, we were billed twice for March. Please refund the duplicate today "
            "or we will cancel our plan."
        ),
        "questions": {
            "department": {
                "type": "choice",
                "instructions": "Which department should handle this email?",
                "criteria": {
                    "billing": "invoices, payments, refunds, charges",
                    "technical": "bugs, outages, errors, connectivity",
                    "sales": "pricing, new contracts, upgrades",
                    "other": "anything else",
                },
            },
            "urgency": {
                "type": "score",
                "instructions": "How urgent is this request?",
                "criteria": ["not urgent", "soon", "critical deadline or blocking issue"],
            },
            "refund_request": {
                "type": "noul",
                "instructions": "Does the sender explicitly ask for a refund?",
            },
        },
    },
    {
        "id": "spam-phishing",
        "title": "Spam / phishing screen",
        "icon": "\U0001F9EA",
        "summary": "Classify an email as spam, phishing, or legit — with calibrated probabilities.",
        "state": (
            "Subject: \u26a0 Final notice: verify your password now\n"
            "Dear customer, your account will be suspended within 24h. Click here to verify: "
            "http://secure-login-bank.example-verify.com/login"
        ),
        "questions": {
            "is_phishing": {"type": "noul", "instructions": "Is this email an attempt to trick the reader into revealing credentials or clicking a malicious link?"},
            "is_spam": {"type": "noul", "instructions": "Is this email unwanted bulk/spam mail?"},
            "legit": {
                "type": "choice",
                "instructions": "Best classification of this email?",
                "criteria": {
                    "phishing": "impersonation or malicious credential harvesting",
                    "spam": "bulk marketing, unwanted",
                    "legit": "legitimate business or personal mail",
                },
            },
        },
    },
    {
        "id": "moderation",
        "title": "Content moderation",
        "icon": "\U0001F6AB",
        "summary": "Toxicity, hate, severity — the guardrail step before a post goes live.",
        "state": (
            "Comment: You and everyone like you are worthless. Go away and never come back, "
            "you pathetic loser. Everyone who thinks like you deserves to be deleted."
        ),
        "questions": {
            "toxic": {"type": "noul", "instructions": "Is this content toxic (abusive, insulting, threatening)?"},
            "hate": {"type": "noul", "instructions": "Does this content attack a person or group based on identity characteristics?"},
            "severity": {
                "type": "score",
                "instructions": "How severe is the violation?",
                "criteria": ["none", "mild (annoying)", "moderate (clearly violating)", "severe (threatening or hateful)"],
            },
            "action": {
                "type": "choice",
                "instructions": "What action does this content call for?",
                "criteria": {
                    "allow": "fine as-is",
                    "flag": "mark for human review",
                    "remove": "delete automatically",
                },
            },
        },
    },
    {
        "id": "model-routing",
        "title": "LLM model router",
        "icon": "\u2699\ufe0f",
        "summary": "The LangChain RouterMiddleware pattern: pick the cheapest model that can handle this request.",
        "state": (
            "User request: What is 2 plus 2? Also tell me my order status for order #5521 — "
            "I have the account email saved."
        ),
        "questions": {
            "route": {
                "type": "choice",
                "instructions": "Which model tier should handle this request?",
                "criteria": {
                    "lookup": "direct fact, arithmetic, or data lookup — no reasoning needed",
                    "standard": "multi-step task with context: tool use, summarization, extraction",
                    "frontier": "complex architecture, high-stakes, or long multi-step reasoning",
                },
            },
            "tool_needed": {"type": "noul", "instructions": "Does this request require an external tool or system (e.g. order database)?"},
        },
    },
    {
        "id": "sentiment",
        "title": "Sentiment & product feedback",
        "icon": "\U0001F4AC",
        "summary": "Polarity on a rubric, plus targeted noul probes — cheap at scale for review mining.",
        "state": (
            "Review: I've used this for three years and it keeps failing at the worst moment. "
            "Support was slow twice. I got it back from a friend because it was cheap, "
            "but honestly I'm done after this last outage."
        ),
        "questions": {
            "sentiment": {
                "type": "score",
                "instructions": "Overall sentiment of the review?",
                "criteria": ["very negative", "negative", "neutral", "positive", "very positive"],
            },
            "mentions_outage": {"type": "noul", "instructions": "Does the reviewer mention a service outage or failure?"},
            "churn_risk": {"type": "noul", "instructions": "Does the reviewer indicate they will stop using the product?"},
        },
    },
    {
        "id": "bug-severity",
        "title": "Bug report grading",
        "icon": "\U0001F41B",
        "summary": "Grade a bug report before it enters the triage queue.",
        "state": (
            "Title: App crashes on startup after update 4.2.1\n"
            "Steps: 1) Update to 4.2.1 2) Launch app 3) Crash immediately, crash log attached. "
            "Happens on 3 of 4 test devices, always. Blocking all QA work."
        ),
        "questions": {
            "severity": {
                "type": "choice",
                "instructions": "Severity of this bug report?",
                "criteria": {
                    "blocker": "crash/data loss/total blockage for many users",
                    "major": "significant feature broken, workaround exists",
                    "minor": "small defect, limited impact",
                    "nit": "cosmetic or wording issue",
                },
            },
            "reproducible": {"type": "noul", "instructions": "Is the failure described as consistently reproducible?"},
            "blocking": {"type": "noul", "instructions": "Is this reported as blocking other work or releases?"},
        },
    },
    {
        "id": "agent-guardrail",
        "title": "Agent action guardrail",
        "icon": "\U0001F6E1\ufe0f",
        "summary": "Auto-Mode pattern: classify a pending tool call before the agent executes it.",
        "state": (
            "Pending tool call: rm -rf /var/www/data && rsync -a --delete backup:/mnt/data /var/www/data "
            "(requested by agent during 'clean up old website files' task)"
        ),
        "questions": {
            "risk": {
                "type": "choice",
                "instructions": "Risk class of this tool call?",
                "criteria": {
                    "safe": "read-only or clearly scoped and reversible",
                    "caution": "destructive but plausibly intended; confirm with user",
                    "block": "dangerous: irreversible broad delete, credentials, exfiltration",
                },
            },
            "destructive": {"type": "noul", "instructions": "Does this action permanently delete or overwrite data?"},
        },
    },
    {
        "id": "beauty-review",
        "title": "Beauty & cosmetics review",
        "icon": "\U0001F485",
        "summary": "Real-world example from a live demo (2026-09-19): sentiment, repurchase intent, primary aspect, scarcity language, toxicity.",
        "state": (
            "this lipstick is in enriched with nourishing grape oil, that makes your lips softer "
            "and less brittle. its sheer texture is very user friendly and makes the product outstanding."
        ),
        "questions": {
            "sentiment": {
                "type": "choice",
                "instructions": "Classify the overall sentiment of the review.",
                "criteria": {
                    "positive": "Expresses satisfaction, praise, or recommendation",
                    "neutral": "Factual description without clear emotional tone",
                    "negative": "Expresses dissatisfaction, criticism, or warning",
                },
            },
            "repurchase_intent": {
                "type": "noul",
                "instructions": "Does the reviewer indicate they would buy or use the product again, explicitly or through strong enthusiastic praise?",
            },
            "aspect_focus": {
                "type": "choice",
                "instructions": "Identify the primary product aspect discussed in the review.",
                "criteria": {
                    "fit": "How the item fits the body",
                    "fabric": "Material quality, texture, or feel",
                    "scent": "Smell or fragrance",
                    "longevity": "Durability or how long it lasts",
                    "value": "Price-to-quality ratio",
                    "appearance": "Visual design, color, or style",
                    "application": "How the product is applied, spread, or used",
                    "ingredients": "Specific components or formulas mentioned",
                    "texture": "Physical feel, consistency, or weight",
                    "not_discussed": "No specific product aspect discussed (e.g. only shipping, service, or a general impression)",
                },
            },
            "urgency": {
                "type": "noul",
                "instructions": "Yes only if the review creates a deadline or shortage (limited stock, an offer ending, 'hurry', 'last few'); no otherwise — an ordinary product description is no.",
            },
            "toxicity": {
                "type": "noul",
                "instructions": "Does the review contain abusive, harassing, or hate speech directed at a person or group?",
            },
        },
    },
]
