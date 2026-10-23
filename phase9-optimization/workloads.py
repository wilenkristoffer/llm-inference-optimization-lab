"""
Phase 9: the three workloads speculative decoding is measured on.

The speedup depends entirely on how often a guess is right, so one prompt says
nothing. Three kinds, three prompts each:

    code         predictable syntax, boilerplate, repeated identifiers
    extraction   fields copied out of a document in the prompt - the agent's
                 tool-result case, and the one ngram (prompt lookup) is for
    prose        open-ended writing, hard to guess

Every prompt asks for more than 256 tokens of answer except extraction, whose
answers are naturally short: forcing length past EOS would measure the model
rambling, not the task.
"""

ORDER_DOC = """{
  "order_id": "ORD-2026-0418-77231",
  "placed_at": "2026-10-18T14:22:09Z",
  "customer": {"id": "CUST-009812", "name": "Elin Bergstrom",
               "email": "elin.bergstrom@example.se", "tier": "gold"},
  "shipping_address": {"street": "Sveavagen 44", "postal_code": "113 59",
                       "city": "Stockholm", "country": "SE"},
  "items": [
    {"sku": "KB-ERGO-MX3", "title": "Ergonomic keyboard MX3, Nordic layout",
     "qty": 1, "unit_price_sek": 1499.00},
    {"sku": "MS-VERT-200", "title": "Vertical mouse 200, wireless",
     "qty": 2, "unit_price_sek": 649.00},
    {"sku": "CB-USBC-2M", "title": "USB-C cable 2 m, braided",
     "qty": 3, "unit_price_sek": 129.00}
  ],
  "payment": {"method": "card", "card_last4": "4417", "status": "captured",
              "transaction_id": "TX-7f3a9c21-0b44-4e8e-9d12-55c0aa61e9b0"},
  "status": "awaiting_shipment",
  "notes": "Leave at reception if nobody answers. Gate code 4471."
}"""

TICKET_DOC = """Ticket INC-58213 (priority P2), opened 2026-10-21 09:14 by
Johan Lindqvist (johan.lindqvist@example.se), assigned to team Platform-Storage.

Summary: Nightly backup job backup-nas-03 failed three nights in a row.

Description: The job runs at 02:00 from scheduler host sched-prod-02 and
writes to share \\\\nas-03.corp.example.se\\backups\\nightly. Since 2026-10-18
it exits with code 23 after about 41 minutes. The log line before the failure
is "rsync: write failed on /mnt/nas03/nightly/db/pg-main-2026-10-20.tar.zst:
No space left on device (28)". Free space on volume vol2 is 0.4% (12.1 GB of
3.0 TB). Retention policy is 14 daily copies; 19 copies are currently present
because the cleanup task cleanup-nightly-retention has been disabled since the
change CHG-11902 on 2026-10-09.

Impact: no fresh backup of pg-main for 3 days. Recovery point objective is
24 hours, so the RPO is breached.

Requested: re-enable cleanup-nightly-retention, delete copies older than 14
days, rerun backup-nas-03 manually, and confirm success in this ticket."""

CODE_DOC = '''import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CacheEntry:
    key: str
    path: Path
    size_bytes: int
    created_at: float
    tags: list[str] = field(default_factory=list)


class DiskCache:
    def __init__(self, root: Path, max_bytes: int = 2_000_000_000) -> None:
        self.root = root
        self.max_bytes = max_bytes
        self.index_path = root / "index.json"
        self.entries: dict[str, CacheEntry] = {}

    def key_for(self, url: str, params: dict[str, str] | None = None) -> str:
        blob = url + json.dumps(params or {}, sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def get(self, key: str) -> bytes | None:
        entry = self.entries.get(key)
        return entry.path.read_bytes() if entry else None

    def put(self, key: str, data: bytes, tags: list[str] | None = None) -> CacheEntry:
        ...

    def evict_until_fits(self, incoming_bytes: int) -> list[str]:
        ...

    def total_bytes(self) -> int:
        return sum(e.size_bytes for e in self.entries.values())
'''

PROMPTS = [
    ("code", "palindrome",
     "Write a Python function that checks whether a string is a palindrome, "
     "ignoring case, spaces and punctuation. Include a docstring, type hints and "
     "five unittest test cases."),
    ("code", "lru",
     "Implement an LRU cache class in Python with get and put methods, both O(1), "
     "using a dict and a doubly linked list. Include docstrings and a short usage "
     "example."),
    ("code", "csv",
     "Write a Python script that reads a CSV file of sales (columns: date, region, "
     "product, units, unit_price), computes total revenue per region and per month, "
     "and prints both as aligned tables. Use only the standard library, with argparse."),

    ("extraction", "order",
     "Here is an order record from our system:\n\n" + ORDER_DOC + "\n\n"
     "Return a JSON object with exactly these fields, values copied verbatim from "
     "the record: order_id, customer_name, customer_email, street, postal_code, "
     "city, transaction_id, and items (a list of objects with sku, title, qty). "
     "Output only the JSON."),
    ("extraction", "ticket",
     "Here is an incident ticket:\n\n" + TICKET_DOC + "\n\n"
     "Extract these fields as JSON, copying values verbatim: ticket_id, priority, "
     "reporter_name, reporter_email, team, job_name, scheduler_host, share_path, "
     "error_line, exit_code, change_id, disabled_task. Output only the JSON."),
    ("extraction", "signatures",
     "Here is a Python module:\n\n```python\n" + CODE_DOC + "```\n\n"
     "List every class and every method signature in it, exactly as written "
     "(name, parameters with type hints, return type), one per line, in order. "
     "Output only the list."),

    ("prose", "lagoon",
     "Write a 500-word description of a coastal lagoon at dawn."),
    ("prose", "story",
     "Write the opening of a short story about a lighthouse keeper who receives a "
     "letter addressed to someone who died forty years ago. About 400 words."),
    ("prose", "parks",
     "Write a short essay arguing that cities should spend more on public parks. "
     "Give three distinct reasons, each with an example."),
]

MAX_TOKENS = 256


def messages(prompt):
    return [{"role": "user", "content": prompt}]
