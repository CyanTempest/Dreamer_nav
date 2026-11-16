"""Compare the rows (and summary) of two nav4.evaluate result JSONs; prints IDENTICAL or the differences."""
import json
import sys

a, b = (json.load(open(p)) for p in sys.argv[1:3])
diff = [i for i, (x, y) in enumerate(zip(a["rows"], b["rows"])) if x != y]
print(json.dumps(dict(files=sys.argv[1:3], rows=[len(a["rows"]), len(b["rows"])], differing_rows=diff,
                      summary_equal=a["summary"] == b["summary"], loop_steps=[a["loop_steps"], b["loop_steps"]],
                      leakage=[a["leakage"], b["leakage"]],
                      checkpoint_sha256_equal=a.get("checkpoint_sha256") == b.get("checkpoint_sha256"))))
print("IDENTICAL" if not diff and len(a["rows"]) == len(b["rows"]) and a["summary"] == b["summary"] else "DIFFERENT")
