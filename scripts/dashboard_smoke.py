import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from streamlit.testing.v1 import AppTest

app_path = BASE_DIR / "dashboard" / "app.py"

at = AppTest.from_file(str(app_path))
at.run()

assert not at.exception, at.exception
assert len(at.title) == 1, len(at.title)
assert len(at.tabs) == 4, len(at.tabs)
assert len(at.metric) >= 4, len(at.metric)
assert len(at.selectbox) >= 1, len(at.selectbox)
assert len(at.button) >= 1, len(at.button)
assert len(at.dataframe) >= 2, len(at.dataframe)

summary = at.metric[0]
print(f"title: {at.title[0].value}")
print(f"tabs: {len(at.tabs)} | metrics: {len(at.metric)} | tables: {len(at.dataframe)}")
print(f"first metric: {summary.label} = {summary.value}")

missing = [name for name in ("summary.json", "rag_eval.json", "ab_routing.json")
           if not (BASE_DIR / "reports" / name).exists()]
assert not missing, f"run the eval scripts first: {missing}"

print("\n[all dashboard_smoke assertions PASSED]", flush=True)
