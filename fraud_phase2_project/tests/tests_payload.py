import json
from pathlib import Path
SAMPLE_PAYLOAD = json.loads((Path(__file__).parent / 'sample_request.json').read_text())
