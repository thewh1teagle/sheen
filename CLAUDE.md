# Claude Development Notes

## Package Managers
- JavaScript/Node.js: `pnpm` (sometimes `pnpx`)
- Python: `uv`
  - Add deps to scripts: `uv add --script example.py <packages> --bounds exact`
  - Run scripts: `uv run example.py`
  - Create scripts: `uv init --script example.py --python 3.12`

## Validation

For each plan, you can create self-contained validation scripts.

Structure:
- `plans/<name>/<name>_001.py`
- `plans/<name>/<name>_001.md`

Each Python file should be a standalone `uv` script with its own dependencies declared at the top.

Example:
```bash
uv run plans/<name>/<name>_001.py
```