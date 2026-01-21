# Claude Development Notes

## Package Managers
- JavaScript/Node.js: `pnpm` (sometimes `pnpx`)
- Python: `uv`
  - Add deps to scripts: `uv add --script example.py <packages> --bounds exact`
  - Run scripts: `uv run example.py`
  - Create scripts: `uv init --script example.py --python 3.12`

## Validation
After many edits:
```bash
cd web && pnpm run check-types && pnpm run lint
```