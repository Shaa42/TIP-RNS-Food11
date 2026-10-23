## Resizing
First make sure to have the "food-11" dataset downloaded and unzipped in the "data" folder. Then, run the following command to resize the images to axb pixels:

```bash
uv sync
uv run python scripts/resize.py --size a b --output-dir food-11-axb --input-dir food-11
```
