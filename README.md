# GeoCount API

Count objects in aerial and satellite images. Upload an image, optionally ask "How many ships?", and get a count, an annotated image, and (with a ground resolution) estimated object lengths.

Backend: FastAPI + YOLO26-OBB with tiled inference. Frontend: React app built with Lovable (see [GeoCount Web](https://github.com/YOUR_USER/geocount-web)).
Motivated by the VQA and counting parts of the ISRO GeoNLI problem statement.

**Live demo:** https://YOUR-SITE.lovable.app  |  **API docs:** https://YOUR_USER-geocount-api.hf.space/docs

<!-- Add demo.gif here -->

## How it works
```
image + question -> question parser -> YOLO26-OBB on overlapping 1024px tiles
                 -> rotated NMS (IoU 0.3, containment 0.7) -> per-class thresholds
                 -> counts + answer sentence + annotated image (+ lengths if GSD given)
```
Recognises the 15 DOTA classes (plane, ship, storage tank, small/large vehicle, harbor, bridge, and others).

## Why tiled YOLO
Benchmarked against SAM 3, Qwen3-VL-4B and EarthDial-4B on 60 DOTA windows, split by object density. Counting MAE (lower is better):

| Method | 1-10 | 11-50 | 51-100 | 100+ |
|---|---|---|---|---|
| YOLO26n full image | 1.6 | 5.7 | 16.7 | 86.0 |
| **YOLO26n tiled** | 1.5 | **5.1** | 20.7 | **41.2** |
| SAM 3 (untiled) | 2.0 | 8.5 | 15.0 | 195.7 |
| Qwen3-VL-4B | 1.5 | 10.7 | 34.1 | 174.5 |
| EarthDial-4B | 1.3 | 9.3 | 39.1 | 147.0 |

Tiling roughly halves the error on dense scenes, and the VLM and SAM 3 options need a GPU and are slower. Details, caveats and failure cases: [docs/benchmarks.md](docs/benchmarks.md).

## API
`POST /api/count` (multipart/form-data)

| Field | Type | Notes |
|---|---|---|
| `image` | file | JPG, PNG or TIFF, up to 15 MB; larger than 4096 px is downscaled |
| `question` | string | optional, e.g. "How many cars?" |
| `use_slicing` | bool | default `true` |
| `conf` | float | 0.05-0.95; omit to use tuned per-class thresholds |
| `gsd` | float | metres per pixel; enables length estimates |

```bash
curl -X POST http://localhost:8000/api/count \
  -F image=@sample.png -F "question=How many ships?" -F gsd=0.26
```
Also: `GET /api/health`, `GET /api/classes`, interactive docs at `/docs`.

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```
Weights download on first request. Choose a model with `MODEL_WEIGHTS` (default `yolo26m-obb.pt`; `yolo26x-obb.pt` is more accurate and much slower on CPU).

Tests (no weights needed): `pip install -r requirements-ci.txt && pytest`

## Deploy
Docker image for Hugging Face Spaces (free CPU): `deploy/push_to_space.sh`. Set `ALLOWED_ORIGINS` to your frontend URL in the Space settings.

## Limitations
- Closed set: only the 15 DOTA classes.
- Dense small objects are still miscounted (see failure cases in the benchmarks).
- Per-class confidence thresholds were tuned for YOLO26-x on a small split; re-tune for other weights.
- CPU inference takes roughly 10-30 s for a 2048 px image.
- Length estimates are only as accurate as the supplied GSD, and boxes cut by the image edge bias them.

## Structure
```
app/          engine.py (tiling, NMS, drawing), query.py (parser), main.py (API)
tests/        unit tests and API tests with a fake model
docs/         benchmarks.md, raw results, LOVABLE_PROMPT.md
deploy/       Hugging Face Space deploy script
```

## Credits
DOTA dataset, Ultralytics YOLO, VRSBench, SAM 3, Qwen3-VL and EarthDial (used in the benchmarks), ISRO GeoNLI problem statement for the motivation.
