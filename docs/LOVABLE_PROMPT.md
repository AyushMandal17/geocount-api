# Prompts for Lovable

Paste **Prompt 1** into a new Lovable project. After it builds, paste **Prompt 2**, then the polish prompts.
Replace `API_URL_HERE` with your deployed API URL (no trailing slash) once the backend is live.

---

## Prompt 1: main app

Build a single-page web app called **GeoCount**: "Count objects in aerial and satellite images."
Use React, TypeScript, Tailwind and shadcn/ui. Do not use Supabase or any login. The app talks to an external REST API.

**Config:** create `src/config.ts` exporting `export const API_URL = "API_URL_HERE";` and use it for every request, so I can change it in one place.

**Layout (dark theme, clean, a little map/satellite feel):**
1. Header with the logo text "GeoCount" and a GitHub link icon.
2. Hero line and one sentence of description.
3. Two columns on desktop, stacked on mobile.
   - **Left (controls):**
     - Drag-and-drop image upload (jpg, png, tif; max 15 MB) with a preview and a "remove" button.
     - Text input "Ask a question (optional)" with placeholder "How many ships are there?" and 3 clickable example chips: "How many planes?", "How many vehicles?", "Count storage tanks".
     - Switch "Sliced inference (better for small objects)", default on.
     - Collapsible "Advanced" section: a confidence slider 0.05-0.95 with a "Use tuned defaults" switch on by default (when on, do not send `conf`), and an optional number input "Ground resolution (metres per pixel)".
     - Primary button "Count objects", disabled until an image is chosen.
   - **Right (results):**
     - Empty state with a short hint.
     - Loading state: spinner and the text "Analysing image, this can take 10-30 seconds on CPU".
     - Result: a large highlighted answer sentence, then the annotated image (click to open full size in a dialog), then a table of class and count with a bar chart (Recharts), then, if present, a "Estimated lengths" table (class, n, mean m, median m), and a footer line with processing time and image size.
     - A "Download annotated image" button.

**API contract:** `POST {API_URL}/api/count` as `multipart/form-data` with fields:
`image` (file), `question` (string, may be empty), `use_slicing` ("true"/"false"), `conf` (only when the tuned-defaults switch is off), `gsd` (only if the user typed a number). Never send empty strings for `conf` or `gsd`; omit the field.

Response JSON:
```json
{
  "answer": "There are 14 ships.",
  "counts": [{"class": "ship", "count": 14}],
  "total": 14,
  "matched_classes": ["ship"],
  "measurements": [{"class": "ship", "n": 14, "mean_length_m": 21.3, "median_length_m": 19.8}],
  "image": {"width": 2048, "height": 1536, "downscaled": false},
  "settings": {"use_slicing": true, "conf": null, "tuned_thresholds": true},
  "elapsed_sec": 8.4,
  "annotated_image": "data:image/jpeg;base64,..."
}
```
`matched_classes` is null when the question did not name a known class; then the table lists everything detected.

**Errors:** show a toast with the server's `detail` message for 400/413/422, and a friendly "The server is waking up or unreachable, try again in a minute" for network errors or 5xx. If `image.downscaled` is true, show a small notice that the image was reduced to 4096 px. If `total` is 0, show "No objects found. Try turning slicing on or lowering the confidence."

Keep the code simple and typed. Put the API call in `src/lib/api.ts`.

---

## Prompt 2: "How it works" and benchmarks section

Add a section below the app with the heading "How it works". Include:
- A 3-step row: Upload, Detect (YOLO26-OBB on overlapping tiles with rotated NMS), Count and answer.
- A "Why tiled?" card with this table (counting mean absolute error, lower is better):

| Method | 1-10 | 11-50 | 51-100 | 100+ |
|---|---|---|---|---|
| YOLO26n full image | 1.6 | 5.7 | 16.7 | 86.0 |
| YOLO26n tiled | 1.5 | 5.1 | 20.7 | 41.2 |
| SAM 3 (untiled) | 2.0 | 8.5 | 15.0 | 195.7 |
| Qwen3-VL-4B | 1.5 | 10.7 | 34.1 | 174.5 |
| EarthDial-4B | 1.3 | 9.3 | 39.1 | 147.0 |

- A "Limitations" card: only the 15 DOTA classes are supported; very dense scenes are still under- or over-counted; images over 4096 px are downscaled; the free server can be slow on the first request.
- A footer with links to the GitHub repos and "Built by Ayush Mandal".

## Polish prompts (use one at a time)
- "Add a sample-images row: three small thumbnails the user can click to load as the input. Use the files in /public/samples."
- "Make the annotated image zoomable with pan (mouse wheel and drag)."
- "Store the last 5 results in React state and show them as a history strip; clicking one restores it."
- "Add a favicon and Open Graph meta tags with the title and description."
