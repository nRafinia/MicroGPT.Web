# MicroGPT — In-Browser Lab

A Blazor WebAssembly demo of [MicroGPT](https://github.com/nRafinia/MicroGPT):
train, sample, and anomaly-score a tiny character-level GPT **entirely inside the
browser**. No server, no upload, no API key — after the initial page load the app
makes zero network requests (system fonts only, verifiable in the network tab).

## What it demonstrates

| Tab | Library feature |
|---|---|
| 1 Data | Streaming ingestion with a seeded reservoir sample (`--max-docs` semantics): memory is bounded by the cap, not the file size, while the character vocabulary and statistics cover the **entire** file. Plain text or CSV with column selection, length/character charts. |
| 2 Train | `Trainer.TrainStep` in adaptive chunks on the UI thread with a live loss chart. Editable hyperparameters (embedding dim, heads, layers, steps, lr, seed, char/BPE tokenizer, merges). A held-out slice calibrates loss percentiles (p50/p90/p95/p99) — thresholds are measured, not arbitrary. |
| 3 Generate | `Sampler.Sample` and prefix-conditioned `SampleContinuation`, temperature control, download as .txt. |
| 4 Score | `Sampler.ComputeLoss` per line for anomaly detection: histogram, calibrated percentile threshold slider, out-of-vocab handling, TSV export. |
| 5 Model | Versioned `.mgpt` binary round trip: download the trained model, upload it later (v1 and v2 files) and skip training entirely. |

## Project layout

```
src/MicroGpt.Web/
  Program.cs, Main.razor          app shell (nameplate, pipeline tabs, footer)
  Models.cs                       records shared by the panels
  Services/
    CsvParser.cs                  RFC4180-ish line parser + delimiter detection
    DatasetService.cs             streaming pass: reservoir sample + full-file stats
    TrainingSession.cs            chunked training, live loss, holdout calibration
    AppState.cs                   session state + file-download helper
  Components/
    Charts/                       dependency-free SVG charts (line, histogram, bars)
    Panels/                       one panel per tab
  wwwroot/                        index.html, css, js interop, bundled sample data
```

The project references `..\MicroGpt\MicroGpt.csproj` and uses **only the library's
public API** — nothing in the library was modified or duplicated.

## Add to the solution and run

```powershell
dotnet sln MicroGpt.sln add src\MicroGpt.Web
dotnet run --project src\MicroGpt.Web
```

## Publish

Interpreter mode (no extra workload needed):

```powershell
dotnet publish src\MicroGpt.Web -c Release -o publish
```

Ahead-of-time compilation — strongly recommended for real training speed in the
browser (larger download, needs the wasm-tools workload):

```powershell
dotnet workload install wasm-tools
dotnet publish src\MicroGpt.Web -c Release -p:RunAOTCompilation=true -o publish-aot
```

Serve any static file host from `publish/wwwroot`:

```powershell
dotnet tool install --global dotnet-serve
dotnet serve -d publish\wwwroot
```

## Deploy to GitHub Pages

1. In `wwwroot/index.html` change `<base href="/" />` to `<base href="/MicroGPT/" />`
   (or your repository name).
2. Use the included workflow: `.github/workflows/deploy-web.yml` publishes on every
   push to `main`. Enable Pages → Source: GitHub Actions in the repository settings.

## Notes and limits

- Blazor WebAssembly is single-threaded: training runs in adaptive chunks that
  yield to the browser (~80 ms per chunk), so the page stays interactive and a
  Stop button works. Stopping early leaves the learning-rate schedule incomplete.
- Attention cost is quadratic in row length; the app warns when the longest line
  exceeds 300 characters.
- The `.mgpt` file stores weights as doubles: parameter count × 8 bytes.
- Everything lives in tab memory. Closing the tab discards the dataset and model —
  download the `.mgpt` file to keep it.
