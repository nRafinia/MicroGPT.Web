using System.Diagnostics;
using MicroGpt.Web.Models;

namespace MicroGpt.Web.Services;

/// <summary>
/// Runs training on the UI thread in adaptive chunks (Blazor WebAssembly is single-threaded),
/// yielding to the browser between chunks so the page stays interactive. Reserves a held-out
/// slice before training and computes percentile loss thresholds on it afterwards, so anomaly
/// thresholds are calibrated instead of arbitrary.
/// </summary>
public sealed class TrainingSession
{
    private const int MaxHoldout = 1000;
    private const int MaxChartPoints = 400;
    private CancellationTokenSource? _cts;

    public bool IsRunning { get; private set; }
    public string Status { get; private set; } = "";
    public int CurrentStep { get; private set; }
    public int TotalSteps { get; private set; }
    public double StepsPerSecond { get; private set; }
    public List<LossPoint> LossHistory { get; } = new();
    public double? LastEma { get; private set; }
    public Calibration? Calibration { get; private set; }
    public TimeSpan Elapsed { get; private set; }

    public event Action? Changed;

    public void Cancel() => _cts?.Cancel();

    public async Task<GptModel?> RunAsync(Dataset dataset, TrainSettings s)
    {
        _cts = new CancellationTokenSource();
        var ct = _cts.Token;
        IsRunning = true;
        LossHistory.Clear();
        Calibration = null;
        LastEma = null;
        CurrentStep = 0;
        TotalSteps = s.Steps;

        try
        {
            // --- Split: seeded shuffle, then reserve a held-out slice for calibration ---
            await SetStatus("Preparing data split");
            var docs = new List<string>(dataset.Docs);
            Shuffle(docs, new Random(s.Seed));
            var holdoutCount = Math.Min(MaxHoldout, (int)(docs.Count * s.HoldoutFraction));
            if (docs.Count - holdoutCount < 1) holdoutCount = 0;
            var holdout = docs.Take(holdoutCount).ToList();
            var train = docs.Skip(holdoutCount).ToList();

            // --- Tokenizer: vocabulary always covers the WHOLE file via the dataset charset ---
            await SetStatus(s.TokenizerKind == "bpe"
                ? $"Training BPE tokenizer ({s.BpeMerges} merges) — the page may pause briefly"
                : "Building character tokenizer");
            ITokenizer tokenizer = s.TokenizerKind == "bpe"
                ? BpeTokenizer.Train(train, numMerges: s.BpeMerges, baseCharsOverride: dataset.Charset)
                : new CharTokenizer(dataset.Charset);

            // --- Block size from the longest sampled document (+1 so the full doc fits) ---
            await SetStatus("Measuring block size");
            var maxTokens = 0;
            for (var i = 0; i < train.Count; i++)
            {
                var t = tokenizer.CountTokens(train[i]);
                if (t > maxTokens) maxTokens = t;
                if (i % 5000 == 4999) { await Task.Delay(1, ct); }
            }

            var config = new ModelConfig
            {
                NumLayers = s.NumLayers,
                EmbeddingDim = s.EmbeddingDim,
                NumHeads = s.NumHeads,
                BlockSize = maxTokens + 1,
            };
            config.Validate();

            var model = GptModel.CreateNew(config, tokenizer, s.Seed);
            var trainer = new Trainer(model, new TrainingOptions { Steps = s.Steps, LearningRate = s.LearningRate });

            // --- Adaptive chunked loop: target ~80 ms of work between browser paints ---
            await SetStatus("Training");
            var sw = Stopwatch.StartNew();
            var chunk = 10;
            double ema = double.NaN;
            while (trainer.Step < s.Steps && !ct.IsCancellationRequested)
            {
                var chunkSw = Stopwatch.StartNew();
                var end = Math.Min(trainer.Step + chunk, s.Steps);
                double loss = 0;
                while (trainer.Step < end)
                {
                    var doc = train[trainer.Step % train.Count];
                    loss = trainer.TrainStep(doc);
                    ema = double.IsNaN(ema) ? loss : 0.97 * ema + 0.03 * loss;
                }

                chunkSw.Stop();
                var ms = Math.Max(1, chunkSw.Elapsed.TotalMilliseconds);
                chunk = Math.Clamp((int)(chunk * 80.0 / ms), 2, 500);

                CurrentStep = trainer.Step;
                Elapsed = sw.Elapsed;
                StepsPerSecond = trainer.Step / Math.Max(0.001, sw.Elapsed.TotalSeconds);
                LastEma = ema;
                LossHistory.Add(new LossPoint(trainer.Step, loss, ema));
                if (LossHistory.Count > MaxChartPoints)
                {
                    // decimate: drop every other point, keeping the most recent one
                    for (var i = LossHistory.Count - 3; i > 0; i -= 2) LossHistory.RemoveAt(i);
                }

                Changed?.Invoke();
                await Task.Delay(1, ct);
            }

            if (ct.IsCancellationRequested)
            {
                await SetStatus($"Stopped at step {trainer.Step:N0} of {s.Steps:N0} — " +
                                "note: the learning-rate schedule did not complete");
                return model;
            }

            // --- Calibration on held-out clean data: percentile thresholds, not arbitrary margins ---
            if (holdout.Count >= 20)
            {
                await SetStatus($"Calibrating on {holdout.Count} held-out documents");
                var sampler = new Sampler(model, seed: s.Seed);
                var losses = new List<double>(holdout.Count);
                for (var i = 0; i < holdout.Count; i++)
                {
                    losses.Add(sampler.ComputeLoss(holdout[i]));
                    if (i % 50 == 49) { Changed?.Invoke(); await Task.Delay(1, ct); }
                }

                losses.Sort();
                Calibration = new Calibration(
                    holdout.Count,
                    Percentile(losses, 0.50), Percentile(losses, 0.90),
                    Percentile(losses, 0.95), Percentile(losses, 0.99));
            }

            await SetStatus($"Done — {s.Steps:N0} steps in {sw.Elapsed.TotalSeconds:F1}s " +
                            $"({StepsPerSecond:F1} steps/s), final loss (EMA) {ema:F3}");
            return model;
        }
        catch (OperationCanceledException)
        {
            await SetStatus("Stopped");
            return null;
        }
        finally
        {
            IsRunning = false;
            Changed?.Invoke();
        }
    }

    private async Task SetStatus(string status)
    {
        Status = status;
        Changed?.Invoke();
        await Task.Delay(15); // give the browser a frame to show the status
    }

    private static void Shuffle<T>(IList<T> list, Random rng)
    {
        for (var i = list.Count - 1; i > 0; i--)
        {
            var j = rng.Next(i + 1);
            (list[i], list[j]) = (list[j], list[i]);
        }
    }

    private static double Percentile(IReadOnlyList<double> sorted, double p)
    {
        if (sorted.Count == 0) return double.NaN;
        var idx = Math.Clamp((int)Math.Ceiling(p * sorted.Count) - 1, 0, sorted.Count - 1);
        return sorted[idx];
    }
}
