namespace MicroGpt;

/// <summary>
/// Inference: pure forward pass with a KV cache, no gradients. Also provides
/// forward-only loss evaluation (used for validation and gradient checking).
/// </summary>
public sealed class Sampler
{
    private readonly GptModel _model;
    private readonly Random _rng;

    public Sampler(GptModel model, Random? rng = null)
    {
        _model = model;
        _rng = rng ?? new Random();
    }

    public Sampler(GptModel model, int seed) : this(model, new Random(seed))
    {
    }

    /// <summary>Generates a single document (stops at BOS or at <see cref="ModelConfig.BlockSize"/>).</summary>
    public string Sample(double temperature = 0.5)
    {
        var cfg = _model.Config;
        var tokenizer = _model.Tokenizer;
        var kv = NewKvCache();

        var tokenId = tokenizer.BosId;
        var sampled = new List<int>();
        for (var posId = 0; posId < cfg.BlockSize; posId++)
        {
            var logits = Forward(tokenId, posId, kv);
            tokenId = SampleToken(logits, temperature, tokenizer);
            if (tokenId == tokenizer.BosId)
            {
                break;
            }

            sampled.Add(tokenId);
        }

        return tokenizer.Decode(sampled);
    }

    /// <summary>
    /// Conditional generation: forwards <paramref name="prefix"/> through the model (forcing
    /// each of its tokens, not sampling them), then samples a continuation from there. Useful
    /// for "fill in the rest of this row given these fields" style generation. Returns only the
    /// newly generated suffix (the prefix is not repeated in the result).
    /// </summary>
    public string SampleContinuation(string prefix, double temperature = 0.5)
    {
        var cfg = _model.Config;
        var tokenizer = _model.Tokenizer;
        var kv = NewKvCache();

        // EncodeWithBos wraps as [BOS, ...prefixTokens, BOS]; drop the trailing BOS so we can
        // force-feed exactly [BOS, ...prefixTokens] and let sampling take over from there.
        var seed = tokenizer.EncodeWithBos(prefix);
        seed = seed[..^1];
        if (seed.Length > cfg.BlockSize)
        {
            throw new ArgumentException(
                $"Prefix encodes to {seed.Length} tokens, which exceeds the model's block size ({cfg.BlockSize}).");
        }

        double[] logits = Array.Empty<double>();
        for (var pos = 0; pos < seed.Length; pos++)
        {
            logits = Forward(seed[pos], pos, kv); // force-feed; logits from the last step seed the continuation
        }

        var sampled = new List<int>();
        var tokenId = SampleToken(logits, temperature, tokenizer);
        for (var posId = seed.Length; tokenId != tokenizer.BosId && posId < cfg.BlockSize; posId++)
        {
            sampled.Add(tokenId);
            logits = Forward(tokenId, posId, kv);
            tokenId = SampleToken(logits, temperature, tokenizer);
        }

        return tokenizer.Decode(sampled);
    }

    private int SampleToken(double[] logits, double temperature, ITokenizer tokenizer)
    {
        for (var i = 0; i < logits.Length; i++)
        {
            logits[i] /= temperature;
        }

        var probs = MathOps.Softmax(logits);
        var r = _rng.NextDouble() * probs.Sum();
        double cum = 0;
        var tokenId = tokenizer.Size - 1;
        for (var i = 0; i < probs.Length; i++)
        {
            cum += probs[i];
            if (!(r < cum))
            {
                continue;
            }

            tokenId = i;
            break;
        }

        return tokenId;
    }

    /// <summary>Generates <paramref name="count"/> documents.</summary>
    public IReadOnlyList<string> Sample(int count, double temperature = 0.5)
    {
        var results = new string[count];
        for (var i = 0; i < count; i++) results[i] = Sample(temperature);
        return results;
    }

    /// <summary>
    /// Mean next-token cross-entropy loss of the model over one document
    /// (same quantity the trainer reports, computed without any gradient side effects).
    /// </summary>
    public double ComputeLoss(string document)
    {
        var cfg = _model.Config;
        var tokens = _model.Tokenizer.EncodeWithBos(document);
        var n = Math.Min(cfg.BlockSize, tokens.Length - 1);
        if (n < 1) throw new ArgumentException("Document is empty.", nameof(document));

        var kv = NewKvCache();
        double loss = 0;
        for (var t = 0; t < n; t++)
        {
            var logits = Forward(tokens[t], t, kv);
            var probs = MathOps.Softmax(logits);
            loss += -Math.Log(probs[tokens[t + 1]]);
        }

        return loss / n;
    }

    // ---------------- Forward one token through all layers (with KV cache) ----------------

    private sealed record KvCache(List<double[]>[] K, List<double[]>[] V);

    private KvCache NewKvCache()
    {
        var nLayer = _model.Config.NumLayers;
        var k = new List<double[]>[nLayer];
        var v = new List<double[]>[nLayer];
        for (var l = 0; l < nLayer; l++)
        {
            k[l] = new List<double[]>();
            v[l] = new List<double[]>();
        }

        return new KvCache(k, v);
    }

    private double[] Forward(int tokenId, int posId, KvCache kv)
    {
        var cfg = _model.Config;
        int nEmbd = cfg.EmbeddingDim, nHead = cfg.NumHeads, headDim = cfg.HeadDim;
        var invSqrtHd = 1.0 / Math.Sqrt(headDim);

        var x0 = new double[nEmbd];
        var te = _model.Wte.Row(tokenId);
        var pe = _model.Wpe.Row(posId);
        for (var i = 0; i < nEmbd; i++)
        {
            x0[i] = te[i] + pe[i];
        }

        var (x, _) = MathOps.RmsNorm(x0);

        for (var l = 0; l < cfg.NumLayers; l++)
        {
            var w = _model.Layers[l];
            var (x2, _) = MathOps.RmsNorm(x);
            var q = MathOps.Linear(x2, w.Wq);
            var K = kv.K[l];
            var V = kv.V[l];
            K.Add(MathOps.Linear(x2, w.Wk));
            V.Add(MathOps.Linear(x2, w.Wv));

            var xattn = new double[nEmbd];
            for (var h = 0; h < nHead; h++)
            {
                var hs = h * headDim;
                var la = new double[K.Count];
                for (var tp = 0; tp < K.Count; tp++)
                {
                    double dot = 0;
                    for (var j = 0; j < headDim; j++)
                    {
                        dot += q[hs + j] * K[tp][hs + j];
                    }

                    la[tp] = dot * invSqrtHd;
                }

                var aw = MathOps.Softmax(la);
                for (var j = 0; j < headDim; j++)
                {
                    double s = 0;
                    for (var tp = 0; tp < K.Count; tp++)
                    {
                        s += aw[tp] * V[tp][hs + j];
                    }

                    xattn[hs + j] = s;
                }
            }

            var x3 = MathOps.Linear(xattn, w.Wo);
            var x4 = new double[nEmbd];
            for (var i = 0; i < nEmbd; i++)
            {
                x4[i] = x3[i] + x[i];
            }

            var (x5, _) = MathOps.RmsNorm(x4);
            var h1 = MathOps.Linear(x5, w.Fc1);
            for (var i = 0; i < h1.Length; i++)
            {
                h1[i] = Math.Max(0, h1[i]);
            }

            var x6 = MathOps.Linear(h1, w.Fc2);
            var x7 = new double[nEmbd];
            for (var i = 0; i < nEmbd; i++)
            {
                x7[i] = x6[i] + x4[i];
            }

            x = x7;
        }

        return MathOps.Linear(x, _model.LmHead);
    }
}
