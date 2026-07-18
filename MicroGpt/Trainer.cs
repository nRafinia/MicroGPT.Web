namespace MicroGpt;

/// <summary>
/// Training: forward pass with cached intermediates, manual BPTT backward, Adam update.
/// Identical math to the original single-layer implementation, generalized to
/// <see cref="ModelConfig.NumLayers"/> transformer blocks.
/// </summary>
public sealed class Trainer
{
    private readonly GptModel _model;
    private readonly TrainingOptions _options;
    private readonly double[][] _mBuf;
    private readonly double[][] _vBuf;
    private int _step;

    public Trainer(GptModel model, TrainingOptions? options = null)
    {
        _model = model;
        _options = options ?? new TrainingOptions();
        _options.Validate();
        _mBuf = model.Tensors.Select(t => new double[t.Data.Length]).ToArray();
        _vBuf = model.Tensors.Select(t => new double[t.Data.Length]).ToArray();
    }

    /// <summary>Steps completed so far.</summary>
    public int Step => _step;

    /// <summary>
    /// Runs <see cref="TrainingOptions.Steps"/> optimization steps, cycling through
    /// <paramref name="documents"/> in order (shuffle beforehand if desired).
    /// </summary>
    public void Train(IReadOnlyList<string> documents, Action<int, double>? onStep = null)
    {
        if (documents.Count == 0) throw new ArgumentException("At least one document is required.", nameof(documents));
        for (var i = 0; i < _options.Steps; i++)
        {
            var loss = TrainStep(documents[_step % documents.Count]);
            onStep?.Invoke(_step, loss); // _step already incremented; reports 1-based step count
        }
    }

    // --- Per-position, per-layer intermediate cache for backward (replaces a computational graph) ---
    private sealed class LayerCache
    {
        public double[] XIn = null!;   // block input (residual source of the attention block)
        public double[] X2 = null!;
        public double S1;              // rmsnorm inside attention
        public double[] Q = null!;     // query for this position
        public double[][] AttnW = null!; // attention weights per head
        public double[] XAttn = null!; // concatenated head outputs
        public double[] X4 = null!;    // input to MLP block (residual)
        public double[] X5 = null!;
        public double S2;              // rmsnorm inside MLP
        public double[] H1 = null!;    // fc1 output (before ReLU)
        public double[] H2 = null!;    // after ReLU
        public double[] X7 = null!;    // block output
    }

    private sealed class PosCache
    {
        public int TokenId;
        public double[] X0 = null!;
        public double S0;              // embedding and first rmsnorm
        public double[] X1 = null!;    // input to the first block
        public LayerCache[] Layers = null!;
        public double[] Probs = null!; // final softmax probabilities
    }

    /// <summary>Runs one forward+backward+update step on a single document and returns the mean loss.</summary>
    public double TrainStep(string document)
    {
        var loss = ForwardBackward(document);
        ApplyAdam();
        _step++;
        return loss;
    }

    /// <summary>
    /// Runs one forward+backward pass, accumulating gradients into the model tensors
    /// WITHOUT applying an optimizer update. Returns the mean loss. Exposed for
    /// gradient checking and custom training loops.
    /// </summary>
    public double ForwardBackward(string document)
    {
        var cfg = _model.Config;
        var tokenizer = _model.Tokenizer;
        int nEmbd = cfg.EmbeddingDim, nHead = cfg.NumHeads, headDim = cfg.HeadDim, nLayer = cfg.NumLayers;
        var invSqrtHd = 1.0 / Math.Sqrt(headDim);

        var tokens = tokenizer.EncodeWithBos(document);
        var n = Math.Min(cfg.BlockSize, tokens.Length - 1);
        if (n < 1) throw new ArgumentException("Document is empty.", nameof(document));

        // ---------------- Forward (saving intermediates for backward) ----------------
        var cache = new PosCache[n];
        var K = NewLayerPosBuffers(nLayer, n); // KV cache: [layer][position][nEmbd]
        var V = NewLayerPosBuffers(nLayer, n);
        double loss = 0;

        for (var t = 0; t < n; t++)
        {
            var c = new PosCache
            {
                TokenId = tokens[t],
                X0 = new double[nEmbd],
                Layers = new LayerCache[nLayer],
            };

            var te = _model.Wte.Row(c.TokenId);
            var pe = _model.Wpe.Row(t);
            for (var i = 0; i < nEmbd; i++)
            {
                c.X0[i] = te[i] + pe[i];
            }

            (c.X1, c.S0) = MathOps.RmsNorm(c.X0);

            var xIn = c.X1;
            for (var l = 0; l < nLayer; l++)
            {
                var w = _model.Layers[l];
                var lc = c.Layers[l] = new LayerCache { XIn = xIn };

                // --- Attention block ---
                (lc.X2, lc.S1) = MathOps.RmsNorm(xIn);
                lc.Q = MathOps.Linear(lc.X2, w.Wq);
                K[l][t] = MathOps.Linear(lc.X2, w.Wk);
                V[l][t] = MathOps.Linear(lc.X2, w.Wv);

                lc.XAttn = new double[nEmbd];
                lc.AttnW = new double[nHead][];
                for (var h = 0; h < nHead; h++)
                {
                    var hs = h * headDim;
                    var logitsA = new double[t + 1];
                    for (var tp = 0; tp <= t; tp++)
                    {
                        double dot = 0;
                        for (var j = 0; j < headDim; j++)
                        {
                            dot += lc.Q[hs + j] * K[l][tp][hs + j];
                        }

                        logitsA[tp] = dot * invSqrtHd;
                    }

                    var aw = MathOps.Softmax(logitsA);
                    lc.AttnW[h] = aw;
                    for (var j = 0; j < headDim; j++)
                    {
                        double s = 0;
                        for (var tp = 0; tp <= t; tp++)
                        {
                            s += aw[tp] * V[l][tp][hs + j];
                        }

                        lc.XAttn[hs + j] = s;
                    }
                }

                var x3 = MathOps.Linear(lc.XAttn, w.Wo);
                lc.X4 = new double[nEmbd];
                for (var i = 0; i < nEmbd; i++)
                {
                    lc.X4[i] = x3[i] + xIn[i]; // residual
                }

                // --- MLP block ---
                (lc.X5, lc.S2) = MathOps.RmsNorm(lc.X4);
                lc.H1 = MathOps.Linear(lc.X5, w.Fc1);
                lc.H2 = new double[lc.H1.Length];
                for (var i = 0; i < lc.H1.Length; i++)
                {
                    lc.H2[i] = Math.Max(0, lc.H1[i]);
                }

                var x6 = MathOps.Linear(lc.H2, w.Fc2);
                lc.X7 = new double[nEmbd];
                for (var i = 0; i < nEmbd; i++)
                {
                    lc.X7[i] = x6[i] + lc.X4[i]; // residual
                }

                xIn = lc.X7;
            }

            // logits and loss
            var logits = MathOps.Linear(xIn, _model.LmHead);
            c.Probs = MathOps.Softmax(logits);
            loss += -Math.Log(c.Probs[tokens[t + 1]]);
            cache[t] = c;
        }

        loss /= n;

        // ---------------- Backward (manual BPTT, reversed over positions) ----------------
        // Gradients for K and V at each position also arrive from later positions (causal attention),
        // so they accumulate in the reverse loop and are complete when we reach that position.
        var dK = NewLayerPosBuffers(nLayer, n, nEmbd);
        var dV = NewLayerPosBuffers(nLayer, n, nEmbd);

        for (var t = n - 1; t >= 0; t--)
        {
            var c = cache[t];

            // cross-entropy+softmax derivative: dlogits = (p − onehot)/n
            var dlogits = new double[tokenizer.Size];
            for (var i = 0; i < tokenizer.Size; i++)
            {
                dlogits[i] = c.Probs[i] / n;
            }

            dlogits[tokens[t + 1]] -= 1.0 / n;

            var xLast = nLayer > 0 ? c.Layers[nLayer - 1].X7 : c.X1;
            var d = MathOps.LinearBackward(xLast, _model.LmHead, dlogits); // d = dX7 of the last layer

            for (var l = nLayer - 1; l >= 0; l--)
            {
                var w = _model.Layers[l];
                var lc = c.Layers[l];
                var dX7 = d;

                // MLP backward (residual: gradient flows to both branches)
                var dX6 = dX7; // fc2 branch
                var dH2 = MathOps.LinearBackward(lc.H2, w.Fc2, dX6);
                var dH1 = new double[dH2.Length];
                for (var i = 0; i < dH1.Length; i++)
                {
                    dH1[i] = lc.H1[i] > 0 ? dH2[i] : 0; // ReLU
                }

                var dX5 = MathOps.LinearBackward(lc.X5, w.Fc1, dH1);
                var dX4 = MathOps.RmsNormBackward(lc.X4, lc.S2, dX5);
                for (var i = 0; i < nEmbd; i++)
                {
                    dX4[i] += dX7[i]; // residual branch
                }

                // Attention backward
                var dX3 = dX4;
                var dXAttn = MathOps.LinearBackward(lc.XAttn, w.Wo, dX3);
                var dQ = new double[nEmbd];
                for (var h = 0; h < nHead; h++)
                {
                    var hs = h * headDim;
                    var aw = lc.AttnW[h];
                    var T = t + 1;

                    // d(attn_weights) and d(V)
                    var da = new double[T];
                    for (var tp = 0; tp < T; tp++)
                    {
                        double s = 0;
                        for (var j = 0; j < headDim; j++)
                        {
                            s += dXAttn[hs + j] * V[l][tp][hs + j];
                            dV[l][tp][hs + j] += aw[tp] * dXAttn[hs + j];
                        }

                        da[tp] = s;
                    }

                    // softmax backward: dlogit_t' = a_t'·(da_t' − Σ a·da)
                    double dotAda = 0;
                    for (var tp = 0; tp < T; tp++)
                    {
                        dotAda += aw[tp] * da[tp];
                    }

                    for (var tp = 0; tp < T; tp++)
                    {
                        var dl = aw[tp] * (da[tp] - dotAda) * invSqrtHd;
                        for (var j = 0; j < headDim; j++)
                        {
                            dQ[hs + j] += dl * K[l][tp][hs + j];
                            dK[l][tp][hs + j] += dl * lc.Q[hs + j];
                        }
                    }
                }

                // dK[l][t] and dV[l][t] are now complete (all positions >= t have been processed)
                var dX2 = MathOps.LinearBackward(lc.X2, w.Wq, dQ);
                var dX2K = MathOps.LinearBackward(lc.X2, w.Wk, dK[l][t]);
                var dX2V = MathOps.LinearBackward(lc.X2, w.Wv, dV[l][t]);
                for (var i = 0; i < nEmbd; i++)
                {
                    dX2[i] += dX2K[i] + dX2V[i];
                }

                var dXIn = MathOps.RmsNormBackward(lc.XIn, lc.S1, dX2);
                for (var i = 0; i < nEmbd; i++)
                {
                    dXIn[i] += dX4[i]; // attention residual branch
                }

                d = dXIn; // becomes dX7 of the layer below
            }

            var dX1 = d;
            var dX0 = MathOps.RmsNormBackward(c.X0, c.S0, dX1);
            var gte = _model.Wte.GradRow(c.TokenId);
            var gpe = _model.Wpe.GradRow(t);
            for (var i = 0; i < nEmbd; i++)
            {
                gte[i] += dX0[i];
                gpe[i] += dX0[i];
            }
        }

        return loss;
    }

    /// <summary>
    /// Forward-only mean loss over one document (no gradient side effects). Used for
    /// validation and gradient checking.
    /// </summary>
    public double EvaluateLoss(string document)
    {
        var sampler = new Sampler(_model, new Random(0));
        return sampler.ComputeLoss(document);
    }

    // ---------------- Adam update (same formula as the original) ----------------
    private void ApplyAdam()
    {
        var o = _options;
        var lrT = o.LearningRate * (1.0 - (double)_step / o.Steps);
        var bc1 = 1 - Math.Pow(o.Beta1, _step + 1);
        var bc2 = 1 - Math.Pow(o.Beta2, _step + 1);
        for (var ti = 0; ti < _model.Tensors.Length; ti++)
        {
            var tns = _model.Tensors[ti];
            var mm = _mBuf[ti];
            var vv = _vBuf[ti];
            for (var i = 0; i < tns.Data.Length; i++)
            {
                var g = tns.Grad[i];
                mm[i] = o.Beta1 * mm[i] + (1 - o.Beta1) * g;
                vv[i] = o.Beta2 * vv[i] + (1 - o.Beta2) * g * g;
                tns.Data[i] -= lrT * (mm[i] / bc1) / (Math.Sqrt(vv[i] / bc2) + o.Epsilon);
                tns.Grad[i] = 0;
            }
        }
    }

    /// <summary>
    /// Allocates a [layer][position] buffer array. When <paramref name="fillWidth"/> is given,
    /// each slot is pre-allocated with zeros (needed for dK/dV, where later positions write
    /// into earlier slots); otherwise slots are filled during the forward pass.
    /// </summary>
    private static double[][][] NewLayerPosBuffers(int nLayer, int n, int? fillWidth = null)
    {
        var buf = new double[nLayer][][];
        for (var l = 0; l < nLayer; l++)
        {
            buf[l] = new double[n][];
            if (fillWidth is { } width)
            {
                for (var t = 0; t < n; t++) buf[l][t] = new double[width];
            }
        }

        return buf;
    }
}
