namespace MicroGpt;

/// <summary>
/// The math kernels of the original implementation, unchanged. Kept internal so the
/// public surface stays small; Trainer and Sampler share them.
/// </summary>
internal static class MathOps
{
    /// <summary>Box–Muller Gaussian sample. Consumes exactly two values from <paramref name="rng"/>.</summary>
    public static double Gauss(Random rng, double mu, double sigma)
    {
        var u1 = 1.0 - rng.NextDouble();
        var u2 = rng.NextDouble();
        return mu + sigma * Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
    }

    // y = W x (forward)
    public static double[] Linear(double[] x, Tensor2 w)
    {
        var y = new double[w.Rows];
        for (var o = 0; o < w.Rows; o++)
        {
            double s = 0;
            var row = w.Row(o);
            for (var i = 0; i < x.Length; i++)
            {
                s += row[i] * x[i]; // same summation order as the scalar version
            }

            y[o] = s;
        }

        return y;
    }

    // backward for y = Wx:  dW += dy ⊗ x  and  dx = Wᵀ dy
    public static double[] LinearBackward(double[] x, Tensor2 w, double[] dy)
    {
        var dx = new double[x.Length];
        for (var o = 0; o < w.Rows; o++)
        {
            var g = dy[o];
            var row = w.Row(o);
            var grow = w.GradRow(o);
            for (var i = 0; i < x.Length; i++)
            {
                grow[i] += g * x[i];
                dx[i] += row[i] * g;
            }
        }

        return dx;
    }

    // rmsnorm forward: y_i = x_i * s ,  s = (mean(x²)+1e-5)^-½  — returns s for backward
    public static (double[] y, double s) RmsNorm(double[] x)
    {
        double ms = 0;
        for (var i = 0; i < x.Length; i++)
        {
            ms += x[i] * x[i];
        }

        ms /= x.Length;
        var s = Math.Pow(ms + 1e-5, -0.5);
        var y = new double[x.Length];
        for (var i = 0; i < x.Length; i++)
        {
            y[i] = x[i] * s;
        }

        return (y, s);
    }

    // rmsnorm backward (analytical derivative):
    // dx_i = s·dy_i − (x_i·s³/n)·Σ_j dy_j·x_j
    public static double[] RmsNormBackward(double[] x, double s, double[] dy)
    {
        var n = x.Length;
        double dot = 0;
        for (var j = 0; j < n; j++) dot += dy[j] * x[j];
        var s3 = s * s * s;
        var dx = new double[n];
        for (var i = 0; i < n; i++)
        {
            dx[i] = s * dy[i] - x[i] * s3 * dot / n;
        }

        return dx;
    }

    // softmax (forward only — during training it is fused with cross-entropy: dlogits = p − onehot)
    public static double[] Softmax(double[] logits)
    {
        var maxVal = logits.Max();
        var e = new double[logits.Length];
        double total = 0;
        for (var i = 0; i < logits.Length; i++)
        {
            e[i] = Math.Exp(logits[i] - maxVal);
            total += e[i];
        }

        for (var i = 0; i < logits.Length; i++)
        {
            e[i] /= total;
        }

        return e;
    }
}
