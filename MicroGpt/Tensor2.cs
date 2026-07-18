namespace MicroGpt;

/// <summary>
/// A simple 2D tensor: data + gradient (a pair of flat arrays instead of thousands of Value nodes).
/// Row-major layout.
/// </summary>
public sealed class Tensor2
{
    public readonly int Rows, Cols;
    public readonly double[] Data; // row-major
    public readonly double[] Grad;

    public Tensor2(int rows, int cols)
    {
        Rows = rows;
        Cols = cols;
        Data = new double[rows * cols];
        Grad = new double[rows * cols];
    }

    public Span<double> Row(int r) => Data.AsSpan(r * Cols, Cols);
    public Span<double> GradRow(int r) => Grad.AsSpan(r * Cols, Cols);
}
