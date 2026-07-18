namespace MicroGpt;

/// <summary>
/// Architecture hyperparameters. Immutable; serialized inside the model file.
/// </summary>
public sealed record ModelConfig
{
    /// <summary>Number of transformer blocks.</summary>
    public required int NumLayers { get; init; }

    /// <summary>Embedding dimension (must be divisible by <see cref="NumHeads"/>).</summary>
    public required int EmbeddingDim { get; init; }

    /// <summary>Number of attention heads.</summary>
    public required int NumHeads { get; init; }

    /// <summary>Maximum sequence length (also the size of the positional embedding table).</summary>
    public required int BlockSize { get; init; }

    /// <summary>Std-dev used for Gaussian weight initialization.</summary>
    public double InitStd { get; init; } = 0.08;

    public int HeadDim => EmbeddingDim / NumHeads;

    public void Validate()
    {
        if (NumLayers < 1) throw new ArgumentOutOfRangeException(nameof(NumLayers), "At least one layer is required.");
        if (NumHeads < 1) throw new ArgumentOutOfRangeException(nameof(NumHeads), "At least one head is required.");
        if (EmbeddingDim < 1) throw new ArgumentOutOfRangeException(nameof(EmbeddingDim));
        if (BlockSize < 1) throw new ArgumentOutOfRangeException(nameof(BlockSize));
        if (EmbeddingDim % NumHeads != 0)
            throw new ArgumentException($"EmbeddingDim ({EmbeddingDim}) must be divisible by NumHeads ({NumHeads}).");
    }

    /// <summary>The defaults of the original micro implementation (1 layer, 16-dim, 4 heads).</summary>
    public static ModelConfig Micro(int blockSize) => new()
    {
        NumLayers = 1,
        EmbeddingDim = 16,
        NumHeads = 4,
        BlockSize = blockSize,
    };
}
