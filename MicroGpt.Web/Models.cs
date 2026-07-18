namespace MicroGpt.Web.Models;

/// <summary>What the first ~64 KB of an uploaded file looks like, before the full pass.</summary>
public sealed record FilePreview
{
    public required string FileName { get; init; }
    public required long FileSizeBytes { get; init; }
    public required IReadOnlyList<string> Lines { get; init; }
    public char? Delimiter { get; init; }
    public bool LooksLikeCsv => Delimiter is not null;
    public IReadOnlyList<string> HeaderCandidates { get; init; } = Array.Empty<string>();
    public double AvgBytesPerLine { get; init; }
    public long EstimatedTotalLines => AvgBytesPerLine > 0 ? (long)(FileSizeBytes / AvgBytesPerLine) : 0;
}

/// <summary>Options for the single streaming pass over the file.</summary>
public sealed record LoadOptions
{
    /// <summary>-1 = use the whole line; otherwise the zero-based CSV column index.</summary>
    public int ColumnIndex { get; init; } = -1;
    public char Delimiter { get; init; } = ',';
    public bool HasHeader { get; init; }
    /// <summary>Reservoir-sample cap. Memory scales with this, not with file size.</summary>
    public int MaxDocs { get; init; } = 50_000;
    public int Seed { get; init; } = 42;
}

/// <summary>Result of the streaming pass: a seeded reservoir sample plus full-file statistics.</summary>
public sealed class Dataset
{
    public required string SourceName { get; init; }
    public required string? ColumnName { get; init; }
    public required List<string> Docs { get; init; }
    public required long TotalLines { get; init; }
    public required long SkippedEmpty { get; init; }
    /// <summary>Distinct characters of the ENTIRE file (not just the sample), sorted ascending.</summary>
    public required string Charset { get; init; }
    /// <summary>Character frequency over the entire file.</summary>
    public required IReadOnlyList<(char Ch, long Count)> TopChars { get; init; }
    /// <summary>lengthCounts[i] = number of lines of length i; last bucket is overflow.</summary>
    public required long[] LengthCounts { get; init; }
    public required int MaxLength { get; init; }
    public required double AvgLength { get; init; }
    public bool WasSampled => TotalLines - SkippedEmpty > Docs.Count;
}

public sealed record TrainSettings
{
    public int EmbeddingDim { get; init; } = 16;
    public int NumHeads { get; init; } = 4;
    public int NumLayers { get; init; } = 1;
    public int Steps { get; init; } = 4000;
    public double LearningRate { get; init; } = 0.01;
    public int Seed { get; init; } = 42;
    public string TokenizerKind { get; init; } = "char"; // "char" | "bpe"
    public int BpeMerges { get; init; } = 256;
    /// <summary>Fraction of docs held out for loss calibration (capped at 1000 docs).</summary>
    public double HoldoutFraction { get; init; } = 0.05;
}

public sealed record ScoreRow(string Text, double Loss, bool OutOfVocab);

public sealed record LossPoint(int Step, double Loss, double Ema);

/// <summary>Percentile thresholds computed on held-out clean data after training.</summary>
public sealed record Calibration(int HoldoutCount, double P50, double P90, double P95, double P99)
{
    public double AtPercentile(double p) => p switch
    {
        <= 50 => P50,
        <= 90 => P90,
        <= 95 => P95,
        _ => P99,
    };
}
