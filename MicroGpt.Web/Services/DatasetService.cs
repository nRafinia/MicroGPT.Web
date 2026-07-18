using System.Text;
using MicroGpt.Web.Models;

namespace MicroGpt.Web.Services;

/// <summary>
/// Reads uploaded files as streams — the file is never fully materialized in memory.
/// Mirrors the CLI's approach: a seeded reservoir sample capped at MaxDocs, while the
/// character vocabulary and statistics are computed over the ENTIRE file.
/// </summary>
public sealed class DatasetService
{
    public const int LengthBuckets = 256; // lengths 0..254, last bucket = 255+
    private const int PreviewBytes = 64 * 1024;

    /// <summary>Reads only the first ~64 KB to detect format and estimate line count.</summary>
    public static async Task<FilePreview> PreviewAsync(Stream stream, string fileName, long fileSize)
    {
        var buffer = new byte[PreviewBytes];
        var read = 0;
        while (read < buffer.Length)
        {
            var n = await stream.ReadAsync(buffer.AsMemory(read, buffer.Length - read));
            if (n == 0) break;
            read += n;
        }

        var text = Encoding.UTF8.GetString(buffer, 0, read);
        var raw = text.Split('\n');
        // The last fragment is usually a cut-off line; drop it unless we read the whole file.
        var complete = read < fileSize ? raw[..^1] : raw;
        var lines = complete.Select(l => l.TrimEnd('\r')).Where(l => l.Length > 0).Take(30).ToList();

        var delimiter = CsvParser.DetectDelimiter(lines);
        var avg = complete.Length > 0 ? (double)read / complete.Length : 0;

        return new FilePreview
        {
            FileName = fileName,
            FileSizeBytes = fileSize,
            Lines = lines,
            Delimiter = delimiter,
            HeaderCandidates = delimiter is { } d && lines.Count > 0
                ? CsvParser.ParseLine(lines[0], d)
                : Array.Empty<string>(),
            AvgBytesPerLine = avg,
        };
    }

    /// <summary>
    /// Single streaming pass: reservoir-samples up to MaxDocs documents while accumulating
    /// full-file charset, character frequencies, and length distribution. Yields to the UI
    /// thread periodically so the page stays responsive on large files.
    /// </summary>
    public static async Task<Dataset> LoadAsync(
        Stream stream, string sourceName, LoadOptions options,
        Func<long, Task>? onProgress = null, CancellationToken ct = default)
    {
        var rng = new Random(options.Seed);
        var docs = new List<string>(Math.Min(options.MaxDocs, 4096));
        var charset = new HashSet<char>();
        var charFreq = new Dictionary<char, long>();
        var lengthCounts = new long[LengthBuckets];
        long totalLines = 0, skippedEmpty = 0, kept = 0, lengthSum = 0;
        var maxLength = 0;

        using var reader = new StreamReader(stream, Encoding.UTF8);
        var isFirst = true;
        while (await reader.ReadLineAsync(ct) is { } line)
        {
            totalLines++;
            if (isFirst)
            {
                isFirst = false;
                line = line.TrimStart('\uFEFF'); // strip UTF-8 BOM if present
                if (options.HasHeader) continue;
            }

            var value = options.ColumnIndex < 0
                ? line
                : ExtractColumn(line, options.Delimiter, options.ColumnIndex);

            value = value.Trim();

            if (value.Length == 0)
            {
                skippedEmpty++;
                continue;
            }

            foreach (var c in value)
            {
                charset.Add(c);
                charFreq[c] = charFreq.GetValueOrDefault(c) + 1;
            }

            lengthCounts[Math.Min(value.Length, LengthBuckets - 1)]++;
            lengthSum += value.Length;
            if (value.Length > maxLength) maxLength = value.Length;

            // Vitter's reservoir sampling, seeded — same guarantee as the CLI's --max-docs.
            if (kept < options.MaxDocs) docs.Add(value);
            else
            {
                var j = rng.Next((int)Math.Min(kept + 1, int.MaxValue));
                if (j < options.MaxDocs) docs[j] = value;
            }

            kept++;

            if (totalLines % 20_000 == 0)
            {
                if (onProgress is not null) await onProgress(totalLines);
                await Task.Delay(1, ct); // let the browser paint
            }
        }

        var topChars = charFreq.OrderByDescending(kv => kv.Value)
            .Take(20).Select(kv => (kv.Key, kv.Value)).ToList();

        return new Dataset
        {
            SourceName = sourceName,
            ColumnName = null,
            Docs = docs,
            TotalLines = totalLines,
            SkippedEmpty = skippedEmpty,
            Charset = string.Concat(charset.OrderBy(c => c)),
            TopChars = topChars,
            LengthCounts = lengthCounts,
            MaxLength = maxLength,
            AvgLength = kept > 0 ? (double)lengthSum / kept : 0,
        };
    }

    private static string ExtractColumn(string line, char delimiter, int index)
    {
        var fields = CsvParser.ParseLine(line, delimiter);
        return index < fields.Count ? fields[index] : string.Empty;
    }
}