namespace MicroGpt.Web.Services;

/// <summary>Tiny dependency-free CSV helpers: quoted-field parsing and delimiter detection.</summary>
public static class CsvParser
{
    private static readonly char[] Candidates = { ',', ';', '\t', '|' };

    /// <summary>Splits one CSV line, honoring double-quoted fields and escaped quotes ("").</summary>
    public static List<string> ParseLine(string line, char delimiter)
    {
        var fields = new List<string>();
        var sb = new System.Text.StringBuilder();
        var inQuotes = false;
        for (var i = 0; i < line.Length; i++)
        {
            var c = line[i];
            if (inQuotes)
            {
                if (c == '"')
                {
                    if (i + 1 < line.Length && line[i + 1] == '"') { sb.Append('"'); i++; }
                    else inQuotes = false;
                }
                else sb.Append(c);
            }
            else if (c == '"' && sb.Length == 0) inQuotes = true;
            else if (c == delimiter) { fields.Add(sb.ToString()); sb.Clear(); }
            else sb.Append(c);
        }

        fields.Add(sb.ToString());
        return fields;
    }

    /// <summary>
    /// Detects a delimiter from sample lines: a candidate qualifies when it yields the same
    /// field count (>1) on at least 80% of non-empty lines. Returns null for plain text.
    /// </summary>
    public static char? DetectDelimiter(IReadOnlyList<string> sampleLines)
    {
        var lines = sampleLines.Where(l => l.Length > 0).Take(30).ToList();
        if (lines.Count < 2) return null;

        foreach (var d in Candidates)
        {
            var counts = lines.Select(l => ParseLine(l, d).Count).ToList();
            var mode = counts.GroupBy(c => c).OrderByDescending(g => g.Count()).First();
            if (mode.Key > 1 && mode.Count() >= lines.Count * 0.8) return d;
        }

        return null;
    }
}
