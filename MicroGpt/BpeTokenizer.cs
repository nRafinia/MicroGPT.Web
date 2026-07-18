namespace MicroGpt;

/// <summary>
/// Character-level BPE tokenizer. Token ids: [0, BaseCount) are single characters
/// (ascending order), [BaseCount, BaseCount + merge count) are merged tokens in merge
/// order, and BOS is the last id. Training is deterministic: the most frequent adjacent
/// pair wins, ties broken by the smaller (left, right) ids.
/// </summary>
public sealed class BpeTokenizer : ITokenizer
{
    private readonly Dictionary<char, int> _charToId;
    private readonly Dictionary<(int Left, int Right), int> _mergeRank; // pair -> merge index
    private readonly string[] _tokenStrings; // id -> decoded string (content tokens only)

    /// <summary>Single-character base tokens, sorted ascending. Index == token id.</summary>
    public string BaseChars { get; }

    /// <summary>Merge rules in training order; merge i produces token id <c>BaseChars.Length + i</c>.</summary>
    public IReadOnlyList<(int Left, int Right)> Merges { get; }

    public int BosId => _tokenStrings.Length;

    public int Size => _tokenStrings.Length + 1;

    public BpeTokenizer(string baseChars, IReadOnlyList<(int Left, int Right)> merges)
    {
        if (string.IsNullOrEmpty(baseChars))
            throw new ArgumentException("Base vocabulary must contain at least one character.", nameof(baseChars));

        var distinct = baseChars.Distinct().OrderBy(c => c).ToArray();
        if (distinct.Length != baseChars.Length)
            throw new ArgumentException("Base characters must be unique.", nameof(baseChars));

        BaseChars = new string(distinct);
        _charToId = new Dictionary<char, int>(BaseChars.Length);
        for (var i = 0; i < BaseChars.Length; i++) _charToId[BaseChars[i]] = i;

        Merges = merges.ToArray();
        _mergeRank = new Dictionary<(int, int), int>(Merges.Count);
        _tokenStrings = new string[BaseChars.Length + Merges.Count];
        for (var i = 0; i < BaseChars.Length; i++) _tokenStrings[i] = BaseChars[i].ToString();
        for (var m = 0; m < Merges.Count; m++)
        {
            var (l, r) = Merges[m];
            var id = BaseChars.Length + m;
            if (l < 0 || l >= id || r < 0 || r >= id)
                throw new ArgumentException($"Merge {m} references token ids ({l},{r}) that do not exist yet.");
            _tokenStrings[id] = _tokenStrings[l] + _tokenStrings[r];
            _mergeRank[(l, r)] = m;
        }
    }

    /// <summary>
    /// Trains a BPE tokenizer on the corpus: repeatedly merges the most frequent adjacent
    /// token pair, up to <paramref name="numMerges"/> times or until no pair occurs at
    /// least <paramref name="minFrequency"/> times. By default digits are kept atomic
    /// (never merged), which keeps numeric fields like codes and phone numbers stable —
    /// pass <paramref name="mergeDigits"/> = true to disable that protection.
    /// </summary>
    public static BpeTokenizer Train(
        IReadOnlyList<string> documents, int numMerges, int minFrequency = 2, bool mergeDigits = false,
        string? baseCharsOverride = null)
    {
        if (numMerges < 0) throw new ArgumentOutOfRangeException(nameof(numMerges));
        if (minFrequency < 2) throw new ArgumentOutOfRangeException(nameof(minFrequency));

        var baseChars = baseCharsOverride is null
            ? string.Concat(string.Concat(documents).Distinct().OrderBy(c => c))
            : string.Concat(baseCharsOverride.Distinct().OrderBy(c => c));
        var charToId = baseChars.Select((c, i) => (c, i)).ToDictionary(t => t.c, t => t.i);

        // token id -> whether its decoded string contains a digit (digit tokens stay atomic by default)
        var hasDigit = new List<bool>(baseChars.Select(char.IsDigit));

        var seqs = documents
            .Select(d => d.Select(c => charToId[c]).ToList())
            .ToList();

        var merges = new List<(int Left, int Right)>(numMerges);
        for (var m = 0; m < numMerges; m++)
        {
            // count adjacent pairs across the whole corpus
            var counts = new Dictionary<(int, int), int>();
            foreach (var seq in seqs)
            {
                for (var i = 0; i < seq.Count - 1; i++)
                {
                    var pair = (seq[i], seq[i + 1]);
                    counts[pair] = counts.GetValueOrDefault(pair) + 1;
                }
            }

            if (counts.Count == 0) break;

            // deterministic winner: highest count, then smallest left id, then smallest right id
            var best = default((int Left, int Right));
            var bestCount = -1;
            foreach (var (pair, count) in counts)
            {
                if (!mergeDigits && (hasDigit[pair.Item1] || hasDigit[pair.Item2])) continue;
                if (count > bestCount ||
                    (count == bestCount && (pair.Item1 < best.Left ||
                                            (pair.Item1 == best.Left && pair.Item2 < best.Right))))
                {
                    best = pair;
                    bestCount = count;
                }
            }

            if (bestCount < minFrequency) break;

            var newId = baseChars.Length + merges.Count;
            merges.Add(best);
            hasDigit.Add(hasDigit[best.Left] || hasDigit[best.Right]);
            foreach (var seq in seqs)
            {
                MergePairInPlace(seq, best, newId);
            }
        }

        return new BpeTokenizer(baseChars, merges);
    }

    public int CountTokens(string document) => Encode(document).Count;

    public int[] EncodeWithBos(string document)
    {
        var content = Encode(document);
        var tokens = new int[content.Count + 2];
        tokens[0] = BosId;
        for (var i = 0; i < content.Count; i++) tokens[i + 1] = content[i];
        tokens[^1] = BosId;
        return tokens;
    }

    /// <summary>Encodes a string into content token ids (no BOS).</summary>
    public List<int> Encode(string text)
    {
        var seq = new List<int>(text.Length);
        foreach (var c in text)
        {
            seq.Add(_charToId.TryGetValue(c, out var id)
                ? id
                : throw new ArgumentException($"Character '{c}' (U+{(int)c:X4}) is not in the vocabulary."));
        }

        // repeatedly apply the earliest-trained merge present in the sequence
        while (seq.Count >= 2)
        {
            var bestRank = int.MaxValue;
            for (var i = 0; i < seq.Count - 1; i++)
            {
                if (_mergeRank.TryGetValue((seq[i], seq[i + 1]), out var rank) && rank < bestRank)
                {
                    bestRank = rank;
                }
            }

            if (bestRank == int.MaxValue) break;

            MergePairInPlace(seq, Merges[bestRank], BaseChars.Length + bestRank);
        }

        return seq;
    }

    public string Decode(IReadOnlyList<int> tokenIds)
    {
        var sb = new System.Text.StringBuilder();
        foreach (var id in tokenIds)
        {
            if (id < 0 || id >= _tokenStrings.Length)
                throw new ArgumentOutOfRangeException(nameof(tokenIds), $"Token id {id} is not a content token (vocab size {Size}).");
            sb.Append(_tokenStrings[id]);
        }

        return sb.ToString();
    }

    /// <summary>Decoded string of a single token id (useful for inspection).</summary>
    public string TokenString(int id) => _tokenStrings[id];

    private static void MergePairInPlace(List<int> seq, (int Left, int Right) pair, int newId)
    {
        var write = 0;
        var read = 0;
        while (read < seq.Count)
        {
            if (read < seq.Count - 1 && seq[read] == pair.Left && seq[read + 1] == pair.Right)
            {
                seq[write++] = newId;
                read += 2;
            }
            else
            {
                seq[write++] = seq[read++];
            }
        }

        seq.RemoveRange(write, seq.Count - write);
    }
}
