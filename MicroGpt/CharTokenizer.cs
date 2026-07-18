namespace MicroGpt;

/// <summary>
/// Character-level tokenizer (the original scheme): every distinct character is a token,
/// BOS is the last id. Token ids are assigned by ascending character order.
/// </summary>
public sealed class CharTokenizer : ITokenizer
{
    private readonly Dictionary<char, int> _charToId;

    /// <summary>All characters in the vocabulary, sorted ascending. Index == token id.</summary>
    public string Chars { get; }

    public int BosId => Chars.Length;

    public int Size => Chars.Length + 1;

    public CharTokenizer(string chars)
    {
        if (string.IsNullOrEmpty(chars))
            throw new ArgumentException("Vocabulary must contain at least one character.", nameof(chars));

        var distinct = chars.Distinct().OrderBy(c => c).ToArray();
        if (distinct.Length != chars.Length)
            throw new ArgumentException("Vocabulary characters must be unique.", nameof(chars));

        Chars = new string(distinct);
        _charToId = new Dictionary<char, int>(Chars.Length);
        for (var i = 0; i < Chars.Length; i++) _charToId[Chars[i]] = i;
    }

    /// <summary>Builds a tokenizer from the distinct characters of the given documents.</summary>
    public static CharTokenizer Build(IEnumerable<string> documents) =>
        new(string.Concat(string.Concat(documents).Distinct()));

    public bool Contains(char c) => _charToId.ContainsKey(c);

    public int GetId(char c) =>
        _charToId.TryGetValue(c, out var id)
            ? id
            : throw new ArgumentException($"Character '{c}' (U+{(int)c:X4}) is not in the vocabulary.");

    public char GetChar(int id) =>
        id >= 0 && id < Chars.Length
            ? Chars[id]
            : throw new ArgumentOutOfRangeException(nameof(id), $"Token id {id} is not a character (vocab size {Size}).");

    public int CountTokens(string document) => document.Length;

    public int[] EncodeWithBos(string document)
    {
        var tokens = new int[document.Length + 2];
        tokens[0] = BosId;
        for (var i = 0; i < document.Length; i++) tokens[i + 1] = GetId(document[i]);
        tokens[^1] = BosId;
        return tokens;
    }

    public string Decode(IReadOnlyList<int> tokenIds)
    {
        var chars = new char[tokenIds.Count];
        for (var i = 0; i < tokenIds.Count; i++) chars[i] = GetChar(tokenIds[i]);
        return new string(chars);
    }
}
