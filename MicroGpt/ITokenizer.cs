namespace MicroGpt;

/// <summary>
/// Maps documents to/from token id sequences. The model itself is tokenizer-agnostic;
/// the BOS token id is always <c>Size - 1</c> by convention.
/// </summary>
public interface ITokenizer
{
    /// <summary>Total vocabulary size, including the BOS token.</summary>
    int Size { get; }

    /// <summary>Token id of the BOS/EOS marker (always <c>Size - 1</c>).</summary>
    int BosId { get; }

    /// <summary>Encodes a document as BOS + content tokens + BOS.</summary>
    int[] EncodeWithBos(string document);

    /// <summary>Number of content tokens the document encodes to (without BOS wrappers).</summary>
    int CountTokens(string document);

    /// <summary>Decodes content token ids (no BOS) back into a string.</summary>
    string Decode(IReadOnlyList<int> tokenIds);
}
