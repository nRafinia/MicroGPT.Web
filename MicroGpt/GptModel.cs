using System.Text;

namespace MicroGpt;

/// <summary>
/// The model: config + tokenizer + named weight tensors, with binary save/load.
/// Weight names and construction order match the original implementation exactly
/// (wte, wpe, lm_head, then per layer: attn_wq, attn_wk, attn_wv, attn_wo, mlp_fc1, mlp_fc2),
/// so initialization from the same seed reproduces the same values.
/// </summary>
public sealed class GptModel
{
    private const string Magic = "MGPT";
    private const int FormatVersion = 2;

    private const int TokenizerKindChar = 0;
    private const int TokenizerKindBpe = 1;

    public ModelConfig Config { get; }
    public ITokenizer Tokenizer { get; }

    /// <summary>Named parameters, in creation order.</summary>
    public IReadOnlyDictionary<string, Tensor2> StateDict => _stateDict;

    /// <summary>All parameter tensors in a stable order (used by the optimizer).</summary>
    public Tensor2[] Tensors { get; }

    public long ParameterCount => Tensors.Sum(t => (long)t.Data.Length);

    internal Tensor2 Wte { get; }
    internal Tensor2 Wpe { get; }
    internal Tensor2 LmHead { get; }
    internal LayerWeights[] Layers { get; }

    internal sealed record LayerWeights(Tensor2 Wq, Tensor2 Wk, Tensor2 Wv, Tensor2 Wo, Tensor2 Fc1, Tensor2 Fc2);

    private readonly Dictionary<string, Tensor2> _stateDict;

    private GptModel(ModelConfig config, ITokenizer tokenizer, Dictionary<string, Tensor2> stateDict)
    {
        config.Validate();
        Config = config;
        Tokenizer = tokenizer;
        _stateDict = stateDict;
        Tensors = stateDict.Values.ToArray();

        Wte = stateDict["wte"];
        Wpe = stateDict["wpe"];
        LmHead = stateDict["lm_head"];
        Layers = new LayerWeights[config.NumLayers];
        for (var l = 0; l < config.NumLayers; l++)
        {
            Layers[l] = new LayerWeights(
                stateDict[$"layer{l}.attn_wq"],
                stateDict[$"layer{l}.attn_wk"],
                stateDict[$"layer{l}.attn_wv"],
                stateDict[$"layer{l}.attn_wo"],
                stateDict[$"layer{l}.mlp_fc1"],
                stateDict[$"layer{l}.mlp_fc2"]);
        }
    }

    /// <summary>Creates a freshly initialized model. The RNG is consumed in the same order as the original code.</summary>
    public static GptModel CreateNew(ModelConfig config, ITokenizer tokenizer, Random rng)
    {
        config.Validate();
        var d = config.EmbeddingDim;

        Tensor2 Matrix(int nout, int nin)
        {
            var t = new Tensor2(nout, nin);
            for (var i = 0; i < t.Data.Length; i++) t.Data[i] = MathOps.Gauss(rng, 0, config.InitStd);
            return t;
        }

        var sd = new Dictionary<string, Tensor2>
        {
            ["wte"] = Matrix(tokenizer.Size, d),
            ["wpe"] = Matrix(config.BlockSize, d),
            ["lm_head"] = Matrix(tokenizer.Size, d),
        };
        for (var l = 0; l < config.NumLayers; l++)
        {
            sd[$"layer{l}.attn_wq"] = Matrix(d, d);
            sd[$"layer{l}.attn_wk"] = Matrix(d, d);
            sd[$"layer{l}.attn_wv"] = Matrix(d, d);
            sd[$"layer{l}.attn_wo"] = Matrix(d, d);
            sd[$"layer{l}.mlp_fc1"] = Matrix(4 * d, d);
            sd[$"layer{l}.mlp_fc2"] = Matrix(d, 4 * d);
        }

        return new GptModel(config, tokenizer, sd);
    }

    public static GptModel CreateNew(ModelConfig config, ITokenizer tokenizer, int seed = 42) =>
        CreateNew(config, tokenizer, new Random(seed));

    // ---------------- Serialization ----------------

    public void Save(string path)
    {
        using var fs = File.Create(path);
        Save(fs);
    }

    public void Save(Stream stream)
    {
        using var w = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true);
        w.Write(Encoding.ASCII.GetBytes(Magic));
        w.Write(FormatVersion);

        // config
        w.Write(Config.NumLayers);
        w.Write(Config.EmbeddingDim);
        w.Write(Config.NumHeads);
        w.Write(Config.BlockSize);
        w.Write(Config.InitStd);

        // tokenizer
        switch (Tokenizer)
        {
            case CharTokenizer ct:
                w.Write(TokenizerKindChar);
                w.Write(ct.Chars);
                break;
            case BpeTokenizer bt:
                w.Write(TokenizerKindBpe);
                w.Write(bt.BaseChars);
                w.Write(bt.Merges.Count);
                foreach (var (l, r) in bt.Merges)
                {
                    w.Write(l);
                    w.Write(r);
                }

                break;
            default:
                throw new NotSupportedException($"Tokenizer type {Tokenizer.GetType().Name} cannot be serialized.");
        }

        // tensors, in creation order
        w.Write(_stateDict.Count);
        foreach (var (name, t) in _stateDict)
        {
            w.Write(name);
            w.Write(t.Rows);
            w.Write(t.Cols);
            foreach (var v in t.Data) w.Write(v);
        }
    }

    public static GptModel Load(string path)
    {
        using var fs = File.OpenRead(path);
        return Load(fs);
    }

    public static GptModel Load(Stream stream)
    {
        using var r = new BinaryReader(stream, Encoding.UTF8, leaveOpen: true);

        var magic = Encoding.ASCII.GetString(r.ReadBytes(4));
        if (magic != Magic) throw new InvalidDataException("Not a MicroGpt model file.");
        var version = r.ReadInt32();
        if (version is < 1 or > FormatVersion)
            throw new InvalidDataException($"Unsupported model format version {version}.");

        var config = new ModelConfig
        {
            NumLayers = r.ReadInt32(),
            EmbeddingDim = r.ReadInt32(),
            NumHeads = r.ReadInt32(),
            BlockSize = r.ReadInt32(),
            InitStd = r.ReadDouble(),
        };

        ITokenizer tokenizer;
        if (version == 1)
        {
            // v1 files always contain a character vocabulary
            tokenizer = new CharTokenizer(r.ReadString());
        }
        else
        {
            var kind = r.ReadInt32();
            tokenizer = kind switch
            {
                TokenizerKindChar => new CharTokenizer(r.ReadString()),
                TokenizerKindBpe => ReadBpe(r),
                _ => throw new InvalidDataException($"Unknown tokenizer kind {kind}."),
            };
        }

        var count = r.ReadInt32();
        var sd = new Dictionary<string, Tensor2>(count);
        for (var i = 0; i < count; i++)
        {
            var name = r.ReadString();
            var rows = r.ReadInt32();
            var cols = r.ReadInt32();
            var t = new Tensor2(rows, cols);
            for (var j = 0; j < t.Data.Length; j++) t.Data[j] = r.ReadDouble();
            sd[name] = t;
        }

        return new GptModel(config, tokenizer, sd);

        static BpeTokenizer ReadBpe(BinaryReader r)
        {
            var baseChars = r.ReadString();
            var mergeCount = r.ReadInt32();
            var merges = new (int, int)[mergeCount];
            for (var i = 0; i < mergeCount; i++)
            {
                merges[i] = (r.ReadInt32(), r.ReadInt32());
            }

            return new BpeTokenizer(baseChars, merges);
        }
    }
}
