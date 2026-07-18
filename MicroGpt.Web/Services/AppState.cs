using Microsoft.JSInterop;
using MicroGpt.Web.Models;

namespace MicroGpt.Web.Services;

/// <summary>Shared session state: the loaded dataset and the current model. Lives only in this tab.</summary>
public sealed class AppState
{
    public Dataset? Dataset { get; private set; }
    public GptModel? Model { get; private set; }
    public TrainSettings TrainSettings { get; set; } = new();

    /// <summary>"trained" when produced in this tab, "uploaded" when loaded from a .mgpt file.</summary>
    public string ModelOrigin { get; private set; } = "";

    public event Action? Changed;

    public void SetDataset(Dataset? dataset)
    {
        Dataset = dataset;
        Changed?.Invoke();
    }

    public void SetModel(GptModel? model, string origin)
    {
        Model = model;
        ModelOrigin = origin;
        Changed?.Invoke();
    }

    public void Notify() => Changed?.Invoke();

    public static async Task DownloadBytesAsync(IJSRuntime js, string fileName, byte[] bytes, string mime)
        => await js.InvokeVoidAsync("appInterop.downloadFile", fileName, Convert.ToBase64String(bytes), mime);

    public static async Task DownloadTextAsync(IJSRuntime js, string fileName, string text, string mime = "text/plain")
        => await DownloadBytesAsync(js, fileName, System.Text.Encoding.UTF8.GetBytes(text), mime);
}
