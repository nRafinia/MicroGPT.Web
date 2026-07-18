using Microsoft.AspNetCore.Components.WebAssembly.Hosting;
using MicroGpt.Web;
using MicroGpt.Web.Services;

var builder = WebAssemblyHostBuilder.CreateDefault(args);
builder.RootComponents.Add<Main>("#app");

builder.Services.AddSingleton(sp => new HttpClient { BaseAddress = new Uri(builder.HostEnvironment.BaseAddress) });
builder.Services.AddSingleton<AppState>();
builder.Services.AddSingleton<DatasetService>();
builder.Services.AddSingleton<TrainingSession>();

await builder.Build().RunAsync();
