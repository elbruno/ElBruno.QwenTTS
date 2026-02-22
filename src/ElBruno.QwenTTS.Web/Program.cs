using ElBruno.QwenTTS.Web.Components;
using ElBruno.QwenTTS.Web.Services;
using Microsoft.Extensions.FileProviders;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container.
builder.Services.AddRazorComponents()
    .AddInteractiveServerComponents();
builder.Services.AddSignalR(options =>
{
    // Allow large JS interop messages (recorded audio returned as base64)
    options.MaximumReceiveMessageSize = 2 * 1024 * 1024; // 2 MB
});
builder.Services.AddSingleton<TtsPipelineService>();
builder.Services.AddSingleton<VoiceClonePipelineService>();

var app = builder.Build();

// Configure the HTTP request pipeline.
if (!app.Environment.IsDevelopment())
{
    app.UseExceptionHandler("/Error", createScopeForErrors: true);
}
app.UseAntiforgery();

// Serve runtime-generated files (WAV audio, references) via dedicated static file providers.
// These are created at runtime and not in the compile-time manifest, so they must be
// served by UseStaticFiles before MapStaticAssets to avoid manifest warnings.
var webRootPath = app.Environment.WebRootPath;
var generatedDir = Path.Combine(webRootPath, "generated");
var referencesDir = Path.Combine(webRootPath, "references");
Directory.CreateDirectory(generatedDir);
Directory.CreateDirectory(referencesDir);

app.UseStaticFiles(new StaticFileOptions
{
    FileProvider = new PhysicalFileProvider(generatedDir),
    RequestPath = "/generated"
});
app.UseStaticFiles(new StaticFileOptions
{
    FileProvider = new PhysicalFileProvider(referencesDir),
    RequestPath = "/references"
});

app.MapStaticAssets();
app.MapRazorComponents<App>()
    .AddInteractiveServerRenderMode();

app.Run();
