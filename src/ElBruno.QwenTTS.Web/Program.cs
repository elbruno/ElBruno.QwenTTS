using ElBruno.QwenTTS.Web.Components;
using ElBruno.QwenTTS.Web.Services;

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

app.MapStaticAssets();
app.UseStaticFiles();
app.MapRazorComponents<App>()
    .AddInteractiveServerRenderMode();

app.Run();
