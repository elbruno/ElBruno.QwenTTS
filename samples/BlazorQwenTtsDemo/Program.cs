using BlazorQwenTtsDemo.Components;
using BlazorQwenTtsDemo.Services;
using ElBruno.QwenTTS.BlazorComponents.Extensions;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container.
builder.Services.AddRazorComponents()
    .AddInteractiveServerComponents();
builder.Services.AddQwenTtsBlazorComponents();
builder.Services.AddSingleton<BaseTtsModelDownloadAdapter>();
builder.Services.AddSingleton<VoiceCloningModelDownloadAdapter>();
builder.Services.AddSingleton<BaseTtsModelDownloadController>();
builder.Services.AddSingleton<VoiceCloningModelDownloadController>();
builder.AddServiceDefaults();

var app = builder.Build();

// Configure the HTTP request pipeline.
if (!app.Environment.IsDevelopment())
{
    app.UseExceptionHandler("/Error", createScopeForErrors: true);
    // The default HSTS value is 30 days. You may want to change this for production scenarios, see https://aka.ms/aspnetcore-hsts.
    app.UseHsts();
}

app.UseHttpsRedirection();

app.UseStaticFiles();
app.UseAntiforgery();
app.MapDefaultEndpoints();

app.MapRazorComponents<App>()
    .AddInteractiveServerRenderMode();

app.Run();
