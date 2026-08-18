var builder = DistributedApplication.CreateBuilder(args);

builder.AddProject("blazor-qwen-tts-demo", @"..\BlazorQwenTtsDemo\BlazorQwenTtsDemo.csproj")
    .WithExternalHttpEndpoints();

builder.Build().Run();
