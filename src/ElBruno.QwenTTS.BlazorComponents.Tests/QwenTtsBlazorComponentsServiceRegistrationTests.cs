using System.Reflection;
using ElBruno.QwenTTS.BlazorComponents.Tests.TestHelpers;
using Microsoft.Extensions.DependencyInjection;

namespace ElBruno.QwenTTS.BlazorComponents.Tests;

public class QwenTtsBlazorComponentsServiceRegistrationTests
{
    [Fact]
    public void AddQwenTtsBlazorComponents_RegistersServicesAndReturnsSameCollection()
    {
        var assembly = BlazorComponentsTestHelpers.RequireBlazorComponentsAssembly();
        var extensionType = assembly.GetType("ElBruno.QwenTTS.BlazorComponents.Extensions.QwenTtsBlazorComponentsServiceExtensions")
            ?? throw new Xunit.Sdk.XunitException("Service extension class 'QwenTtsBlazorComponentsServiceExtensions' was not found.");

        var method = extensionType.GetMethod(
            "AddQwenTtsBlazorComponents",
            BindingFlags.Public | BindingFlags.Static,
            binder: null,
            types: [typeof(IServiceCollection)],
            modifiers: null);

        Assert.NotNull(method);

        var services = new ServiceCollection();
        var countBefore = services.Count;

        var result = method!.Invoke(null, [services]);

        Assert.Same(services, result);
        Assert.True(services.Count > countBefore, "AddQwenTtsBlazorComponents should register at least one service.");
    }
}
