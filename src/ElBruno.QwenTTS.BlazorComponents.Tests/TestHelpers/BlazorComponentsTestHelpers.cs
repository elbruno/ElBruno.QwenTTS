using System.Reflection;
using Microsoft.AspNetCore.Components;
using Xunit.Sdk;

namespace ElBruno.QwenTTS.BlazorComponents.Tests.TestHelpers;

internal static class BlazorComponentsTestHelpers
{
    internal const string AssemblyName = "ElBruno.QwenTTS.BlazorComponents";

    internal static Assembly RequireBlazorComponentsAssembly()
    {
        var loaded = AppDomain.CurrentDomain.GetAssemblies()
            .FirstOrDefault(a => string.Equals(a.GetName().Name, AssemblyName, StringComparison.Ordinal));
        if (loaded is not null) return loaded;

        try
        {
            return Assembly.Load(AssemblyName);
        }
        catch
        {
            throw new XunitException($"{AssemblyName} could not be loaded. Ensure the Blazor components project is part of the solution.");
        }
    }

    internal static Type RequireType(string fullName)
    {
        var type = RequireBlazorComponentsAssembly().GetType(fullName, throwOnError: false);
        return type ?? throw new XunitException($"Type '{fullName}' not found.");
    }

    internal static PropertyInfo RequireParameterProperty(Type componentType, string propertyName)
    {
        var property = componentType.GetProperty(propertyName);
        Assert.NotNull(property);
        var hasParameterAttribute = property!.GetCustomAttribute<ParameterAttribute>() is not null;
        Assert.True(hasParameterAttribute, $"{componentType.Name}.{propertyName} must be decorated with [Parameter].");
        return property;
    }

    internal static object CreateNoOpProxy(Type interfaceType)
    {
        if (!interfaceType.IsInterface)
            throw new ArgumentException($"{interfaceType.FullName} is not an interface.");

        var method = typeof(DispatchProxy).GetMethods(BindingFlags.Public | BindingFlags.Static)
            .First(m => m.Name == nameof(DispatchProxy.Create) &&
                        m.IsGenericMethodDefinition &&
                        m.GetGenericArguments().Length == 2 &&
                        m.GetParameters().Length == 0);
        var generic = method!.MakeGenericMethod(interfaceType, typeof(NoOpDispatchProxy));
        return generic.Invoke(null, null)!;
    }

    private class NoOpDispatchProxy : DispatchProxy
    {
        protected override object? Invoke(MethodInfo? targetMethod, object?[]? args)
        {
            if (targetMethod is null) return null;
            var returnType = targetMethod.ReturnType;

            if (returnType == typeof(void))
                return null;

            if (returnType == typeof(Task))
                return Task.CompletedTask;

            if (returnType.IsGenericType && returnType.GetGenericTypeDefinition() == typeof(Task<>))
            {
                var resultType = returnType.GetGenericArguments()[0];
                var fromResult = typeof(Task).GetMethod(nameof(Task.FromResult))!.MakeGenericMethod(resultType);
                var defaultValue = resultType.IsValueType ? Activator.CreateInstance(resultType) : null;
                return fromResult.Invoke(null, [defaultValue]);
            }

            return returnType.IsValueType ? Activator.CreateInstance(returnType) : null;
        }
    }
}
