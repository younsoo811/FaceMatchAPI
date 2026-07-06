using FaceMatchAPI.Middleware;
using FaceMatchAPI.Services;

var builder = WebApplication.CreateBuilder(args);

builder.Logging.ClearProviders();
builder.Logging.AddConsole();
builder.Logging.AddDebug();

// Add services to the container.
builder.Services.AddSingleton<MongoService>();
builder.Services.AddSingleton<FaceService>();

builder.Services.AddControllers();
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

var app = builder.Build();

var appLogger = app.Services.GetRequiredService<ILogger<Program>>();

AppDomain.CurrentDomain.UnhandledException += (sender, e) =>
{
    var ex = e.ExceptionObject as Exception;
    appLogger.LogCritical(ex,
        "[AppDomain.UnhandledException] 치명적인 미처리 예외 발생. IsTerminating={IsTerminating}",
        e.IsTerminating);
};

TaskScheduler.UnobservedTaskException += (sender, e) =>
{
    appLogger.LogError(e.Exception,
        "[TaskScheduler.UnobservedTaskException] Task에서 관찰되지 않은 예외 발생.");
    e.SetObserved(); // 프로세스 종료 방지
};

app.UseGlobalExceptionHandler();

app.UseSwagger();
app.UseSwaggerUI();

app.UseAuthorization();
app.MapControllers();

app.Run();