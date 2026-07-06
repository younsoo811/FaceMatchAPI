using FaceMatchAPI.Dtos;
using System.Text.Json;

namespace FaceMatchAPI.Middleware
{
    public class GlobalExceptionMiddleware
    {
        private readonly RequestDelegate _next;
        private readonly ILogger<GlobalExceptionMiddleware> _logger;

        public GlobalExceptionMiddleware(RequestDelegate next, ILogger<GlobalExceptionMiddleware> logger)
        {
            _next = next;
            _logger = logger;
        }

        public async Task InvokeAsync(HttpContext context)
        {
            try
            {
                await _next(context);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex,
                    "[GlobalException] 처리되지 않은 예외 발생 | {Method} {Path} | {ExType}: {Message}",
                    context.Request.Method,
                    context.Request.Path,
                    ex.GetType().Name,
                    ex.Message);

                // 이미 응답이 시작된 경우에는 헤더를 변경할 수 없으므로 그냥 연결 종료
                if (context.Response.HasStarted)
                {
                    _logger.LogWarning("[GlobalException] 응답이 이미 시작되어 오류 응답을 전송할 수 없습니다.");
                    return;
                }

                context.Response.StatusCode = 500;
                context.Response.ContentType = "application/json";

                var response = ResponseDTO<object>.ErrorResponse("500", "서버 내부 오류가 발생했습니다.");
                await context.Response.WriteAsync(JsonSerializer.Serialize(response));
            }
        }
    }

    public static class GlobalExceptionMiddlewareExtensions
    {
        public static IApplicationBuilder UseGlobalExceptionHandler(this IApplicationBuilder app)
            => app.UseMiddleware<GlobalExceptionMiddleware>();
    }
}

namespace FaceMatchAPI.Exceptions
{
    /// <summary>
    /// 얼굴 처리 파이프라인의 특정 단계에서 발생한 예외를 나타냅니다.
    /// </summary>
    public class FaceProcessingException : Exception
    {
        /// <summary>
        /// 예외가 발생한 처리 단계 (예: "Detection", "Alignment", "FeatureExtraction")
        /// </summary>
        public string Stage { get; }

        public FaceProcessingException(string stage, string message, Exception? inner = null)
            : base(message, inner)
        {
            Stage = stage;
        }
    }
}