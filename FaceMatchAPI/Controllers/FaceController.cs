using FaceMatchAPI.Dtos;
using FaceMatchAPI.Exceptions;
using FaceMatchAPI.Services;
using FaceMatchAPI.Utils;
using Microsoft.AspNetCore.Mvc;
using MongoDB.Bson;
using MongoDB.Driver;

namespace FaceMatchAPI.Controllers
{
    [ApiController]
    [Route("api/face")]
    public class FaceController : ControllerBase
    {
        private readonly MongoService _mongo;
        private readonly FaceService _face;
        private readonly ILogger<FaceController> _logger;

        public FaceController(MongoService mongo, FaceService face, ILogger<FaceController> logger)
        {
            _mongo = mongo;
            _face = face;
            _logger = logger;
        }

        [HttpPost("upload")]
        public async Task<IActionResult> Upload([FromBody] UploadRequest req)
        {
            try
            {
                var (isValid, format, errorMsg) = Base64ImageValidator.Validate(req.Base64);
                if (!isValid)
                    return BadRequest(ResponseDTO<object>.ErrorResponse("400", errorMsg!));

                string? imageId = string.Empty;

                if (req.ImageSave)
                {
                    var image = new FaceImage { CreatedAt = DateTime.UtcNow, Base64 = req.Base64 };
                    await _mongo.FaceImages.InsertOneAsync(image);
                    imageId = image.Id.ToString();
                }

                // async로 Pool에서 처리 (대기 중 요청 취소 지원)
                var vector = await _face.ExtractFeatureWithFlipAsync(req.Base64, HttpContext.RequestAborted);

                var faceVector = new FaceVector
                {
                    CreatedAt = DateTime.UtcNow,
                    ImageId = imageId,
                    SubId = req.SubId,
                    Vector = vector
                };

                if (req.ImageType == ImageType.Target)
                    await _mongo.TargetVectors.InsertOneAsync(faceVector);
                else
                    await _mongo.FaceVectors.InsertOneAsync(faceVector);

                var data = new
                {
                    vectorId = faceVector.Id.ToString(),
                    imageId,
                    imageType = req.ImageType.ToString()
                };

                return Ok(ResponseDTO<object>.SuccessResponse(data));
            }
            catch (OperationCanceledException)
            {
                _logger.LogInformation("[Upload] 클라이언트가 요청을 취소했습니다. SubId={SubId}", req.SubId);
                return StatusCode(499, ResponseDTO<object>.ErrorResponse("499", "요청이 취소되었습니다."));
            }
            catch (FaceProcessingException fpEx)
            {
                _logger.LogWarning(fpEx, "[Upload] 얼굴 처리 실패 | Stage={Stage} | SubId={SubId}",
                    fpEx.Stage, req.SubId);
                return StatusCode(422, ResponseDTO<object>.ErrorResponse("422",
                    $"얼굴 처리 실패 [{fpEx.Stage}]: {fpEx.Message}"));
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[Upload] 처리 중 오류 | SubId={SubId} | {ExType}: {Message}",
                    req.SubId, ex.GetType().Name, ex.Message);
                return StatusCode(500, ResponseDTO<object>.ErrorResponse("500", "서버 내부 오류가 발생했습니다."));
            }
        }

        [HttpPost("search")]
        public async Task<IActionResult> Search([FromBody] SearchRequest req)
        {
            try
            {
                var (isValid, format, errorMsg) = Base64ImageValidator.Validate(req.Base64);
                if (!isValid)
                    return BadRequest(ResponseDTO<object>.ErrorResponse("400", errorMsg!));

                var queryVector = await _face.ExtractFeatureWithFlipAsync(req.Base64, HttpContext.RequestAborted);

                var filter = Builders<FaceVector>.Filter.Empty;

                if (req.StartDate.HasValue)
                    filter &= Builders<FaceVector>.Filter.Gte(x => x.CreatedAt, req.StartDate);
                if (req.EndDate.HasValue)
                    filter &= Builders<FaceVector>.Filter.Lte(x => x.CreatedAt, req.EndDate);

                var collection = req.SearchType == ImageType.Target
                    ? _mongo.TargetVectors
                    : _mongo.FaceVectors;

                var list = await collection.Find(filter).ToListAsync();

                var result = list
                    .Select(x => new
                    {
                        Id = x.Id.ToString(),
                        x.ImageId,
                        x.SubId,
                        Score = CosineSimilarity(queryVector, x.Vector)
                    })
                    .Where(x => x.Score >= req.MinScore)
                    .OrderByDescending(x => x.Score)
                    .ToList();

                return Ok(ResponseDTO<object>.SuccessResponse(result));
            }
            catch (OperationCanceledException)
            {
                _logger.LogInformation("[Search] 클라이언트가 요청을 취소했습니다.");
                return StatusCode(499, ResponseDTO<object>.ErrorResponse("499", "요청이 취소되었습니다."));
            }
            catch (FaceProcessingException fpEx)
            {
                _logger.LogWarning(fpEx, "[Search] 얼굴 처리 실패 | Stage={Stage}", fpEx.Stage);
                return StatusCode(422, ResponseDTO<object>.ErrorResponse("422",
                    $"얼굴 처리 실패 [{fpEx.Stage}]: {fpEx.Message}"));
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[Search] 처리 중 오류 | {ExType}: {Message}", ex.GetType().Name, ex.Message);
                return StatusCode(500, ResponseDTO<object>.ErrorResponse("500", "서버 내부 오류가 발생했습니다."));
            }
        }

        [HttpPost("search/byVectorIds")]
        public async Task<IActionResult> SearchByVectorIds([FromBody] SearchByVectorIdsRequest req)
        {
            try
            {
                var (isValid, format, errorMsg) = Base64ImageValidator.Validate(req.Base64);
                if (!isValid)
                    return BadRequest(ResponseDTO<object>.ErrorResponse("400", errorMsg!));

                var queryVector = await _face.ExtractFeatureWithFlipAsync(req.Base64, HttpContext.RequestAborted);

                var filter = Builders<FaceVector>.Filter.Empty;

                if (req.VectorIds.Count > 0)
                {
                    var objectIds = req.VectorIds
                        .Where(id => !string.IsNullOrWhiteSpace(id) && ObjectId.TryParse(id, out _))
                        .Select(ObjectId.Parse)
                        .Distinct()
                        .ToList();

                    if (objectIds.Count == 0)
                        return BadRequest(ResponseDTO<object>.ErrorResponse("400", "유효한 VectorIds가 없습니다."));

                    filter &= Builders<FaceVector>.Filter.In(x => x.Id, objectIds);
                }

                var collection = req.SearchType == ImageType.Target
                    ? _mongo.TargetVectors
                    : _mongo.FaceVectors;

                var list = await collection.Find(filter).ToListAsync();

                var result = list
                    .Select(x => new
                    {
                        Id = x.Id.ToString(),
                        x.ImageId,
                        x.SubId,
                        Score = CosineSimilarity(queryVector, x.Vector)
                    })
                    .Where(x => x.Score >= req.MinScore)
                    .OrderByDescending(x => x.Score)
                    .ToList();

                return Ok(ResponseDTO<object>.SuccessResponse(result));
            }
            catch (OperationCanceledException)
            {
                _logger.LogInformation("[SearchByVectorIds] 클라이언트가 요청을 취소했습니다.");
                return StatusCode(499, ResponseDTO<object>.ErrorResponse("499", "요청이 취소되었습니다."));
            }
            catch (FaceProcessingException fpEx)
            {
                _logger.LogWarning(fpEx, "[SearchByVectorIds] 얼굴 처리 실패 | Stage={Stage}", fpEx.Stage);
                return StatusCode(422, ResponseDTO<object>.ErrorResponse("422",
                    $"얼굴 처리 실패 [{fpEx.Stage}]: {fpEx.Message}"));
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[SearchByVectorIds] 처리 중 오류 | {ExType}: {Message}",
                    ex.GetType().Name, ex.Message);
                return StatusCode(500, ResponseDTO<object>.ErrorResponse("500", "서버 내부 오류가 발생했습니다."));
            }
        }

        private static float CosineSimilarity(float[] a, float[] b)
        {
            float dot = 0, normA = 0, normB = 0;
            for (int i = 0; i < a.Length; i++)
            {
                dot += a[i] * b[i];
                normA += a[i] * a[i];
                normB += b[i] * b[i];
            }
            return dot / (float)(Math.Sqrt(normA) * Math.Sqrt(normB));
        }
    }
}