using FaceMatchAPI.Dtos;
using FaceMatchAPI.Services;
using FaceMatchAPI.Utils;
using Microsoft.AspNetCore.Mvc;
using MongoDB.Bson;
using MongoDB.Driver;
using System.Buffers.Text;

namespace FaceMatchAPI.Controllers
{
    [ApiController]
    [Route("api/face")]
    public class FaceController : ControllerBase
    {
        private readonly MongoService _mongo;
        private readonly FaceService _face;

        public FaceController(MongoService mongo, FaceService face)
        {
            _mongo = mongo;
            _face = face;
        }

        // 1️. 이미지 저장(선택) + 벡터 저장
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
                    var image = new FaceImage
                    {
                        CreatedAt = DateTime.UtcNow,
                        Base64 = req.Base64
                    };

                    await _mongo.FaceImages.InsertOneAsync(image);

                    imageId = image.Id.ToString();
                }

                var vector = _face.ExtractFeatureWithFlip(req.Base64);

                var faceVector = new FaceVector
                {
                    CreatedAt = DateTime.UtcNow,
                    ImageId = imageId,
                    Vector = vector
                };

                if (req.ImageType == ImageType.Target)
                    await _mongo.TargetVectors.InsertOneAsync(faceVector);
                else
                    await _mongo.FaceVectors.InsertOneAsync(faceVector);

                var data = new
                {
                    vectorId = faceVector.Id.ToString(),
                    imageId = imageId,
                    imageType = req.ImageType.ToString()
                };

                return Ok(ResponseDTO<object>.SuccessResponse(data));
            }
            catch (Exception ex)
            {
                return StatusCode(500, ResponseDTO<object>.ErrorResponse("500", ex.Message));
            }
        }

        // 2️. 얼굴 검색
        [HttpPost("search")]
        public async Task<IActionResult> Search([FromBody] SearchRequest req)
        {
            try
            {
                var (isValid, format, errorMsg) = Base64ImageValidator.Validate(req.Base64);
                if (!isValid)
                    return BadRequest(ResponseDTO<object>.ErrorResponse("400", errorMsg!));

                var queryVector = _face.ExtractFeatureWithFlip(req.Base64);

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
            catch (Exception ex)
            {
                return StatusCode(500, ResponseDTO<object>.ErrorResponse("500", ex.Message));
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

                var queryVector = _face.ExtractFeatureWithFlip(req.Base64);

                var filter = Builders<FaceVector>.Filter.Empty;

                if(req.VectorIds.Count > 0)
                    filter &= Builders<FaceVector>.Filter.In(x => x.Id, req.VectorIds.Select(id => ObjectId.Parse(id)));

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
            catch (Exception ex)
            {
                return StatusCode(500, ResponseDTO<object>.ErrorResponse("500", ex.Message));
            }
        }

        [HttpPost("images")]
        public async Task<IActionResult> GetImages([FromBody] GetImageRequest req)
        {
            try
            {
                if (req.Ids == null || req.Ids.Count == 0)
                    return BadRequest(ResponseDTO<object>.ErrorResponse("400", "조회할 ID를 하나 이상 입력해주세요."));

                var (validIds, invalidIds) = ParseObjectIds(req.Ids);

                if (invalidIds.Count > 0)
                    return BadRequest(ResponseDTO<object>.ErrorResponse("400",
                        $"유효하지 않은 ID 형식이 포함되어 있습니다: [{string.Join(", ", invalidIds)}]"));

                var filter = Builders<FaceImage>.Filter.In(x => x.Id, validIds);
                var images = await _mongo.FaceImages.Find(filter).ToListAsync();

                if (images.Count == 0)
                    return NotFound(ResponseDTO<object>.ErrorResponse("404", "일치하는 이미지 데이터가 존재하지 않습니다."));

                // 요청했지만 DB에 없는 ID 계산
                var foundIds = images.Select(x => x.Id.ToString()).ToHashSet();
                var notFoundIds = req.Ids.Where(id => !foundIds.Contains(id)).ToList();

                var data = new
                {
                    requestedCount = req.Ids.Count,
                    foundCount = images.Count,
                    notFoundIds,
                    images = images.Select(x => new
                    {
                        id = x.Id.ToString(),
                        base64 = x.Base64,
                        createdAt = x.CreatedAt
                    })
                };

                return Ok(ResponseDTO<object>.SuccessResponse(data));
            }
            catch (Exception ex)
            {
                return StatusCode(500, ResponseDTO<object>.ErrorResponse("500", ex.Message));
            }
        }

        [HttpPost("vectors")]
        public async Task<IActionResult> GetVectors([FromBody] GetVectorRequest req)
        {
            try
            {
                if (req.VectorIds == null || req.VectorIds.Count == 0)
                    return BadRequest(ResponseDTO<object>.ErrorResponse("400", "조회할 VectorId를 하나 이상 입력해주세요."));

                var (validIds, invalidIds) = ParseObjectIds(req.VectorIds);

                if (invalidIds.Count > 0)
                    return BadRequest(ResponseDTO<object>.ErrorResponse("400",
                        $"유효하지 않은 ID 형식이 포함되어 있습니다: [{string.Join(", ", invalidIds)}]"));

                // ImageType에 따라 컬렉션 분기
                var collection = req.ImageType == ImageType.Target
                    ? _mongo.TargetVectors
                    : _mongo.FaceVectors;

                var filter = Builders<FaceVector>.Filter.In(x => x.Id, validIds);
                var vectors = await collection.Find(filter).ToListAsync();

                if (vectors.Count == 0)
                    return NotFound(ResponseDTO<object>.ErrorResponse("404", "일치하는 벡터 데이터가 존재하지 않습니다."));

                // 요청했지만 DB에 없는 ID 계산
                var foundIds = vectors.Select(x => x.Id.ToString()).ToHashSet();
                var notFoundIds = req.VectorIds.Where(id => !foundIds.Contains(id)).ToList();

                var data = new
                {
                    requestedCount = req.VectorIds.Count,
                    foundCount = vectors.Count,
                    imageType = req.ImageType.ToString(),
                    notFoundIds,
                    vectors = vectors.Select(x => new
                    {
                        id = x.Id.ToString(),
                        imageId = x.ImageId,
                        vector = x.Vector,
                        createdAt = x.CreatedAt
                    })
                };

                return Ok(ResponseDTO<object>.SuccessResponse(data));
            }
            catch (Exception ex)
            {
                return StatusCode(500, ResponseDTO<object>.ErrorResponse("500", ex.Message));
            }
        }

        [HttpPost("vectors/page")]
        public async Task<IActionResult> GetVectorPage([FromBody] GetVectorPageRequest req)
        {
            try
            {
                // 페이지 값 유효성 검사
                if (req.Page < 1)
                    return BadRequest(ResponseDTO<object>.ErrorResponse("400", "Page는 1 이상이어야 합니다."));

                if (req.PageSize < 1 || req.PageSize > 100)
                    return BadRequest(ResponseDTO<object>.ErrorResponse("400", "PageSize는 1 이상 100 이하이어야 합니다."));

                // 날짜 범위 유효성 검사
                if (req.StartDate.HasValue && req.EndDate.HasValue && req.StartDate > req.EndDate)
                    return BadRequest(ResponseDTO<object>.ErrorResponse("400", "StartDate는 EndDate보다 이전이어야 합니다."));

                // ImageType에 따라 컬렉션 분기
                var collection = req.ImageType == ImageType.Target
                    ? _mongo.TargetVectors
                    : _mongo.FaceVectors;

                // 필터 구성
                var filter = Builders<FaceVector>.Filter.Empty;
                if (req.StartDate.HasValue)
                    filter &= Builders<FaceVector>.Filter.Gte(x => x.CreatedAt, req.StartDate.Value);
                if (req.EndDate.HasValue)
                    filter &= Builders<FaceVector>.Filter.Lte(x => x.CreatedAt, req.EndDate.Value);

                // 전체 건수와 페이지 데이터를 병렬로 조회
                var totalCountTask = collection.CountDocumentsAsync(filter);
                var vectorsTask = collection
                    .Find(filter)
                    .SortByDescending(x => x.CreatedAt)
                    .Skip((req.Page - 1) * req.PageSize)
                    .Limit(req.PageSize)
                    .ToListAsync();

                await Task.WhenAll(totalCountTask, vectorsTask);

                var totalCount = await totalCountTask;
                var vectors = await vectorsTask;

                if (totalCount == 0)
                    return NotFound(ResponseDTO<object>.ErrorResponse("404", "조회된 벡터 데이터가 없습니다."));

                var totalPages = (int)Math.Ceiling((double)totalCount / req.PageSize);

                var data = new VectorPageResult
                {
                    Page = req.Page,
                    PageSize = req.PageSize,
                    TotalCount = totalCount,
                    TotalPages = totalPages,
                    HasPrevious = req.Page > 1,
                    HasNext = req.Page < totalPages,
                    ImageType = req.ImageType.ToString(),
                    Vectors = vectors.Select(x => new VectorItem
                    {
                        Id = x.Id.ToString(),
                        ImageId = x.ImageId,
                        SubId = x.SubId,
                        Vector = x.Vector,
                        CreatedAt = x.CreatedAt
                    })
                };

                return Ok(ResponseDTO<VectorPageResult>.SuccessResponse(data));
            }
            catch (Exception ex)
            {
                return StatusCode(500, ResponseDTO<object>.ErrorResponse("500", ex.Message));
            }
        }

        [HttpPost("delete")]
        public async Task<IActionResult> Delete([FromBody] DeleteRequest req)
        {
            try
            {
                bool useSubIds = req.SubIds != null && req.SubIds.Count > 0;

                if (!useSubIds && (req.Ids == null || req.Ids.Count == 0))
                {
                    return BadRequest(
                        ResponseDTO<object>.ErrorResponse(
                            "400",
                            "삭제할 ID 또는 SubId를 하나 이상 입력해주세요."));
                }

                // ImageType에 따라 컬렉션 분기
                var collection = req.ImageType == ImageType.Target
                    ? _mongo.TargetVectors
                    : _mongo.FaceVectors;

                FilterDefinition<FaceVector> vectorFilter;

                List<string> invalidIds = [];

                // SubIds 기준 삭제
                if (useSubIds)
                {
                    vectorFilter = Builders<FaceVector>.Filter.In(
                        x => x.SubId,
                        req.SubIds);
                }
                // ObjectId 기준 삭제
                else
                {
                    var objectIds = new List<ObjectId>();

                    foreach (var id in req.Ids)
                    {
                        if (ObjectId.TryParse(id, out var objectId))
                            objectIds.Add(objectId);
                        else
                            invalidIds.Add(id);
                    }

                    if (invalidIds.Count > 0)
                    {
                        return BadRequest(
                            ResponseDTO<object>.ErrorResponse(
                                "400",
                                $"유효하지 않은 ID 형식이 포함되어 있습니다: [{string.Join(", ", invalidIds)}]"));
                    }

                    vectorFilter = Builders<FaceVector>.Filter.In(
                        x => x.Id,
                        objectIds);
                }

                var foundVectors = await collection.Find(vectorFilter).ToListAsync();

                if (foundVectors.Count == 0)
                    return NotFound(ResponseDTO<object>.ErrorResponse("404", "일치하는 데이터가 존재하지 않습니다."));

                // 찾은 벡터들의 ImageId 수집
                var imageObjectIds = foundVectors
                    .Select(x => ObjectId.Parse(x.ImageId))
                    .ToList();

                // FaceVectors(or TargetVectors) 삭제
                var vectorDeleteResult = await collection.DeleteManyAsync(vectorFilter);

                // Image 삭제
                long imageDeletedCount = 0;

                if (imageObjectIds.Count > 0)
                {
                    var imageFilter = Builders<FaceImage>.Filter.In(
                        x => x.Id,
                        imageObjectIds);

                    var imageDeleteResult =
                        await _mongo.FaceImages.DeleteManyAsync(imageFilter);

                    imageDeletedCount = imageDeleteResult.DeletedCount;
                }

                // 찾지 못한 항목 계산
                List<string> notFoundIds = [];

                if (useSubIds)
                {
                    var foundSubIds = foundVectors
                        .Select(x => x.SubId)
                        .Where(x => !string.IsNullOrEmpty(x))
                        .ToHashSet();

                    notFoundIds = req.SubIds
                        .Where(x => !foundSubIds.Contains(x))
                        .ToList();
                }
                else
                {
                    var foundIds = foundVectors
                        .Select(x => x.Id.ToString())
                        .ToHashSet();

                    notFoundIds = req.Ids
                        .Where(x => !foundIds.Contains(x))
                        .ToList();
                }

                var data = new
                {
                    requestedCount = useSubIds
                        ? req.SubIds.Count
                        : req.Ids.Count,

                    deletedCount = (int)vectorDeleteResult.DeletedCount,

                    imageDeletedCount,

                    notFoundIds
                };

                return Ok(ResponseDTO<object>.SuccessResponse(data, "삭제가 완료되었습니다."));
            }
            catch (Exception ex)
            {
                return StatusCode(500, ResponseDTO<object>.ErrorResponse("500", ex.Message));
            }
        }

        private float CosineSimilarity(float[] a, float[] b)
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

        private (List<ObjectId> Valid, List<string> Invalid) ParseObjectIds(List<string> ids)
        {
            var valid = new List<ObjectId>();
            var invalid = new List<string>();

            foreach (var id in ids)
            {
                if (ObjectId.TryParse(id, out var objectId))
                    valid.Add(objectId);
                else
                    invalid.Add(id);
            }

            return (valid, invalid);
        }
    }
}
