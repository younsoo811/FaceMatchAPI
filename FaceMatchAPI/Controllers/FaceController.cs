using FaceMatchAPI.Dtos;
using FaceMatchAPI.Services;
using Microsoft.AspNetCore.Mvc;
using MongoDB.Driver;

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

        // 1️. 이미지 저장 + 벡터 저장
        [HttpPost("upload")]
        public async Task<IActionResult> Upload([FromBody] UploadRequest req)
        {
            var image = new FaceImage
            {
                CreatedAt = DateTime.UtcNow,
                Base64 = req.Base64
            };

            await _mongo.FaceImages.InsertOneAsync(image);

            var vector = _face.ExtractFeature(req.Base64);

            var faceVector = new FaceVector
            {
                CreatedAt = DateTime.UtcNow,
                ImageId = image.Id.ToString(),
                Vector = vector
            };

            await _mongo.FaceVectors.InsertOneAsync(faceVector);

            return Ok(new { imageId = image.Id });
        }

        // 2️. 얼굴 검색
        [HttpPost("search")]
        public async Task<IActionResult> Search([FromBody] SearchRequest req)
        {
            var queryVector = _face.ExtractFeature(req.Base64);

            var filter = Builders<FaceVector>.Filter.Empty;

            if (req.StartDate.HasValue)
                filter &= Builders<FaceVector>.Filter.Gte(x => x.CreatedAt, req.StartDate);

            if (req.EndDate.HasValue)
                filter &= Builders<FaceVector>.Filter.Lte(x => x.CreatedAt, req.EndDate);

            var list = await _mongo.FaceVectors.Find(filter).ToListAsync();

            var result = list
                .Select(x => new
                {
                    x.ImageId,
                    Score = CosineSimilarity(queryVector, x.Vector)
                })
                .Where(x => x.Score >= req.MinScore)
                .OrderByDescending(x => x.Score)
                .ToList();

            return Ok(result);
        }

        private float CosineSimilarity(float[] a, float[] b)
        {
            float dot = 0;
            for (int i = 0; i < a.Length; i++)
                dot += a[i] * b[i];
            return dot;
        }
    }
}
