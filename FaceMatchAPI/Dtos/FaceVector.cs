using MongoDB.Bson.Serialization.Attributes;
using MongoDB.Bson;

namespace FaceMatchAPI.Dtos
{
    public class FaceVector
    {
        [BsonId]
        public ObjectId Id { get; set; }

        public DateTime CreatedAt { get; set; }

        public string ImageId { get; set; } = string.Empty;

        public float[] Vector { get; set; } = Array.Empty<float>();
    }
}
