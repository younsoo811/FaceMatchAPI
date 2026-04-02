using MongoDB.Bson.Serialization.Attributes;
using MongoDB.Bson;

namespace FaceMatchAPI.Dtos
{
    public class FaceImage
    {
        [BsonId]
        public ObjectId Id { get; set; }

        public DateTime CreatedAt { get; set; }

        public string Base64 { get; set; } = string.Empty;
    }
}
