using MongoDB.Bson;
using MongoDB.Bson.Serialization.Attributes;

namespace FaceMatchAPI.Dtos
{
    public class FaceGroup
    {
        [BsonId]
        public string Name { get; set; } = string.Empty;

        public List<ObjectId> MemberIds { get; set; } = [];
    }
}
