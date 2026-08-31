using FaceMatchAPI.Dtos;
using MongoDB.Driver;

namespace FaceMatchAPI.Services
{
    public class MongoService
    {
        public IMongoDatabase Database { get; }

        public IMongoCollection<FaceVector> FaceVectors { get; }
        public IMongoCollection<FaceVector> TargetVectors { get; }
        public IMongoCollection<FaceImage> FaceImages { get; }
        public IMongoCollection<FaceGroup> FaceGroups { get; }

        public MongoService()
        {
            var conn = Environment.GetEnvironmentVariable("MONGODB_CONNECTION_STR");
            var client = new MongoClient(conn);

            Database = client.GetDatabase("face_db");

            FaceVectors = Database.GetCollection<FaceVector>("face_vectors");
            TargetVectors = Database.GetCollection<FaceVector>("target_vectors");
            FaceImages = Database.GetCollection<FaceImage>("face_images");
            FaceGroups = Database.GetCollection<FaceGroup>("face_groups");
        }
    }
}
