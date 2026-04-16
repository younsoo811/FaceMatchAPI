using FaceMatchAPI.Dtos;
using MongoDB.Driver;

namespace FaceMatchAPI.Services
{
    public class MongoService
    {
        public IMongoDatabase Database { get; }

        public IMongoCollection<FaceVector> FaceVectors { get; }
        public IMongoCollection<FaceImage> FaceImages { get; }

        public MongoService()
        {
            var conn = Environment.GetEnvironmentVariable("MONGODB_CONNECTION_STR");
            var client = new MongoClient(conn);

            Database = client.GetDatabase("face_db");

            FaceVectors = Database.GetCollection<FaceVector>("face_vectors");
            FaceImages = Database.GetCollection<FaceImage>("face_images");
        }
    }
}
