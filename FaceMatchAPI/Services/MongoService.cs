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
            var client = new MongoClient("mongodb://seo_ai:qrqvud3@127.0.0.1:27017");

            Database = client.GetDatabase("face_db");

            FaceVectors = Database.GetCollection<FaceVector>("face_vectors");
            FaceImages = Database.GetCollection<FaceImage>("face_images");
        }
    }
}
