namespace FaceMatchAPI.Dtos
{
    public class UploadRequest
    {
        public string Base64 { get; set; } = string.Empty;
    }

    public class SearchRequest
    {
        public string Base64 { get; set; } = string.Empty;
        public DateTime? StartDate { get; set; }
        public DateTime? EndDate { get; set; }
        public float MinScore { get; set; } = 0.5f;
    }
}
