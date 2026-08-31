namespace FaceMatchAPI.Dtos
{
    public enum ImageType
    {
        Normal,
        Target
    }

    public class UploadRequest
    {
        public ImageType ImageType { get; set; } = ImageType.Normal;
        public string Base64 { get; set; } = string.Empty;
        public bool ImageSave { get; set; } = true;
        public string SubId { get; set; } = string.Empty;
    }

    public class SearchRequest
    {
        public ImageType SearchType { get; set; } = ImageType.Normal;
        public string Base64 { get; set; } = string.Empty;
        public DateTime? StartDate { get; set; }
        public DateTime? EndDate { get; set; }
        public float MinScore { get; set; } = 0.5f;
    }

    public class SearchByVectorIdsRequest
    {
        public ImageType SearchType { get; set; } = ImageType.Normal;
        public string Base64 { get; set; } = string.Empty;
        public List<string> VectorIds { get; set; } = [];
        public float MinScore { get; set; } = 0.5f;
    }

    public class SearchByGroupRequest
    {
        public string GroupName { get; set; } = string.Empty;
        public string Base64 { get; set; } = string.Empty;
        public float MinScore { get; set; } = 0.5f;
        public int MaxMembers { get; set; } = 100000;
    }

    public class AddGroupMemberRequest
    {
        public string GroupName { get; set; } = string.Empty;
        public string SubId { get; set; } = string.Empty;
        public string Base64 { get; set; } = string.Empty;
        public bool ImageSave { get; set; }
    }

    public class DeleteGroupsRequest
    {
        public string GroupNamePrefix { get; set; } = string.Empty;
        public string DeleteThroughDate { get; set; } = string.Empty;
    }

    public class GetImageRequest
    {
        public List<string> Ids { get; set; } = [];
    }

    public class GetVectorRequest
    {
        public ImageType ImageType { get; set; } = ImageType.Normal;
        public List<string> VectorIds { get; set; } = [];
    }

    public class GetVectorPageRequest
    {
        public ImageType ImageType { get; set; } = ImageType.Normal;
        public DateTime? StartDate { get; set; }
        public DateTime? EndDate { get; set; }
        public string VectorId { get; set; } = string.Empty;
        public string SubId { get; set; } = string.Empty;
        public int Page { get; set; } = 1;       // 1부터 시작
        public int PageSize { get; set; } = 20;  // 기본 20개
    }

    public class VectorPageResult
    {
        public int Page { get; set; }
        public int PageSize { get; set; }
        public long TotalCount { get; set; }
        public int TotalPages { get; set; }
        public bool HasPrevious { get; set; }
        public bool HasNext { get; set; }
        public string ImageType { get; set; } = string.Empty;
        public IEnumerable<VectorItem> Vectors { get; set; } = [];
    }

    public class VectorItem
    {
        public string Id { get; set; } = string.Empty;
        public string ImageId { get; set; } = string.Empty;
        public string SubId { get; set; } = string.Empty;
        public float[] Vector { get; set; } = [];
        public DateTime CreatedAt { get; set; }
    }

    public class DeleteRequest
    {
        public ImageType ImageType { get; set; } = ImageType.Normal;
        public List<string> Ids { get; set; } = []; // VectorId 리스트
        public List<string> SubIds { get; set; } = []; // SubId 리스트 (VectorId 대신 SubId로 삭제할 때 사용)
    }
}
