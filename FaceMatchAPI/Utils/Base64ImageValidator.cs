namespace FaceMatchAPI.Utils
{
    public static class Base64ImageValidator
    {
        // 지원할 이미지 포맷별 매직 바이트 정의
        private static readonly Dictionary<string, byte[][]> ImageSignatures = new()
        {
            ["JPEG"] = [[0xFF, 0xD8, 0xFF]],
            ["PNG"] = [[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]],
            ["GIF"] = [[ 0x47, 0x49, 0x46, 0x38, 0x37, 0x61 ],   // GIF87a
                    [ 0x47, 0x49, 0x46, 0x38, 0x39, 0x61 ]],  // GIF89a
            ["BMP"] = [[0x42, 0x4D]],
            ["WEBP"] = [[0x52, 0x49, 0x46, 0x46]],
        };

        public static (bool IsValid, string? Format, string? ErrorMsg) Validate(string base64)
        {
            if (string.IsNullOrWhiteSpace(base64))
                return (false, null, "Base64 값이 비어있습니다.");

            // data:image/jpeg;base64,xxx 형태의 prefix 제거
            var raw = StripDataUrlPrefix(base64);

            // Base64 디코딩 시도
            byte[] bytes;
            try
            {
                bytes = Convert.FromBase64String(raw);
            }
            catch (FormatException)
            {
                return (false, null, "유효하지 않은 Base64 형식입니다.");
            }

            if (bytes.Length < 8)
                return (false, null, "데이터가 너무 짧아 이미지로 인식할 수 없습니다.");

            // 매직 바이트로 이미지 포맷 판별
            foreach (var (format, signatures) in ImageSignatures)
            {
                foreach (var sig in signatures)
                {
                    if (bytes.Length >= sig.Length && bytes.Take(sig.Length).SequenceEqual(sig))
                        return (true, format, null);
                }
            }

            return (false, null, "지원하지 않는 파일 형식입니다. (JPEG, PNG, GIF, BMP, WEBP만 허용)");
        }

        private static string StripDataUrlPrefix(string base64)
        {
            // "data:image/jpeg;base64," 같은 prefix 제거
            var commaIndex = base64.IndexOf(',');
            return commaIndex >= 0 ? base64[(commaIndex + 1)..] : base64;
        }
    }
}
