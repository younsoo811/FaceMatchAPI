namespace FaceMatchAPI.Dtos
{
    public class ResponseDTO<T>
    {
        public string Code { get; set; }
        public string Msg { get; set; }
        public T? Data { get; set; }
        public bool Success { get; set; }

        public static ResponseDTO<T> SuccessResponse(T data, string msg = "OK")
        {
            return new ResponseDTO<T>
            {
                Code = "200",
                Msg = msg,
                Data = data,
                Success = true
            };
        }

        public static ResponseDTO<T> ErrorResponse(string code, string msg)
        {
            return new ResponseDTO<T>
            {
                Code = code,
                Msg = msg,
                Data = default,
                Success = false
            };
        }
    }
}
