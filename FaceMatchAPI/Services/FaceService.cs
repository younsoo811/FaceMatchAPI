using DlibDotNet;
using DlibDotNet.Extensions;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using OpenCvSharp.Dnn;
using Drawing = System.Drawing;

namespace FaceMatchAPI.Services
{
    public class FaceService
    {
        private FaceDetector _detector;
        private FaceAligner _aligner;
        private FaceModel _model;

        public FaceService(IWebHostEnvironment env)
        {
            string baseDir = Path.Combine(env.ContentRootPath, "Models");

            _detector = new FaceDetector(
                Path.Combine(baseDir, "deploy.prototxt"),
                Path.Combine(baseDir, "res10_300x300_ssd_iter_140000.caffemodel")
            );

            _aligner = new FaceAligner(
                Path.Combine(baseDir, "shape_predictor_68_face_landmarks.dat")
            );

            _model = new FaceModel(
                Path.Combine(baseDir, "model.onnx")
            );
        }

        public float[] ExtractFeature(string base64)
        {
            var bmp = Base64ToBitmap(base64);
            var face = _detector.Detect(bmp);
            var aligned = _aligner.Align(To24bpp(face));

            return _model.GetFeature(aligned);
        }

        // ===== Utils =====

        private static System.Drawing.Bitmap Base64ToBitmap(string base64)
        {
            if (base64.Contains(",")) base64 = base64.Split(',')[1];
            var bytes = Convert.FromBase64String(base64);
            using var ms = new MemoryStream(bytes);
            return new System.Drawing.Bitmap(ms);
        }

        private static System.Drawing.Bitmap To24bpp(System.Drawing.Bitmap src)
        {
            var bmp = new System.Drawing.Bitmap(src.Width, src.Height,
                System.Drawing.Imaging.PixelFormat.Format24bppRgb);

            using var g = System.Drawing.Graphics.FromImage(bmp);
            g.DrawImage(src, 0, 0);
            return bmp;
        }
    }

    // ================= 얼굴 검출 =================
    class FaceDetector
    {
        private Net _net;

        public FaceDetector(string proto, string model)
        {
            if (!File.Exists(proto))
                throw new Exception($"prototxt 없음: {proto}");

            if (!File.Exists(model))
                throw new Exception($"caffemodel 없음: {model}");

            _net = CvDnn.ReadNetFromCaffe(proto, model);

            if (_net.Empty())
                throw new Exception("DNN 모델 로딩 실패");
        }

        public Drawing.Bitmap Detect(Drawing.Bitmap bitmap)
        {
            // todo: windows 에서만 동작함
            var mat = OpenCvSharp.Extensions.BitmapConverter.ToMat(bitmap);

            // 4채널(png) → 3채널 변환
            if (mat.Channels() == 4)
            {
                Cv2.CvtColor(mat, mat, ColorConversionCodes.BGRA2BGR);
            }

            var blob = CvDnn.BlobFromImage(mat, 1.0,
                new OpenCvSharp.Size(300, 300),
                new Scalar(104, 177, 123));

            _net.SetInput(blob);
            var output = _net.Forward();

            int w = mat.Width;
            int h = mat.Height;

            // 데이터 추출
            var data = new float[output.Total()];
            System.Runtime.InteropServices.Marshal.Copy(
                output.Data,
                data,
                0,
                data.Length
            );

            int count = data.Length / 7;

            float maxConf = 0;
            int best = -1;

            for (int i = 0; i < count; i++)
            {
                float conf = data[i * 7 + 2];

                if (conf > maxConf)
                {
                    maxConf = conf;
                    best = i;
                }
            }

            if (best == -1 || maxConf < 0.5)
            {
                Console.WriteLine("얼굴 검출 실패 → 중앙 crop");
                return CenterCrop(bitmap);
            }

            int idx = best * 7;

            int x1 = (int)(data[idx + 3] * w);
            int y1 = (int)(data[idx + 4] * h);
            int x2 = (int)(data[idx + 5] * w);
            int y2 = (int)(data[idx + 6] * h);

            // 좌표 보정
            x1 = Math.Max(0, x1);
            y1 = Math.Max(0, y1);
            x2 = Math.Min(w - 1, x2);
            y2 = Math.Min(h - 1, y2);

            var rect = new OpenCvSharp.Rect(x1, y1, x2 - x1, y2 - y1);

            var face = new Mat(mat, rect);

            return OpenCvSharp.Extensions.BitmapConverter.ToBitmap(face);
        }

        private Drawing.Bitmap CenterCrop(Drawing.Bitmap src)
        {
            int size = Math.Min(src.Width, src.Height);

            var rect = new Drawing.Rectangle(
                (src.Width - size) / 2,
                (src.Height - size) / 2,
                size, size);

            var result = new Drawing.Bitmap(size, size);

            using var g = Drawing.Graphics.FromImage(result);
            g.DrawImage(src, new Drawing.Rectangle(0, 0, size, size), rect, Drawing.GraphicsUnit.Pixel);

            return result;
        }
    }


    // ================= 얼굴 정렬 =================
    class FaceAligner
    {
        private FrontalFaceDetector _detector;
        private ShapePredictor _predictor;

        public FaceAligner(string modelPath)
        {
            _detector = Dlib.GetFrontalFaceDetector();
            _predictor = ShapePredictor.Deserialize(modelPath);
        }

        public Drawing.Bitmap Align(Drawing.Bitmap bitmap)
        {
            var img = bitmap.ToMatrix<RgbPixel>();            
            var faces = _detector.Operator(img);

            if (faces.Length == 0)
                return bitmap;

            var shape = _predictor.Detect(img, faces[0]);

            var leftEye = GetPoint(shape, 36, 41);
            var rightEye = GetPoint(shape, 42, 47);

            return Rotate(bitmap, leftEye, rightEye);
        }

        private Drawing.Point GetPoint(FullObjectDetection shape, int s, int e)
        {
            int x = 0, y = 0;
            for (int i = s; i <= e; i++)
            {
                x += shape.GetPart((uint)i).X;
                y += shape.GetPart((uint)i).Y;
            }
            return new Drawing.Point(x / (e - s + 1), y / (e - s + 1));
        }

        private Drawing.Bitmap Rotate(Drawing.Bitmap src, Drawing.Point l, Drawing.Point r)
        {
            double dy = r.Y - l.Y;
            double dx = r.X - l.X;
            double angle = Math.Atan2(dy, dx) * 180.0 / Math.PI;

            var result = new Drawing.Bitmap(src.Width, src.Height);

            using var g = Drawing.Graphics.FromImage(result);
            g.TranslateTransform(src.Width / 2, src.Height / 2);
            g.RotateTransform((float)-angle);
            g.TranslateTransform(-src.Width / 2, -src.Height / 2);
            g.DrawImage(src, new Drawing.Point(0, 0));

            return result;
        }
    }


    // ================= ArcFace =================
    class FaceModel
    {
        private InferenceSession _session;

        public FaceModel(string path)
        {
            _session = new InferenceSession(path);
        }

        public float[] GetFeature(Drawing.Bitmap bmp)
        {
            var resized = new Drawing.Bitmap(bmp, new Drawing.Size(112, 112));

            float[] input = new float[1 * 3 * 112 * 112];

            int channelSize = 112 * 112;

            for (int y = 0; y < 112; y++)
            {
                for (int x = 0; x < 112; x++)
                {
                    var p = resized.GetPixel(x, y);

                    int idx = y * 112 + x;

                    // BGR + CHW 순서
                    input[idx] = (p.B - 127.5f) / 128f;
                    input[channelSize + idx] = (p.G - 127.5f) / 128f;
                    input[channelSize * 2 + idx] = (p.R - 127.5f) / 128f;
                }
            }

            var tensor = new DenseTensor<float>(input, new[] { 1, 3, 112, 112 });

            var inputName = _session.InputMetadata.Keys.First();

            var result = _session.Run(new[]
            {
        NamedOnnxValue.CreateFromTensor(inputName, tensor)
    });

            var output = result.First().AsEnumerable<float>().ToArray();

            return Normalize(output);
        }

        private float[] Normalize(float[] v)
        {
            float sum = 0;
            foreach (var x in v) sum += x * x;
            float norm = (float)Math.Sqrt(sum);
            return v.Select(x => x / norm).ToArray();
        }
    }
}
