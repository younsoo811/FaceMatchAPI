using DlibDotNet;
using DlibDotNet.Extensions;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using OpenCvSharp.Dnn;
using System.Runtime.InteropServices;
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

        public float[] ExtractFeatureWithFlip(string base64)
        {
            var mat = Base64ToMat(base64);

            var face = _detector.Detect(mat);
            var aligned = _aligner.Align(face);

            var f1 = _model.GetFeature(aligned);

            // flip (OpenCV)
            var flipped = new Mat();
            Cv2.Flip(aligned, flipped, FlipMode.Y);

            var f2 = _model.GetFeature(flipped);

            return _model.Average(f1, f2);
        }

        // ===== Utils =====

        private Mat Base64ToMat(string base64)
        {
            if (base64.Contains(",")) base64 = base64.Split(',')[1];
            var bytes = Convert.FromBase64String(base64);

            return Cv2.ImDecode(bytes, ImreadModes.Color);
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

        public Mat Detect(Mat mat)
        {
            if (mat.Empty())
                throw new Exception("이미지 로딩 실패");

            // 4채널 → 3채널
            if (mat.Channels() == 4)
            {
                Cv2.CvtColor(mat, mat, ColorConversionCodes.BGRA2BGR);
            }

            var blob = CvDnn.BlobFromImage(mat, 1.0,
                new Size(300, 300),
                new Scalar(104, 177, 123));

            _net.SetInput(blob);
            var output = _net.Forward();

            int w = mat.Width;
            int h = mat.Height;

            var data = new float[output.Total()];
            Marshal.Copy(output.Data, data, 0, data.Length);

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
                return CenterCrop(mat);
            }

            int idx = best * 7;

            int x1 = (int)(data[idx + 3] * w);
            int y1 = (int)(data[idx + 4] * h);
            int x2 = (int)(data[idx + 5] * w);
            int y2 = (int)(data[idx + 6] * h);

            int padding = (int)((x2 - x1) * 0.2);

            x1 = Math.Max(0, x1 - padding);
            y1 = Math.Max(0, y1 - padding);
            x2 = Math.Min(w - 1, x2 + padding);
            y2 = Math.Min(h - 1, y2 + padding);

            if (x2 <= x1 || y2 <= y1)
                return CenterCrop(mat);

            var rect = new Rect(x1, y1, x2 - x1, y2 - y1);

            return new Mat(mat, rect);
        }

        private Mat CenterCrop(Mat src)
        {
            int size = Math.Min(src.Width, src.Height);

            int x = (src.Width - size) / 2;
            int y = (src.Height - size) / 2;

            return new Mat(src, new Rect(x, y, size, size));
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

        public Mat Align(Mat mat)
        {
            var img = MatToDlib(mat);

            var faces = _detector.Operator(img);

            if (faces.Length == 0)
                return mat;

            var shape = _predictor.Detect(img, faces[0]);

            var leftEye = GetPoint(shape, 36, 41);
            var rightEye = GetPoint(shape, 42, 47);
            var nose = shape.GetPart(30);
            var leftMouth = shape.GetPart(48);
            var rightMouth = shape.GetPart(54);

            var src = new[]
            {
                new Point2f(leftEye.X, leftEye.Y),
                new Point2f(rightEye.X, rightEye.Y),
                new Point2f(nose.X, nose.Y),
                new Point2f(leftMouth.X, leftMouth.Y),
                new Point2f(rightMouth.X, rightMouth.Y)
            };

            var dst = new[]
            {
                new Point2f(38.2946f, 51.6963f),
                new Point2f(73.5318f, 51.5014f),
                new Point2f(56.0252f, 71.7366f),
                new Point2f(41.5493f, 92.3655f),
                new Point2f(70.7299f, 92.2041f)
            };

            var srcMat = new Mat(src.Length, 2, MatType.CV_32F);
            var dstMat = new Mat(dst.Length, 2, MatType.CV_32F);

            for (int i = 0; i < src.Length; i++)
            {
                srcMat.Set(i, 0, src[i].X);
                srcMat.Set(i, 1, src[i].Y);

                dstMat.Set(i, 0, dst[i].X);
                dstMat.Set(i, 1, dst[i].Y);
            }

            var matTransform = Cv2.EstimateAffinePartial2D(srcMat, dstMat);
            //var matTransform = Cv2.EstimateAffinePartial2D(src, dst);

            var aligned = new Mat();
            Cv2.WarpAffine(mat, aligned, matTransform, new Size(112, 112));

            return aligned;
        }

        private Array2D<RgbPixel> MatToDlib(Mat mat)
        {
            var img = new Array2D<RgbPixel>((int)mat.Height, (int)mat.Width);

            for (int y = 0; y < mat.Height; y++)
            {
                for (int x = 0; x < mat.Width; x++)
                {
                    var p = mat.At<Vec3b>(y, x);

                    img[y][x] = new RgbPixel
                    {
                        Red = p.Item2,
                        Green = p.Item1,
                        Blue = p.Item0
                    };
                }
            }

            return img;
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
    }


    // ================= ArcFace =================
    class FaceModel
    {
        private InferenceSession _session;

        public FaceModel(string path)
        {
            _session = new InferenceSession(path);
        }

        public float[] GetFeature(Mat mat)
        {
            var resized = new Mat();
            Cv2.Resize(mat, resized, new Size(112, 112));

            float[] input = new float[1 * 3 * 112 * 112];
            int channelSize = 112 * 112;

            for (int y = 0; y < 112; y++)
            {
                for (int x = 0; x < 112; x++)
                {
                    var pixel = resized.At<Vec3b>(y, x);

                    int idx = y * 112 + x;

                    // RGB
                    input[idx] = (pixel.Item2 - 127.5f) / 128f;     // R
                    input[channelSize + idx] = (pixel.Item1 - 127.5f) / 128f; // G
                    input[channelSize * 2 + idx] = (pixel.Item0 - 127.5f) / 128f; // B
                }
            }

            var tensor = new DenseTensor<float>(input, new[] { 1, 3, 112, 112 });

            var result = _session.Run(new[]
            {
                NamedOnnxValue.CreateFromTensor(_session.InputMetadata.Keys.First(), tensor)
            });

            return Normalize(result.First().AsEnumerable<float>().ToArray());
        }

        public float[] Average(float[] a, float[] b)
        {
            var r = new float[a.Length];

            for (int i = 0; i < a.Length; i++)
                r[i] = (a[i] + b[i]) / 2f;

            return Normalize(r);
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
