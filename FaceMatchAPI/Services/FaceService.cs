using DlibDotNet;
using DlibDotNet.Extensions;
using FaceMatchAPI.Exceptions;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using OpenCvSharp.Dnn;
using System.Collections.Concurrent;
using System.Runtime.InteropServices;
using Drawing = System.Drawing;

namespace FaceMatchAPI.Services
{
    // ================================================================
    //  FaceProcessor - 얼굴 처리 파이프라인 인스턴스 1개
    //  Pool에 N개가 존재하며, 각 인스턴스는 단일 스레드에서만 사용됩니다.
    //  (스레드 안전 문제 없음)
    // ================================================================
    public class FaceProcessor
    {
        private readonly FaceDetector _detector;
        private readonly FaceAligner _aligner;
        private readonly FaceModel _model;

        public FaceProcessor(string baseDir, ILogger logger)
        {
            _detector = new FaceDetector(
                Path.Combine(baseDir, "deploy.prototxt"),
                Path.Combine(baseDir, "res10_300x300_ssd_iter_140000.caffemodel"),
                logger);

            _aligner = new FaceAligner(
                Path.Combine(baseDir, "shape_predictor_68_face_landmarks.dat"),
                logger);

            _model = new FaceModel(
                Path.Combine(baseDir, "model.onnx"),
                logger);
        }

        public float[] ExtractFeatureWithFlip(string base64, ILogger logger)
        {
            Mat? mat = null;
            Mat? face = null;
            Mat? aligned = null;
            Mat? flipped = null;

            try
            {
                try { mat = Base64ToMat(base64); }
                catch (Exception ex) { throw new FaceProcessingException("Decode", "Base64 이미지 디코딩 실패", ex); }

                try
                {
                    logger.LogDebug("[FaceProcessor] 얼굴 검출 시작");
                    face = _detector.Detect(mat);
                }
                catch (FaceProcessingException) { throw; }
                catch (Exception ex) { throw new FaceProcessingException("Detection", "얼굴 검출 중 오류 발생", ex); }

                try
                {
                    logger.LogDebug("[FaceProcessor] 얼굴 정렬 시작");
                    aligned = _aligner.Align(face);
                }
                catch (FaceProcessingException) { throw; }
                catch (Exception ex) { throw new FaceProcessingException("Alignment", "얼굴 정렬 중 오류 발생", ex); }

                float[] f1;
                try { f1 = _model.GetFeature(aligned); }
                catch (FaceProcessingException) { throw; }
                catch (Exception ex) { throw new FaceProcessingException("FeatureExtraction", "원본 특징 추출 중 오류 발생", ex); }

                float[] f2;
                try
                {
                    flipped = new Mat();
                    Cv2.Flip(aligned, flipped, FlipMode.Y);
                    f2 = _model.GetFeature(flipped);
                }
                catch (FaceProcessingException) { throw; }
                catch (Exception ex) { throw new FaceProcessingException("FeatureExtraction", "반전 이미지 특징 추출 중 오류 발생", ex); }

                return _model.Average(f1, f2);
            }
            finally
            {
                mat?.Dispose();
                face?.Dispose();
                aligned?.Dispose();
                flipped?.Dispose();
            }
        }

        private static Mat Base64ToMat(string base64)
        {
            if (base64.Contains(",")) base64 = base64.Split(',')[1];
            var bytes = Convert.FromBase64String(base64);
            var mat = Cv2.ImDecode(bytes, ImreadModes.Color);
            if (mat == null || mat.Empty())
                throw new FaceProcessingException("Decode", "이미지 디코딩 결과가 비어 있습니다.");
            return mat;
        }
    }

    // ================================================================
    //  FaceService - FaceProcessor Object Pool 래퍼
    //  서버 시작 시 PoolSize개의 FaceProcessor를 미리 생성해둡니다.
    //  요청이 오면 가용 인스턴스를 빌려 처리하고 반환합니다.
    //  Pool이 모두 사용 중이면 가용 인스턴스가 생길 때까지 대기합니다.
    // ================================================================
    public class FaceService : IDisposable
    {
        private readonly ILogger<FaceService> _logger;
        private readonly ConcurrentBag<FaceProcessor> _pool;
        private readonly SemaphoreSlim _semaphore;
        private readonly int _poolSize;

        public FaceService(IWebHostEnvironment env, ILogger<FaceService> logger, IConfiguration config)
        {
            _logger = logger;

            // appsettings.json의 "FacePool:Size" 값 사용, 없으면 CPU 코어 수 기준
            _poolSize = config.GetValue<int>("FacePool:Size");
            if (_poolSize <= 0)
                _poolSize = Math.Max(2, Environment.ProcessorCount);

            _pool = new ConcurrentBag<FaceProcessor>();
            _semaphore = new SemaphoreSlim(_poolSize, _poolSize);

            string baseDir = Path.Combine(env.ContentRootPath, "Models");

            _logger.LogInformation("[FaceService] FaceProcessor Pool 초기화 시작 (PoolSize={Size})", _poolSize);

            // 서버 시작 시 PoolSize개 인스턴스를 미리 생성 (모델 로딩)
            for (int i = 0; i < _poolSize; i++)
            {
                _pool.Add(new FaceProcessor(baseDir, logger));
                _logger.LogInformation("[FaceService] FaceProcessor [{Idx}/{Total}] 생성 완료", i + 1, _poolSize);
            }

            _logger.LogInformation("[FaceService] Pool 초기화 완료. 최대 {Size}개 요청 병렬 처리 가능.", _poolSize);
        }

        public async Task<float[]> ExtractFeatureWithFlipAsync(string base64, CancellationToken ct = default)
        {
            // Pool에 가용 슬롯이 생길 때까지 대기
            await _semaphore.WaitAsync(ct);

            if (!_pool.TryTake(out var processor))
            {
                // 안전망: 세마포어 슬롯은 있는데 bag이 비어 있는 예외 상황
                _logger.LogError("[FaceService] Pool에서 인스턴스를 가져오지 못했습니다.");
                _semaphore.Release();
                throw new InvalidOperationException("FaceProcessor Pool 오류: 가용 인스턴스 없음");
            }

            try
            {
                _logger.LogDebug("[FaceService] FaceProcessor 대여 (pool 잔여={Remaining})", _semaphore.CurrentCount);
                return processor.ExtractFeatureWithFlip(base64, _logger);
            }
            catch (FaceProcessingException fpEx)
            {
                _logger.LogWarning(fpEx, "[FaceService] 얼굴 처리 실패 | Stage={Stage}", fpEx.Stage);
                throw;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[FaceService] 예상치 못한 오류 | {ExType}: {Message}",
                    ex.GetType().Name, ex.Message);
                throw;
            }
            finally
            {
                // 성공/실패 관계없이 반드시 반환
                _pool.Add(processor);
                _semaphore.Release();
                _logger.LogDebug("[FaceService] FaceProcessor 반환 (pool 잔여={Remaining})", _semaphore.CurrentCount);
            }
        }

        public void Dispose()
        {
            _semaphore.Dispose();
        }
    }


    // ================= 얼굴 검출 =================
    class FaceDetector
    {
        private readonly Net _net;
        private readonly ILogger _logger;

        public FaceDetector(string proto, string model, ILogger logger)
        {
            _logger = logger;

            if (!File.Exists(proto))
                throw new FileNotFoundException($"prototxt 파일 없음: {proto}");

            if (!File.Exists(model))
                throw new FileNotFoundException($"caffemodel 파일 없음: {model}");

            _net = CvDnn.ReadNetFromCaffe(proto, model);

            if (_net.Empty())
                throw new InvalidOperationException("DNN 모델 로딩 실패");
        }

        public Mat Detect(Mat mat)
        {
            if (mat == null || mat.Empty())
                throw new FaceProcessingException("Detection", "입력 이미지가 비어 있습니다.");

            // 4채널 → 3채널
            if (mat.Channels() == 4)
                Cv2.CvtColor(mat, mat, ColorConversionCodes.BGRA2BGR);

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
                _logger.LogWarning("[FaceDetector] 얼굴 검출 실패 (maxConf={Conf:F4}) → 중앙 crop으로 대체", maxConf);
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
            {
                _logger.LogWarning("[FaceDetector] 검출된 영역이 유효하지 않음 (x1={X1}, y1={Y1}, x2={X2}, y2={Y2}) → 중앙 crop으로 대체",
                    x1, y1, x2, y2);
                return CenterCrop(mat);
            }

            _logger.LogDebug("[FaceDetector] 얼굴 검출 성공 (conf={Conf:F4}, rect=[{X1},{Y1},{X2},{Y2}])",
                maxConf, x1, y1, x2, y2);

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
        private readonly FrontalFaceDetector _detector;
        private readonly ShapePredictor _predictor;
        private readonly ILogger _logger;

        public FaceAligner(string modelPath, ILogger logger)
        {
            _logger = logger;

            if (!File.Exists(modelPath))
                throw new FileNotFoundException($"shape_predictor 파일 없음: {modelPath}");

            _detector = Dlib.GetFrontalFaceDetector();
            _predictor = ShapePredictor.Deserialize(modelPath);
        }

        public Mat Align(Mat mat)
        {
            var img = MatToDlib(mat);

            var faces = _detector.Operator(img);

            if (faces.Length == 0)
            {
                _logger.LogWarning("[FaceAligner] Dlib에서 얼굴 랜드마크 검출 실패 → 정렬 없이 원본 반환 (뒷모습 또는 옆모습일 수 있음)");
                return mat;
            }

            _logger.LogDebug("[FaceAligner] 얼굴 랜드마크 검출 성공 (faces={Count})", faces.Length);

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
        private readonly InferenceSession _session;
        private readonly ILogger _logger;

        public FaceModel(string path, ILogger logger)
        {
            _logger = logger;

            if (!File.Exists(path))
                throw new FileNotFoundException($"ONNX 모델 파일 없음: {path}");

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

                    // RGB 정규화
                    input[idx] = (pixel.Item2 - 127.5f) / 128f;
                    input[channelSize + idx] = (pixel.Item1 - 127.5f) / 128f;
                    input[channelSize * 2 + idx] = (pixel.Item0 - 127.5f) / 128f;
                }
            }

            var tensor = new DenseTensor<float>(input, new[] { 1, 3, 112, 112 });

            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> result;
            try
            {
                result = _session.Run(new[]
                {
                    NamedOnnxValue.CreateFromTensor(_session.InputMetadata.Keys.First(), tensor)
                });
            }
            catch (Exception ex)
            {
                throw new FaceProcessingException("FeatureExtraction", $"ONNX 추론 실패: {ex.Message}", ex);
            }

            var features = Normalize(result.First().AsEnumerable<float>().ToArray());
            _logger.LogDebug("[FaceModel] 추론 완료 (dim={Dim})", features.Length);
            return features;
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