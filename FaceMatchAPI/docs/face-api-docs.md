# 얼굴 인식 API 문서

> **Base URL** `http://localhost:5214/api/face`  
> **Content-Type** `application/json`  
> **버전** `v1.1`  
> **최종 수정** 2026-08-03  
> **기준 코드** `FaceMatchAPI/Controllers/FaceController.cs`, `FaceMatchAPI/Dtos/RequestDTO.cs`

---

## 목차

1. [공통 규격](#1-공통-규격)
2. [이미지 업로드 및 벡터 저장](#2-이미지-업로드-및-벡터-저장-post-upload)
3. [유사 얼굴 검색](#3-유사-얼굴-검색-post-search)
4. [지정 벡터 대상 유사 얼굴 검색](#4-지정-벡터-대상-유사-얼굴-검색-post-searchbyvectorids)
5. [이미지 조회](#5-이미지-조회-post-images)
6. [벡터 조회](#6-벡터-조회-post-vectors)
7. [벡터 페이지 조회](#7-벡터-페이지-조회-post-vectorspage)
8. [데이터 삭제](#8-데이터-삭제-post-delete)
9. [오류 응답](#9-오류-응답)
10. [MongoDB 컬렉션 구조](#10-mongodb-컬렉션-구조)
11. [API 전체 목록](#11-api-전체-목록)

---

## 1. 공통 규격

### 1.1 공통 응답 구조 `ResponseDTO<T>`

모든 API 응답은 아래 구조를 사용합니다. 실제 응답 데이터는 `data` 필드에 포함됩니다.

```json
{
  "code": "200",
  "msg": "OK",
  "data": {},
  "success": true
}
```

| 필드 | 타입 | 설명 |
|---|---|---|
| `code` | `string` | 응답 코드. 정상 응답은 `"200"` |
| `msg` | `string` | 결과 메시지 |
| `data` | `T \| null` | 응답 데이터. 실패 시 `null` |
| `success` | `boolean` | 성공 여부 |

### 1.2 `ImageType` / `SearchType`

`imageType`과 `searchType`은 같은 enum 값을 사용합니다. 현재 서버에는 문자열 enum 변환 설정이 없으므로 Request에는 `"Normal"`, `"Target"` 문자열이 아니라 숫자 enum 값을 전달해야 합니다.

| Request 값 | enum 이름 | 설명 | 대상 컬렉션 |
|---|---|---|---|
| `0` | `Normal` | 일반 얼굴 벡터 데이터 | `face_vectors` |
| `1` | `Target` | 비교/대상 얼굴 벡터 데이터 | `target_vectors` |

값을 생략하면 기본값은 `0` (`Normal`)입니다. Response의 `imageType`은 서버 코드에서 `ToString()`을 사용하므로 `"Normal"` 또는 `"Target"` 문자열로 반환됩니다.

### 1.3 Base64 이미지 검증

이미지 입력(`base64`)은 서버에서 Base64 디코딩과 매직 바이트 검증을 수행합니다.

지원 포맷:

| 포맷 | 설명 |
|---|---|
| `JPEG` | JPEG 이미지 |
| `PNG` | PNG 이미지 |
| `GIF` | GIF87a/GIF89a 이미지 |
| `BMP` | BMP 이미지 |
| `WEBP` | RIFF 헤더 기반 WEBP 이미지 |

`data:image/jpeg;base64,` 같은 Data URL prefix가 포함되어도 서버에서 자동으로 제거합니다.

### 1.4 얼굴 처리 파이프라인

업로드와 검색 API는 이미지에서 얼굴 특징 벡터를 추출합니다.

처리 순서:

1. Base64 이미지 디코딩
2. OpenCV DNN 얼굴 검출
3. Dlib 랜드마크 기반 얼굴 정렬
4. ONNX 모델 특징 추출
5. 좌우 반전 이미지 특징 추출
6. 두 특징 벡터 평균 후 정규화

얼굴 검출 또는 정렬 단계에서 실패하면 일부 경우 중앙 crop 또는 원본 이미지로 대체 처리될 수 있습니다. 얼굴 처리 단계에서 예외가 발생하면 `422` 응답을 반환합니다.

---

## 2. 이미지 업로드 및 벡터 저장 `POST /upload`

이미지를 Base64로 전달하면 얼굴 특징 벡터를 추출해 저장합니다. `imageSave`가 `true`이면 원본 Base64 이미지도 `face_images`에 저장합니다.

### Request

```json
{
  "base64": "/9j/4AAQSkZJRgAB...",
  "imageType": 0,
  "imageSave": true,
  "subId": "person-001"
}
```

| 필드 | 타입 | 필수 | 기본값 | 설명 |
|---|---|---|---|---|
| `base64` | `string` | 예 | `""` | Base64 인코딩 이미지 |
| `imageType` | `ImageType` | 아니오 | `0` | 벡터를 저장할 컬렉션 구분. `0=Normal`, `1=Target` |
| `imageSave` | `boolean` | 아니오 | `true` | 원본 이미지를 `face_images`에 저장할지 여부 |
| `subId` | `string` | 아니오 | `""` | 외부 시스템 식별자 또는 사용자 정의 보조 ID |

### Response `200 OK`

```json
{
  "code": "200",
  "msg": "OK",
  "data": {
    "vectorId": "665f1a2b3c4d5e6f7a8b9c0d",
    "imageId": "665f0a1b2c3d4e5f6a7b8c9d",
    "imageType": "Normal"
  },
  "success": true
}
```

| 필드 | 타입 | 설명 |
|---|---|---|
| `vectorId` | `string` | 저장된 벡터 문서의 MongoDB ObjectId |
| `imageId` | `string` | 저장된 이미지 문서의 MongoDB ObjectId. `imageSave=false`이면 빈 문자열 |
| `imageType` | `string` | 저장 대상 컬렉션 구분 |

---

## 3. 유사 얼굴 검색 `POST /search`

입력 이미지에서 특징 벡터를 추출하고, 지정 컬렉션의 저장된 벡터들과 코사인 유사도를 비교합니다.

### Request

```json
{
  "base64": "/9j/4AAQSkZJRgAB...",
  "searchType": 0,
  "startDate": "2026-01-01T00:00:00Z",
  "endDate": "2026-12-31T23:59:59Z",
  "minScore": 0.5
}
```

| 필드 | 타입 | 필수 | 기본값 | 설명 |
|---|---|---|---|---|
| `base64` | `string` | 예 | `""` | 검색할 얼굴 이미지 |
| `searchType` | `ImageType` | 아니오 | `0` | 검색할 벡터 컬렉션. `0=Normal`, `1=Target` |
| `startDate` | `datetime?` | 아니오 | `null` | `createdAt` 조회 시작 시각 |
| `endDate` | `datetime?` | 아니오 | `null` | `createdAt` 조회 종료 시각 |
| `minScore` | `float` | 아니오 | `0.5` | 반환할 최소 코사인 유사도 |

### Response `200 OK`

```json
{
  "code": "200",
  "msg": "OK",
  "data": [
    {
      "id": "665f1a2b3c4d5e6f7a8b9c0d",
      "imageId": "665f0a1b2c3d4e5f6a7b8c9d",
      "subId": "person-001",
      "score": 0.9712
    }
  ],
  "success": true
}
```

| 필드 | 타입 | 설명 |
|---|---|---|
| `data[].id` | `string` | 매칭된 벡터 문서의 ObjectId |
| `data[].imageId` | `string` | 연결된 이미지 문서의 ObjectId. 이미지가 저장되지 않은 벡터는 빈 문자열 |
| `data[].subId` | `string` | 벡터 저장 시 전달한 보조 ID |
| `data[].score` | `float` | Cosine 유사도. 높을수록 유사 |

결과는 `score` 내림차순으로 정렬됩니다. 조건에 맞는 데이터가 없어도 `404`가 아니라 빈 배열을 포함한 `200 OK`가 반환됩니다.

---

## 4. 지정 벡터 대상 유사 얼굴 검색 `POST /search/byVectorIds`

입력 이미지와 지정한 벡터 ID 목록만 비교합니다. `vectorIds`가 비어 있으면 전체 벡터를 대상으로 검색합니다.

### Request

```json
{
  "base64": "/9j/4AAQSkZJRgAB...",
  "searchType": 1,
  "vectorIds": [
    "665f1a2b3c4d5e6f7a8b9c0d",
    "665f2b3c4d5e6f7a8b9c0e1f"
  ],
  "minScore": 0.5
}
```

| 필드 | 타입 | 필수 | 기본값 | 설명 |
|---|---|---|---|---|
| `base64` | `string` | 예 | `""` | 검색할 얼굴 이미지 |
| `searchType` | `ImageType` | 아니오 | `0` | 검색할 벡터 컬렉션. `0=Normal`, `1=Target` |
| `vectorIds` | `string[]` | 아니오 | `[]` | 비교 대상 벡터 ObjectId 목록. 빈 배열이면 전체 검색 |
| `minScore` | `float` | 아니오 | `0.5` | 반환할 최소 코사인 유사도 |

### Response `200 OK`

```json
{
  "code": "200",
  "msg": "OK",
  "data": [
    {
      "id": "665f1a2b3c4d5e6f7a8b9c0d",
      "imageId": "665f0a1b2c3d4e5f6a7b8c9d",
      "subId": "person-001",
      "score": 0.9345
    }
  ],
  "success": true
}
```

주의:

| 조건 | 동작 |
|---|---|
| `vectorIds`가 빈 배열 | 전체 벡터 검색 |
| `vectorIds`가 있지만 유효한 ObjectId가 하나도 없음 | `400`, `"유효한 VectorIds가 없습니다."` |
| 일부 ID만 ObjectId 형식이 아님 | 유효한 ID만 검색에 사용되고, 잘못된 ID는 무시됨 |

---

## 5. 이미지 조회 `POST /images`

`face_images` 컬렉션에서 이미지 ObjectId 목록으로 이미지를 조회합니다.

### Request

```json
{
  "ids": [
    "665f0a1b2c3d4e5f6a7b8c9d",
    "665f0a1b2c3d4e5f6a7b8c9e"
  ]
}
```

| 필드 | 타입 | 필수 | 기본값 | 설명 |
|---|---|---|---|---|
| `ids` | `string[]` | 예 | `[]` | 조회할 이미지 ObjectId 목록 |

### Response `200 OK`

```json
{
  "code": "200",
  "msg": "OK",
  "data": {
    "requestedCount": 2,
    "foundCount": 1,
    "notFoundIds": [
      "665f0a1b2c3d4e5f6a7b8c9e"
    ],
    "images": [
      {
        "id": "665f0a1b2c3d4e5f6a7b8c9d",
        "base64": "/9j/4AAQSkZJRgAB...",
        "createdAt": "2026-08-03T00:00:00Z"
      }
    ]
  },
  "success": true
}
```

모든 ID가 조회되지 않으면 `404`를 반환합니다. 일부만 조회된 경우에는 `200 OK`와 함께 `notFoundIds`가 포함됩니다.

---

## 6. 벡터 조회 `POST /vectors`

`face_vectors` 또는 `target_vectors` 컬렉션에서 벡터 ObjectId 목록으로 벡터를 조회합니다.

### Request

```json
{
  "imageType": 1,
  "vectorIds": [
    "665f1a2b3c4d5e6f7a8b9c0d"
  ]
}
```

| 필드 | 타입 | 필수 | 기본값 | 설명 |
|---|---|---|---|---|
| `imageType` | `ImageType` | 아니오 | `0` | 조회할 벡터 컬렉션. `0=Normal`, `1=Target` |
| `vectorIds` | `string[]` | 예 | `[]` | 조회할 벡터 ObjectId 목록 |

### Response `200 OK`

```json
{
  "code": "200",
  "msg": "OK",
  "data": {
    "requestedCount": 1,
    "foundCount": 1,
    "imageType": "Target",
    "notFoundIds": [],
    "vectors": [
      {
        "id": "665f1a2b3c4d5e6f7a8b9c0d",
        "imageId": "665f0a1b2c3d4e5f6a7b8c9d",
        "vector": [0.12, 0.85, 0.33],
        "createdAt": "2026-08-03T00:00:00Z"
      }
    ]
  },
  "success": true
}
```

현재 `/vectors` 응답에는 `subId`가 포함되지 않습니다. `subId`까지 필요하면 `/vectors/page`를 사용합니다.

---

## 7. 벡터 페이지 조회 `POST /vectors/page`

벡터 컬렉션을 페이지 단위로 조회합니다. 결과는 `createdAt` 내림차순입니다.

### Request

```json
{
  "imageType": 0,
  "startDate": "2026-01-01T00:00:00Z",
  "endDate": "2026-12-31T23:59:59Z",
  "vectorId": "665f1a2b3c4d5e6f7a8b9c0d",
  "subId": "person-001",
  "page": 1,
  "pageSize": 20
}
```

| 필드 | 타입 | 필수 | 기본값 | 제약 | 설명 |
|---|---|---|---|---|---|
| `imageType` | `ImageType` | 아니오 | `0` | - | 조회할 벡터 컬렉션. `0=Normal`, `1=Target` |
| `startDate` | `datetime?` | 아니오 | `null` | - | `createdAt` 조회 시작 시각 |
| `endDate` | `datetime?` | 아니오 | `null` | `startDate` 이상 | `createdAt` 조회 종료 시각 |
| `vectorId` | `string` | 아니오 | `""` | ObjectId 형식 | 특정 벡터 ID 필터 |
| `subId` | `string` | 아니오 | `""` | - | 특정 보조 ID 필터 |
| `page` | `int` | 아니오 | `1` | `>= 1` | 페이지 번호 |
| `pageSize` | `int` | 아니오 | `20` | `1 ~ 100` | 페이지당 반환 개수 |

### Response `200 OK`

```json
{
  "code": "200",
  "msg": "OK",
  "data": {
    "page": 1,
    "pageSize": 20,
    "totalCount": 53,
    "totalPages": 3,
    "hasPrevious": false,
    "hasNext": true,
    "imageType": "Normal",
    "vectors": [
      {
        "id": "665f1a2b3c4d5e6f7a8b9c0d",
        "imageId": "665f0a1b2c3d4e5f6a7b8c9d",
        "subId": "person-001",
        "vector": [0.12, 0.85, 0.33],
        "createdAt": "2026-08-03T00:00:00Z"
      }
    ]
  },
  "success": true
}
```

주의:

| 조건 | 동작 |
|---|---|
| 조회 결과가 없음 | `404`, `"조회된 벡터 데이터가 없습니다."` |
| `vectorId`가 ObjectId 형식이 아님 | 현재 컨트롤러에서 직접 `ObjectId.Parse` 예외가 발생해 `500`으로 처리될 수 있음 |

---

## 8. 데이터 삭제 `POST /delete`

벡터와 해당 벡터가 참조하는 이미지를 삭제합니다. 삭제 기준은 `ids` 또는 `subIds`입니다.

### Request: 벡터 ID로 삭제

```json
{
  "imageType": 0,
  "ids": [
    "665f1a2b3c4d5e6f7a8b9c0d"
  ],
  "subIds": []
}
```

### Request: SubId로 삭제

```json
{
  "imageType": 0,
  "ids": [],
  "subIds": [
    "person-001",
    "person-002"
  ]
}
```

| 필드 | 타입 | 필수 | 기본값 | 설명 |
|---|---|---|---|---|
| `imageType` | `ImageType` | 아니오 | `0` | 삭제 대상 벡터 컬렉션. `0=Normal`, `1=Target` |
| `ids` | `string[]` | 조건부 | `[]` | 삭제할 벡터 ObjectId 목록 |
| `subIds` | `string[]` | 조건부 | `[]` | 삭제할 보조 ID 목록 |

`subIds`가 하나 이상 전달되면 `ids`보다 `subIds`가 우선 사용됩니다.

### Response `200 OK`

```json
{
  "code": "200",
  "msg": "삭제가 완료되었습니다.",
  "data": {
    "requestedCount": 2,
    "deletedCount": 1,
    "imageDeletedCount": 1,
    "notFoundIds": [
      "person-002"
    ]
  },
  "success": true
}
```

| 필드 | 타입 | 설명 |
|---|---|---|
| `requestedCount` | `int` | 요청한 ID 또는 SubId 개수 |
| `deletedCount` | `int` | 삭제된 벡터 개수 |
| `imageDeletedCount` | `long` | 삭제된 이미지 개수 |
| `notFoundIds` | `string[]` | 찾지 못한 ID 또는 SubId 목록 |

모든 대상이 존재하지 않으면 `404`를 반환합니다.

---

## 9. 오류 응답

### 오류 응답 예시

```json
{
  "code": "400",
  "msg": "유효하지 않은 Base64 형식입니다.",
  "data": null,
  "success": false
}
```

### 주요 오류 코드

| HTTP | `code` | 발생 조건 |
|---|---|---|
| `400` | `"400"` | `base64`가 비어 있음 |
| `400` | `"400"` | Base64 형식이 유효하지 않음 |
| `400` | `"400"` | 이미지 데이터가 너무 짧음 |
| `400` | `"400"` | 지원하지 않는 이미지 형식 |
| `400` | `"400"` | 조회/삭제 ID 목록이 비어 있음 |
| `400` | `"400"` | ObjectId 형식이 유효하지 않음 |
| `400` | `"400"` | `page < 1` |
| `400` | `"400"` | `pageSize < 1` 또는 `pageSize > 100` |
| `400` | `"400"` | `startDate > endDate` (`/vectors/page`) |
| `404` | `"404"` | 이미지, 벡터, 삭제 대상 데이터가 하나도 조회되지 않음 |
| `422` | `"422"` | 얼굴 처리 파이프라인 실패 |
| `499` | `"499"` | 클라이언트가 업로드/검색 요청을 취소함 |
| `500` | `"500"` | 서버 내부 오류 |

### 얼굴 처리 실패 응답 예시

```json
{
  "code": "422",
  "msg": "얼굴 처리 실패 [FeatureExtraction]: ONNX 추론 실패: ...",
  "data": null,
  "success": false
}
```

---

## 10. MongoDB 컬렉션 구조

데이터베이스 이름은 `face_db`입니다. MongoDB 연결 문자열은 환경 변수 `MONGODB_CONNECTION_STR`에서 읽습니다.

### `face_images`

| 필드 | 타입 | 설명 |
|---|---|---|
| `_id` | `ObjectId` | 이미지 문서 ID |
| `base64` | `string` | Base64 인코딩 이미지 데이터 |
| `createdAt` | `datetime` | 생성 시각 UTC |

### `face_vectors` / `target_vectors`

| 필드 | 타입 | 설명 |
|---|---|---|
| `_id` | `ObjectId` | 벡터 문서 ID |
| `createdAt` | `datetime` | 생성 시각 UTC |
| `imageId` | `string` | 연결된 `face_images._id`. 이미지 미저장 시 빈 문자열 |
| `subId` | `string` | 외부 시스템 식별자 또는 사용자 정의 보조 ID |
| `vector` | `float[]` | 얼굴 특징 벡터 |

---

## 11. API 전체 목록

| Method | Endpoint | 설명 |
|---|---|---|
| `POST` | `/upload` | 이미지에서 얼굴 벡터를 추출해 저장 |
| `POST` | `/search` | 전체 또는 기간 필터 기반 유사 얼굴 검색 |
| `POST` | `/search/byVectorIds` | 지정 벡터 ID 목록 대상 유사 얼굴 검색 |
| `POST` | `/images` | 이미지 ObjectId 기반 원본 이미지 조회 |
| `POST` | `/vectors` | 벡터 ObjectId 기반 벡터 조회 |
| `POST` | `/vectors/page` | 벡터 페이지 조회 및 `vectorId`/`subId` 필터 조회 |
| `POST` | `/delete` | 벡터 ID 또는 SubId 기반 벡터와 연결 이미지 삭제 |

---

## 12. 실행 및 설정 참고

| 항목 | 값 |
|---|---|
| 로컬 HTTP URL | `http://localhost:5214` |
| Swagger UI | `http://localhost:5214/swagger` |
| MongoDB 연결 환경 변수 | `MONGODB_CONNECTION_STR` |
| 얼굴 처리 풀 설정 | `FacePool:Size` (`appsettings.json` 기본값 `4`) |
| 모델 파일 위치 | 애플리케이션 `Models` 디렉터리 |
