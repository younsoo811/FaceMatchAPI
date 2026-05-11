# 안면 인식 API 문서

> **Base URL** `http://localhost:5000/api/face/`  
> **Content-Type** `application/json`  
> **버전** `v1.0`  
> **최종 수정** 2026-05-11

---

## 목차

1. [공통 규격](#1-공통-규격)
2. [이미지 업로드](#2-이미지-업로드-post-upload)
3. [유사 얼굴 검색](#3-유사-얼굴-검색-post-search)
4. [데이터 삭제](#4-데이터-삭제-post-delete)
5. [이미지 조회](#5-이미지-조회-post-images)
6. [벡터 ID 조회](#6-벡터-id-조회-post-vectors)
7. [벡터 페이징 조회](#7-벡터-페이징-조회-post-vectorspage)
8. [에러 코드](#8-에러-코드)

---

## 1. 공통 규격

### 1.1 공통 응답 구조 `ResponseDTO<T>`

모든 API 응답은 아래 구조를 공통으로 사용합니다. 실제 데이터는 항상 `data` 필드 안에 포함됩니다.

```json
{
  "code":    "200",
  "msg":     "OK",
  "data":    { ... },
  "success": true
}
```

| 필드 | 타입 | 설명 |
|---|---|---|
| `code` | `string` | HTTP 상태 코드 문자열 |
| `msg` | `string` | 결과 메시지 |
| `data` | `T \| null` | 응답 데이터. 실패 시 `null` |
| `success` | `boolean` | 성공 여부 |

---

### 1.2 ImageType / SearchType

| 값 | 설명 | 저장 컬렉션 |
|---|---|---|
| `"Normal"` (기본값) | 일반 안면 데이터 | `face_vectors` |
| `"Target"` | 비교 대상(타깃) 데이터 | `target_vectors` |

---

### 1.3 Base64 이미지 유효성 검사

모든 이미지 입력(`base64`)은 서버에서 매직 바이트(Magic Bytes) 방식으로 실제 이미지 포맷을 검증합니다.  
지원 포맷: **JPEG · PNG · BMP · WEBP · GIF**  
`data:image/jpeg;base64,` 형태의 Data URL prefix는 자동으로 제거됩니다.

---

## 2. 이미지 업로드 `POST /upload`

안면 이미지를 Base64로 전송하면 이미지와 안면 벡터를 함께 저장합니다.

### Request

```json
{
  "base64":    "<Base64 인코딩된 이미지>",
  "imageType": "Normal"
}
```

| 필드 | 타입 | 필수 | 기본값 | 설명 |
|---|---|---|---|---|
| `base64` | `string` | ✅ | — | Base64 인코딩된 이미지 |
| `imageType` | `ImageType` | ❌ | `Normal` | 저장할 컬렉션 구분 |

### Response `200 OK`

```json
{
  "code": "200",
  "msg": "OK",
  "data": {
    "imageId":   "665f1a2b3c4d5e6f7a8b9c0d",
    "imageType": "Normal",
    "format":    "JPEG"
  },
  "success": true
}
```

| 필드 | 타입 | 설명 |
|---|---|---|
| `imageId` | `string` | 저장된 이미지의 MongoDB ObjectId |
| `imageType` | `string` | 저장된 컬렉션 타입 |
| `format` | `string` | 감지된 이미지 포맷 |

### 처리 흐름

```
요청 수신
  → Base64 유효성 검사 (형식, 이미지 포맷)
  → FaceImages 컬렉션에 이미지 저장
  → 안면 벡터 추출 (ExtractFeatureWithFlip)
  → ImageType에 따라 FaceVectors 또는 TargetVectors에 벡터 저장
  → 저장된 imageId 반환
```

---

## 3. 유사 얼굴 검색 `POST /search`

입력 이미지와 저장된 벡터를 코사인 유사도로 비교하여 일치하는 데이터를 반환합니다.

### Request

```json
{
  "base64":     "<Base64 인코딩된 이미지>",
  "searchType": "Normal",
  "startDate":  "2025-01-01T00:00:00Z",
  "endDate":    "2025-12-31T23:59:59Z",
  "minScore":   0.5
}
```

| 필드 | 타입 | 필수 | 기본값 | 설명 |
|---|---|---|---|---|
| `base64` | `string` | ✅ | — | 검색에 사용할 이미지 |
| `searchType` | `SearchType` | ❌ | `Normal` | 검색할 컬렉션 구분 |
| `startDate` | `datetime` | ❌ | — | 등록일 범위 시작 (ISO 8601) |
| `endDate` | `datetime` | ❌ | — | 등록일 범위 종료 (ISO 8601) |
| `minScore` | `float` | ❌ | `0.5` | 반환할 최소 유사도 (0.0 ~ 1.0) |

### Response `200 OK`

```json
{
  "code": "200",
  "msg": "OK",
  "data": [
    { "imageId": "665f1a...", "score": 0.97 },
    { "imageId": "665f2b...", "score": 0.83 }
  ],
  "success": true
}
```

| 필드 | 타입 | 설명 |
|---|---|---|
| `imageId` | `string` | 매칭된 이미지의 ObjectId |
| `score` | `float` | 코사인 유사도 (0.0 ~ 1.0). 높을수록 유사 |

> 결과는 `score` 내림차순으로 정렬됩니다.

---

## 4. 데이터 삭제 `POST /delete`

벡터 데이터와 연결된 이미지 데이터를 함께 삭제합니다. 여러 ID를 한 번에 처리할 수 있습니다.

### Request

```json
{
  "imageType": "Normal",
  "ids": [
    "665f1a2b3c4d5e6f7a8b9c0d",
    "665f2b3c4d5e6f7a8b9c0e1f"
  ]
}
```

| 필드 | 타입 | 필수 | 기본값 | 설명 |
|---|---|---|---|---|
| `imageType` | `ImageType` | ❌ | `Normal` | 삭제할 컬렉션 구분 |
| `ids` | `string[]` | ✅ | — | 삭제할 벡터 ObjectId 목록 (1개 이상) |

### Response `200 OK`

```json
{
  "code": "200",
  "msg": "삭제가 완료되었습니다.",
  "data": {
    "requestedCount": 3,
    "deletedCount":   2,
    "notFoundIds":    ["665f3c4d5e6f7a8b9c0f1a2b"]
  },
  "success": true
}
```

| 필드 | 타입 | 설명 |
|---|---|---|
| `requestedCount` | `int` | 요청한 총 ID 수 |
| `deletedCount` | `int` | 실제 삭제된 건수 |
| `notFoundIds` | `string[]` | 요청했으나 존재하지 않았던 ID 목록 |

### 처리 흐름

```
ids 목록 수신
  → ObjectId 형식 유효성 검사
  → ImageType에 따라 컬렉션 선택
  → 해당 컬렉션에서 ids 조회
  → 조회된 벡터의 ImageId 수집
  → FaceVectors(또는 TargetVectors) DeleteMany
  → FaceImages DeleteMany (ImageId 기준)
  → 삭제 결과 반환
```

---

## 5. 이미지 조회 `POST /images`

`FaceImages` 컬렉션에서 ID 목록으로 이미지 데이터를 조회합니다.

### Request

```json
{
  "ids": [
    "665f1a2b3c4d5e6f7a8b9c0d",
    "665f2b3c4d5e6f7a8b9c0e1f"
  ]
}
```

| 필드 | 타입 | 필수 | 설명 |
|---|---|---|---|
| `ids` | `string[]` | ✅ | 조회할 FaceImage ObjectId 목록 (1개 이상) |

### Response `200 OK`

```json
{
  "code": "200",
  "msg": "OK",
  "data": {
    "requestedCount": 2,
    "foundCount":     2,
    "notFoundIds":    [],
    "images": [
      {
        "id":        "665f1a2b3c4d5e6f7a8b9c0d",
        "base64":    "/9j/4AAQSkZJRgAB...",
        "createdAt": "2025-01-01T00:00:00Z"
      }
    ]
  },
  "success": true
}
```

| 필드 | 타입 | 설명 |
|---|---|---|
| `requestedCount` | `int` | 요청한 총 ID 수 |
| `foundCount` | `int` | 실제 조회된 건수 |
| `notFoundIds` | `string[]` | 요청했으나 존재하지 않았던 ID 목록 |
| `images[].id` | `string` | 이미지 ObjectId |
| `images[].base64` | `string` | Base64 인코딩된 이미지 데이터 |
| `images[].createdAt` | `datetime` | 등록 일시 (UTC) |

---

## 6. 벡터 ID 조회 `POST /vectors`

`FaceVectors` 또는 `TargetVectors` 컬렉션에서 VectorId 목록으로 벡터 데이터를 조회합니다.

### Request

```json
{
  "imageType": "Target",
  "vectorIds": [
    "665f1a2b3c4d5e6f7a8b9c0d",
    "665f2b3c4d5e6f7a8b9c0e1f"
  ]
}
```

| 필드 | 타입 | 필수 | 기본값 | 설명 |
|---|---|---|---|---|
| `imageType` | `ImageType` | ❌ | `Normal` | 조회할 컬렉션 구분 |
| `vectorIds` | `string[]` | ✅ | — | 조회할 벡터 ObjectId 목록 (1개 이상) |

### Response `200 OK`

```json
{
  "code": "200",
  "msg": "OK",
  "data": {
    "requestedCount": 2,
    "foundCount":     2,
    "imageType":      "Target",
    "notFoundIds":    [],
    "vectors": [
      {
        "id":        "665f1a2b3c4d5e6f7a8b9c0d",
        "imageId":   "665f0a1b2c3d4e5f6a7b8c9d",
        "vector":    [0.12, 0.85, 0.33, "..."],
        "createdAt": "2025-01-01T00:00:00Z"
      }
    ]
  },
  "success": true
}
```

| 필드 | 타입 | 설명 |
|---|---|---|
| `requestedCount` | `int` | 요청한 총 ID 수 |
| `foundCount` | `int` | 실제 조회된 건수 |
| `imageType` | `string` | 조회된 컬렉션 타입 |
| `notFoundIds` | `string[]` | 요청했으나 존재하지 않았던 ID 목록 |
| `vectors[].id` | `string` | 벡터 ObjectId |
| `vectors[].imageId` | `string` | 연결된 FaceImage ObjectId |
| `vectors[].vector` | `float[]` | 안면 특징 벡터 배열 |
| `vectors[].createdAt` | `datetime` | 등록 일시 (UTC) |

---

## 7. 벡터 페이징 조회 `POST /vectors/page`

`FaceVectors` 또는 `TargetVectors` 컬렉션의 데이터를 페이지 단위로 조회합니다.  
`createdAt` 내림차순으로 정렬됩니다.

### Request

```json
{
  "imageType": "Normal",
  "startDate": "2025-01-01T00:00:00Z",
  "endDate":   "2025-12-31T23:59:59Z",
  "page":      1,
  "pageSize":  20
}
```

| 필드 | 타입 | 필수 | 기본값 | 제약 | 설명 |
|---|---|---|---|---|---|
| `imageType` | `ImageType` | ❌ | `Normal` | — | 조회할 컬렉션 구분 |
| `startDate` | `datetime` | ❌ | — | — | 등록일 범위 시작 (ISO 8601) |
| `endDate` | `datetime` | ❌ | — | `>= startDate` | 등록일 범위 종료 (ISO 8601) |
| `page` | `int` | ❌ | `1` | `>= 1` | 조회할 페이지 번호 (1부터 시작) |
| `pageSize` | `int` | ❌ | `20` | `1 ~ 100` | 페이지당 반환 건수 |

### Response `200 OK`

```json
{
  "code": "200",
  "msg": "OK",
  "data": {
    "page":        2,
    "pageSize":    10,
    "totalCount":  53,
    "totalPages":  6,
    "hasPrevious": true,
    "hasNext":     true,
    "imageType":   "Normal",
    "vectors": [
      {
        "id":        "665f1a2b3c4d5e6f7a8b9c0d",
        "imageId":   "665f0a1b2c3d4e5f6a7b8c9d",
        "vector":    [0.12, 0.85, 0.33, "..."],
        "createdAt": "2025-01-05T12:00:00Z"
      }
    ]
  },
  "success": true
}
```

| 필드 | 타입 | 설명 |
|---|---|---|
| `page` | `int` | 현재 페이지 번호 |
| `pageSize` | `int` | 페이지당 데이터 수 |
| `totalCount` | `long` | 필터 조건에 해당하는 전체 데이터 수 |
| `totalPages` | `int` | 전체 페이지 수 |
| `hasPrevious` | `boolean` | 이전 페이지 존재 여부 |
| `hasNext` | `boolean` | 다음 페이지 존재 여부 |
| `imageType` | `string` | 조회된 컬렉션 타입 |
| `vectors` | `VectorItem[]` | 조회된 벡터 목록 |

---

## 8. 에러 코드

| code | HTTP 상태 | 발생 조건 |
|---|---|---|
| `400` | Bad Request | 유효하지 않은 Base64 형식 |
| `400` | Bad Request | 지원하지 않는 이미지 포맷 |
| `400` | Bad Request | 유효하지 않은 ObjectId 형식 |
| `400` | Bad Request | ID 목록이 비어있음 |
| `400` | Bad Request | `page < 1` 또는 `pageSize` 범위 초과 |
| `400` | Bad Request | `startDate > endDate` |
| `404` | Not Found | 일치하는 데이터 없음 |
| `500` | Internal Server Error | 서버 내부 오류 |

### 에러 응답 예시

```json
{
  "code":    "400",
  "msg":     "지원하지 않는 파일 형식입니다. (JPEG, PNG, GIF, BMP, WEBP만 허용)",
  "data":    null,
  "success": false
}
```

---

## 9. MongoDB 컬렉션 구조

### `face_images`

| 필드 | 타입 | 설명 |
|---|---|---|
| `_id` | `ObjectId` | 문서 고유 ID |
| `base64` | `string` | Base64 인코딩된 이미지 데이터 |
| `createdAt` | `datetime` | 등록 일시 (UTC) |

### `face_vectors` / `target_vectors`

| 필드 | 타입 | 설명 |
|---|---|---|
| `_id` | `ObjectId` | 문서 고유 ID |
| `imageId` | `string` | 연결된 `face_images._id` 참조 |
| `vector` | `float[]` | 안면 특징 벡터 배열 |
| `createdAt` | `datetime` | 등록 일시 (UTC) |

---

## 10. API 전체 목록

| Method | Endpoint | 설명 |
|---|---|---|
| `POST` | `/upload` | 이미지 + 벡터 저장 |
| `POST` | `/search` | 유사 얼굴 검색 |
| `POST` | `/delete` | 벡터 + 이미지 삭제 |
| `POST` | `/images` | FaceImages ID 기반 조회 |
| `POST` | `/vectors` | VectorId 기반 조회 |
| `POST` | `/vectors/page` | 벡터 페이징 조회 |
