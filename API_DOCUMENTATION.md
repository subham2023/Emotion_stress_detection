# API Documentation

Complete reference for all tRPC endpoints in the Emotion Stress Detection System.

## Authentication

All protected endpoints require a valid session cookie obtained through Manus OAuth. Public endpoints are accessible without authentication.

### Session Management

Sessions are managed via HTTP-only cookies. The authentication flow:

1. User clicks login button
2. Redirected to Manus OAuth portal
3. After authentication, callback sets session cookie
4. Session cookie automatically included in all requests

## Base URL

```
https://api.emotion-detector.ai/api/trpc
```

## Response Format

All responses follow the tRPC format:

```json
{
  "result": {
    "data": { /* response data */ }
  }
}
```

Errors return:

```json
{
  "error": {
    "code": "INTERNAL_SERVER_ERROR",
    "message": "Error description"
  }
}
```

---

## Endpoints

### Authentication

#### `auth.me`

Get current authenticated user information.

**Type**: Query (Public)

**Response**:
```typescript
{
  id: number;
  openId: string;
  name: string | null;
  email: string | null;
  role: 'user' | 'admin';
  createdAt: Date;
  lastSignedIn: Date;
}
```

**Example**:
```typescript
const user = await trpc.auth.me.useQuery();
```

---

#### `auth.logout`

Logout current user and clear session.

**Type**: Mutation (Public)

**Response**:
```typescript
{
  success: boolean;
}
```

**Example**:
```typescript
const { mutate } = trpc.auth.logout.useMutation();
mutate();
```

---

### Session Management

#### `session.create`

Create a new detection session.

**Type**: Mutation (Protected)

**Input**:
```typescript
{
  sessionType: 'image' | 'webcam' | 'video';
  notes?: string;
}
```

**Response**:
```typescript
{
  success: boolean;
  sessionId: number;
}
```

**Example**:
```typescript
const { mutate } = trpc.session.create.useMutation();
mutate({
  sessionType: 'image',
  notes: 'Testing emotion detection'
});
```

---

#### `session.list`

Get list of user's detection sessions.

**Type**: Query (Protected)

**Input**:
```typescript
{
  limit?: number; // default: 10
}
```

**Response**:
```typescript
Array<{
  id: number;
  userId: number;
  sessionType: 'image' | 'webcam' | 'video';
  startTime: Date;
  endTime: Date | null;
  duration: number | null;
  totalFrames: number | null;
  averageStress: string | null;
  maxStress: string | null;
  minStress: string | null;
  dominantEmotion: string | null;
  notes: string | null;
  createdAt: Date;
  updatedAt: Date;
}>
```

**Example**:
```typescript
const { data: sessions } = trpc.session.list.useQuery({ limit: 20 });
```

---

#### `session.get`

Get specific session details.

**Type**: Query (Protected)

**Input**:
```typescript
{
  sessionId: number;
}
```

**Response**:
```typescript
{
  id: number;
  userId: number;
  sessionType: 'image' | 'webcam' | 'video';
  startTime: Date;
  endTime: Date | null;
  duration: number | null;
  totalFrames: number | null;
  averageStress: string | null;
  maxStress: string | null;
  minStress: string | null;
  dominantEmotion: string | null;
  notes: string | null;
  createdAt: Date;
  updatedAt: Date;
}
```

**Example**:
```typescript
const { data: session } = trpc.session.get.useQuery({ sessionId: 123 });
```

---

#### `session.update`

Update session with analysis results.

**Type**: Mutation (Protected)

**Input**:
```typescript
{
  sessionId: number;
  duration?: number;
  totalFrames?: number;
  averageStress?: number;
  maxStress?: number;
  minStress?: number;
  dominantEmotion?: string;
}
```

**Response**:
```typescript
{
  success: boolean;
}
```

**Example**:
```typescript
const { mutate } = trpc.session.update.useMutation();
mutate({
  sessionId: 123,
  duration: 120,
  totalFrames: 120,
  averageStress: 35.5,
  maxStress: 75.2,
  minStress: 10.3,
  dominantEmotion: 'happy'
});
```

---

### Detection Results

#### `result.create`

Store emotion detection result.

**Type**: Mutation (Protected)

**Input**:
```typescript
{
  sessionId: number;
  frameNumber?: number;
  dominantEmotion: string;
  emotionConfidence: number; // 0-1
  stressScore: number; // 0-100
  stressLevel: 'low' | 'moderate' | 'high' | 'critical';
  emotionProbabilities?: Record<string, number>;
  facesDetected?: number;
}
```

**Response**:
```typescript
{
  success: boolean;
}
```

**Example**:
```typescript
const { mutate } = trpc.result.create.useMutation();
mutate({
  sessionId: 123,
  frameNumber: 1,
  dominantEmotion: 'happy',
  emotionConfidence: 0.92,
  stressScore: 15,
  stressLevel: 'low',
  emotionProbabilities: {
    angry: 0.02,
    disgust: 0.01,
    fear: 0.01,
    happy: 0.92,
    sad: 0.02,
    surprise: 0.01,
    neutral: 0.01
  },
  facesDetected: 1
});
```

---

#### `result.list`

Get all results for a session.

**Type**: Query (Protected)

**Input**:
```typescript
{
  sessionId: number;
}
```

**Response**:
```typescript
Array<{
  id: number;
  sessionId: number;
  userId: number;
  frameNumber: number | null;
  dominantEmotion: string;
  emotionConfidence: string;
  stressScore: string;
  stressLevel: 'low' | 'moderate' | 'high' | 'critical';
  emotionProbabilities: Record<string, number> | null;
  facesDetected: number | null;
  timestamp: Date;
  createdAt: Date;
}>
```

**Example**:
```typescript
const { data: results } = trpc.result.list.useQuery({ sessionId: 123 });
```

---

#### `result.recent`

Get user's recent detection results.

**Type**: Query (Protected)

**Input**:
```typescript
{
  limit?: number; // default: 50
}
```

**Response**:
```typescript
Array<{
  id: number;
  sessionId: number;
  userId: number;
  frameNumber: number | null;
  dominantEmotion: string;
  emotionConfidence: string;
  stressScore: string;
  stressLevel: 'low' | 'moderate' | 'high' | 'critical';
  emotionProbabilities: Record<string, number> | null;
  facesDetected: number | null;
  timestamp: Date;
  createdAt: Date;
}>
```

**Example**:
```typescript
const { data: results } = trpc.result.recent.useQuery({ limit: 100 });
```

---

### File Management

#### `file.create`

Register uploaded file in database.

**Type**: Mutation (Protected)

**Input**:
```typescript
{
  fileName: string;
  fileType: 'image' | 'video';
  fileSize: number;
  s3Key: string;
  s3Url: string;
  sessionId?: number;
}
```

**Response**:
```typescript
{
  success: boolean;
}
```

**Example**:
```typescript
const { mutate } = trpc.file.create.useMutation();
mutate({
  fileName: 'photo.jpg',
  fileType: 'image',
  fileSize: 2048576,
  s3Key: 'user-123/photos/photo-abc123.jpg',
  s3Url: 'https://s3.amazonaws.com/bucket/user-123/photos/photo-abc123.jpg',
  sessionId: 123
});
```

---

#### `file.list`

Get user's uploaded files.

**Type**: Query (Protected)

**Input**:
```typescript
{
  limit?: number; // default: 20
}
```

**Response**:
```typescript
Array<{
  id: number;
  userId: number;
  fileName: string;
  fileType: 'image' | 'video';
  fileSize: number | null;
  s3Key: string;
  s3Url: string | null;
  sessionId: number | null;
  uploadedAt: Date;
  createdAt: Date;
}>
```

**Example**:
```typescript
const { data: files } = trpc.file.list.useQuery({ limit: 50 });
```

---

### Statistics

#### `stats.get`

Get user's aggregate statistics.

**Type**: Query (Protected)

**Response**:
```typescript
{
  id: number;
  userId: number;
  totalSessions: number;
  totalDetections: number;
  averageStress: string | null;
  mostCommonEmotion: string | null;
  lastDetectionTime: Date | null;
  updatedAt: Date;
  createdAt: Date;
}
```

**Example**:
```typescript
const { data: stats } = trpc.stats.get.useQuery();
```

---

#### `stats.update`

Update user statistics.

**Type**: Mutation (Protected)

**Input**:
```typescript
{
  totalSessions?: number;
  totalDetections?: number;
  averageStress?: number;
  mostCommonEmotion?: string;
}
```

**Response**:
```typescript
{
  success: boolean;
}
```

**Example**:
```typescript
const { mutate } = trpc.stats.update.useMutation();
mutate({
  totalSessions: 10,
  totalDetections: 500,
  averageStress: 42.5,
  mostCommonEmotion: 'happy'
});
```

---

### Models

#### `model.active`

Get information about the active model.

**Type**: Query (Public)

**Response**:
```typescript
{
  id: number;
  modelName: string;
  modelType: 'custom_cnn' | 'resnet50' | 'mobilenetv2' | 'vgg16';
  version: string;
  accuracy: string | null;
  trainingDate: Date | null;
  isActive: number;
  description: string | null;
  createdAt: Date;
  updatedAt: Date;
} | null
```

**Example**:
```typescript
const { data: model } = trpc.model.active.useQuery();
```

---

#### `model.list`

Get list of all available models.

**Type**: Query (Public)

**Response**:
```typescript
Array<{
  id: number;
  modelName: string;
  modelType: 'custom_cnn' | 'resnet50' | 'mobilenetv2' | 'vgg16';
  version: string;
  accuracy: string | null;
  trainingDate: Date | null;
  isActive: number;
  description: string | null;
  createdAt: Date;
  updatedAt: Date;
}>
```

**Example**:
```typescript
const { data: models } = trpc.model.list.useQuery();
```

---

## Error Codes

| Code | Status | Description |
|------|--------|-------------|
| `PARSE_ERROR` | 400 | Invalid input format |
| `BAD_REQUEST` | 400 | Invalid request parameters |
| `UNAUTHORIZED` | 401 | Authentication required |
| `FORBIDDEN` | 403 | Access denied |
| `NOT_FOUND` | 404 | Resource not found |
| `CONFLICT` | 409 | Resource already exists |
| `INTERNAL_SERVER_ERROR` | 500 | Server error |

---

## Rate Limiting

- **Public endpoints**: 100 requests per minute
- **Protected endpoints**: 1000 requests per minute
- **File uploads**: 50 MB per request, 1 GB per day

---

## Pagination

List endpoints support pagination via query parameters:

```typescript
{
  limit: number;  // Items per page (default: 10, max: 100)
  offset?: number; // Starting position (default: 0)
}
```

---

## Webhooks

Webhook support for real-time notifications:

```typescript
// Available events
'detection.completed'
'session.started'
'session.ended'
'stress.alert'
```

Configure webhooks in Settings → Webhooks.

---

## SDK Examples

### React Hook Usage

```typescript
import { trpc } from '@/lib/trpc';

function MyComponent() {
  // Query
  const { data: sessions, isLoading } = trpc.session.list.useQuery({ limit: 10 });

  // Mutation
  const { mutate: createSession } = trpc.session.create.useMutation({
    onSuccess: (data) => {
      console.log('Session created:', data.sessionId);
    },
    onError: (error) => {
      console.error('Error:', error.message);
    }
  });

  return (
    <div>
      {isLoading ? 'Loading...' : sessions?.map(s => <div key={s.id}>{s.id}</div>)}
      <button onClick={() => createSession({ sessionType: 'image' })}>
        Create Session
      </button>
    </div>
  );
}
```

### Error Handling

```typescript
const { mutate } = trpc.session.create.useMutation({
  onError: (error) => {
    if (error.code === 'UNAUTHORIZED') {
      // Redirect to login
    } else if (error.code === 'BAD_REQUEST') {
      // Show validation error
      console.error('Validation error:', error.message);
    }
  }
});
```

### Optimistic Updates

```typescript
const { mutate } = trpc.result.create.useMutation({
  onMutate: async (newResult) => {
    // Cancel outgoing refetches
    await trpc.result.recent.cancel();

    // Snapshot previous data
    const previousResults = trpc.result.recent.getData();

    // Optimistically update
    trpc.result.recent.setData(
      undefined,
      (old) => [newResult as any, ...old || []]
    );

    return { previousResults };
  },
  onError: (err, newResult, context) => {
    // Rollback on error
    if (context?.previousResults) {
      trpc.result.recent.setData(undefined, context.previousResults);
    }
  },
  onSuccess: () => {
    // Refetch to confirm
    trpc.result.recent.invalidate();
  }
});
```

---

## Versioning

Current API Version: **v1**

Breaking changes will increment major version. Non-breaking changes use minor versions.

---

## Support

For API questions or issues:
- Check the [README.md](./README.md) for general documentation
- Review [examples](./examples/) for code samples
- Open an issue on GitHub

---

**Last Updated**: November 2024
**API Version**: 1.0.0
