# Testing Guide

Comprehensive testing strategy for the Emotion Stress Detection System covering unit tests, integration tests, and end-to-end tests.

## Test Structure

```
emotion_stress_detector/
├── src/
│   ├── test_preprocessing.py      # Data preprocessing tests
│   ├── test_model.py              # Model architecture tests
│   ├── test_train.py              # Training pipeline tests
│   ├── test_predict.py            # Inference tests
│   └── test_stress_analyzer.py    # Stress analysis tests
│
├── client/
│   ├── src/__tests__/
│   │   ├── pages/                 # Page component tests
│   │   ├── components/            # Component tests
│   │   └── lib/                   # Utility tests
│   └── vitest.config.ts           # Vitest configuration
│
└── server/
    ├── __tests__/
    │   ├── routers.test.ts        # API endpoint tests
    │   └── db.test.ts             # Database tests
    └── vitest.config.ts
```

## Python Testing

### Setup

Install test dependencies:

```bash
pip install pytest pytest-cov pytest-mock pytest-asyncio
```

### Running Tests

Run all tests:

```bash
pytest src/ -v
```

Run specific test file:

```bash
pytest src/test_preprocessing.py -v
```

Run with coverage:

```bash
pytest src/ --cov=src --cov-report=html
```

### Test Categories

#### 1. Data Preprocessing Tests (`test_preprocessing.py`)

Tests for face detection, image preprocessing, and data augmentation:

```python
def test_face_detection():
    """Test face detection on sample images."""
    detector = FaceDetector(method='haar')
    image = cv2.imread('test_image.jpg')
    faces = detector.detect_faces(image)
    assert len(faces) > 0
    assert all(len(face) == 4 for face in faces)

def test_image_preprocessing():
    """Test image loading and preprocessing."""
    preprocessor = ImagePreprocessor()
    image = preprocessor.load_image('test_image.jpg')
    assert image is not None
    assert image.shape[0] > 0

def test_data_augmentation():
    """Test image augmentation transformations."""
    augmentor = DataAugmentor()
    image = np.random.rand(48, 48, 1)
    
    # Test rotation
    rotated = augmentor.rotate_image(image, 15)
    assert rotated.shape == image.shape
    
    # Test flip
    flipped = augmentor.flip_image(image, 'horizontal')
    assert flipped.shape == image.shape

def test_dataset_splitting():
    """Test stratified dataset splitting."""
    manager = DatasetManager()
    # Mock dataset
    X = np.random.rand(100, 48, 48, 1)
    y = np.random.randint(0, 7, 100)
    
    X_train, X_val, X_test = manager._split_dataset(X, y)
    
    assert len(X_train) == 80
    assert len(X_val) == 10
    assert len(X_test) == 10
```

#### 2. Model Architecture Tests (`test_model.py`)

Tests for CNN and transfer learning models:

```python
def test_custom_cnn_creation():
    """Test custom CNN model creation."""
    manager = ModelManager()
    model = manager.create_custom_cnn()
    
    assert model is not None
    assert len(model.layers) > 0
    assert model.output_shape[-1] == 7  # 7 emotions

def test_transfer_learning_model():
    """Test transfer learning model creation."""
    manager = ModelManager()
    model = manager.create_transfer_learning_model('resnet50')
    
    assert model is not None
    assert model.output_shape[-1] == 7

def test_model_compilation():
    """Test model compilation."""
    manager = ModelManager()
    model = manager.create_custom_cnn()
    manager.compile_model(model)
    
    assert model.optimizer is not None
    assert model.loss is not None

def test_model_save_load():
    """Test model saving and loading."""
    manager = ModelManager()
    model = manager.create_custom_cnn()
    
    # Save
    manager.save_model(model, 'test_model.h5')
    
    # Load
    loaded_model = manager.load_model('test_model.h5')
    assert loaded_model is not None
    
    # Cleanup
    import os
    os.remove('test_model.h5')
```

#### 3. Training Pipeline Tests (`test_train.py`)

Tests for training loops and callbacks:

```python
def test_data_generator():
    """Test data generator creation."""
    manager = TrainingManager(model=None)
    
    X = np.random.rand(100, 48, 48, 1)
    y = np.random.randint(0, 7, 100)
    
    train_gen = manager.create_data_generators(X, y)
    assert train_gen is not None

def test_class_weights():
    """Test class weight calculation."""
    manager = TrainingManager(model=None)
    
    y = np.array([0, 0, 0, 1, 1, 2])  # Imbalanced
    weights = manager.calculate_class_weights(y)
    
    assert weights[0] < weights[2]  # Less frequent class has higher weight

def test_callbacks_creation():
    """Test callback creation."""
    manager = TrainingManager(model=None)
    callbacks = manager.create_callbacks('test_model')
    
    assert len(callbacks) > 0
    assert any(isinstance(cb, keras.callbacks.EarlyStopping) for cb in callbacks)
```

#### 4. Inference Tests (`test_predict.py`)

Tests for prediction and inference:

```python
def test_single_image_prediction():
    """Test single image prediction."""
    predictor = EmotionPredictor('models/test_model.h5')
    
    # Create test image
    test_image = np.random.rand(48, 48, 3)
    cv2.imwrite('test_image.jpg', test_image * 255)
    
    result = predictor.predict_single_image('test_image.jpg')
    
    assert 'results' in result
    assert len(result['results']) > 0
    assert 'dominant_emotion' in result['results'][0]

def test_batch_prediction():
    """Test batch image prediction."""
    predictor = EmotionPredictor('models/test_model.h5')
    
    # Create test images
    image_paths = ['test_1.jpg', 'test_2.jpg']
    
    results = predictor.predict_batch(image_paths)
    
    assert len(results) == 2
    assert all('results' in r for r in results)

def test_visualization():
    """Test prediction visualization."""
    predictor = EmotionPredictor('models/test_model.h5')
    
    result = {
        'dominant_emotion': 'happy',
        'confidence': 0.92,
        'stress_score': 15,
        'stress_level': 'low',
        'bbox': (10, 10, 100, 100)
    }
    
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    vis_image = predictor._visualize_predictions(image, [result])
    
    assert vis_image.shape == image.shape
```

#### 5. Stress Analysis Tests (`test_stress_analyzer.py`)

Tests for stress calculation and analysis:

```python
def test_stress_calculation():
    """Test stress score calculation."""
    analyzer = StressAnalyzer()
    
    emotion_probs = np.array([0.1, 0.05, 0.05, 0.7, 0.05, 0.02, 0.03])
    stress = analyzer.analyze_frame(emotion_probs)
    
    assert 'combined_stress' in stress
    assert 'stress_level' in stress
    assert 0 <= stress['combined_stress'] <= 100

def test_stress_level_classification():
    """Test stress level classification."""
    analyzer = StressAnalyzer()
    
    # Test each level
    assert analyzer._classify_stress_level(10) == 'low'
    assert analyzer._classify_stress_level(40) == 'moderate'
    assert analyzer._classify_stress_level(60) == 'high'
    assert analyzer._classify_stress_level(80) == 'critical'

def test_temporal_analysis():
    """Test temporal trend analysis."""
    analyzer = StressAnalyzer()
    
    # Add multiple frames
    for i in range(10):
        emotion_probs = np.random.rand(7)
        emotion_probs /= emotion_probs.sum()
        analyzer.analyze_frame(emotion_probs)
    
    temporal = analyzer.get_temporal_analysis()
    
    assert 'average_stress' in temporal
    assert 'trend' in temporal
    assert temporal['trend'] in ['increasing', 'decreasing', 'stable']
```

## TypeScript/React Testing

### Setup

Install test dependencies:

```bash
pnpm add -D vitest @testing-library/react @testing-library/jest-dom
```

### Running Tests

Run all tests:

```bash
pnpm test
```

Run in watch mode:

```bash
pnpm test:watch
```

Run with coverage:

```bash
pnpm test:coverage
```

### Test Examples

#### Component Tests

```typescript
// client/src/__tests__/pages/Home.test.tsx
import { describe, it, expect, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import Home from '@/pages/Home';

describe('Home Page', () => {
  it('renders title', () => {
    render(<Home />);
    expect(screen.getByText(/Facial Emotion & Stress Detection/i)).toBeDefined();
  });

  it('displays feature cards', () => {
    render(<Home />);
    expect(screen.getByText(/7 Emotion Detection/i)).toBeDefined();
    expect(screen.getByText(/Stress Analysis/i)).toBeDefined();
  });

  it('shows login button when not authenticated', () => {
    render(<Home />);
    // Mock useAuth to return unauthenticated state
    expect(screen.getByText(/Sign In Now/i)).toBeDefined();
  });
});
```

#### Hook Tests

```typescript
// client/src/__tests__/lib/trpc.test.ts
import { describe, it, expect } from 'vitest';
import { trpc } from '@/lib/trpc';

describe('tRPC Client', () => {
  it('initializes correctly', () => {
    expect(trpc).toBeDefined();
    expect(trpc.auth).toBeDefined();
    expect(trpc.session).toBeDefined();
  });
});
```

## API Testing

### Setup

Install testing dependencies:

```bash
pnpm add -D supertest @types/supertest
```

### Example Tests

```typescript
// server/__tests__/routers.test.ts
import { describe, it, expect, beforeEach } from 'vitest';
import { appRouter } from '../routers';
import { createTRPCMsw } from 'trpc-msw';

describe('Session Router', () => {
  it('creates session successfully', async () => {
    const caller = appRouter.createCaller({
      user: { id: 1, role: 'user' } as any,
      req: {} as any,
      res: {} as any,
    });

    const result = await caller.session.create({
      sessionType: 'image',
      notes: 'Test session'
    });

    expect(result.success).toBe(true);
  });

  it('rejects unauthorized access', async () => {
    const caller = appRouter.createCaller({
      user: null,
      req: {} as any,
      res: {} as any,
    });

    expect(() => 
      caller.session.create({
        sessionType: 'image'
      })
    ).rejects.toThrow();
  });
});
```

## End-to-End Testing

### Setup

Install E2E testing framework:

```bash
pnpm add -D playwright
pnpm exec playwright install
```

### Example E2E Test

```typescript
// e2e/upload.spec.ts
import { test, expect } from '@playwright/test';

test('upload image and get results', async ({ page }) => {
  // Navigate to upload page
  await page.goto('/upload');

  // Check page loads
  await expect(page.locator('text=Select an Image')).toBeVisible();

  // Upload file
  await page.locator('input[type="file"]').setInputFiles('test_image.jpg');

  // Click analyze button
  await page.locator('button:has-text("Analyze Image")').click();

  // Wait for results
  await expect(page.locator('text=Analysis Results')).toBeVisible();

  // Verify results displayed
  await expect(page.locator('text=Dominant Emotion')).toBeVisible();
  await expect(page.locator('text=Stress Level')).toBeVisible();
});
```

## Performance Testing

### Load Testing

Test API performance under load:

```bash
# Install Apache Bench
ab -n 1000 -c 10 https://api.emotion-detector.ai/api/trpc/model.active
```

### Memory Profiling

Profile Python inference memory usage:

```python
import tracemalloc
from src.predict import create_predictor

tracemalloc.start()

predictor = create_predictor('models/model.h5')
result = predictor.predict_single_image('test_image.jpg')

current, peak = tracemalloc.get_traced_memory()
print(f"Current: {current / 1024 / 1024:.1f}MB; Peak: {peak / 1024 / 1024:.1f}MB")
```

## Continuous Integration

### GitHub Actions

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v2
      
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      
      - name: Setup Node
        uses: actions/setup-node@v2
        with:
          node-version: '22'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pnpm install
      
      - name: Run Python tests
        run: pytest src/ --cov=src
      
      - name: Run TypeScript tests
        run: pnpm test
      
      - name: Run E2E tests
        run: pnpm test:e2e
```

## Test Coverage Goals

| Component | Target Coverage |
|-----------|-----------------|
| Data Preprocessing | 90% |
| Model Architecture | 85% |
| Training Pipeline | 80% |
| Inference | 90% |
| Stress Analysis | 85% |
| API Endpoints | 90% |
| React Components | 80% |

## Debugging Tests

### Python Debugging

```python
import pdb

def test_something():
    pdb.set_trace()  # Breakpoint
    # Code to debug
```

### TypeScript Debugging

```typescript
// Add debugger statement
debugger;

// Or use console
console.log('Debug info:', variable);
```

## Best Practices

1. **Isolation**: Each test should be independent
2. **Clarity**: Test names should describe what they test
3. **Coverage**: Aim for high coverage but focus on critical paths
4. **Speed**: Keep tests fast (< 1s per test)
5. **Mocking**: Mock external dependencies
6. **Assertions**: Use clear, specific assertions
7. **Documentation**: Document complex test logic

## Troubleshooting

### Tests Failing Locally

1. Clear cache: `pnpm clean && pnpm install`
2. Update dependencies: `pnpm update`
3. Check environment variables
4. Review recent changes

### Performance Issues

1. Profile with `pytest --profile`
2. Identify slow tests
3. Optimize or parallelize
4. Consider mocking expensive operations

---

**Last Updated**: November 2024
