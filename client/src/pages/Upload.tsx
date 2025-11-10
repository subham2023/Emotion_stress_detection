import { useState } from "react";
import { useAuth } from "@/_core/hooks/useAuth";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Link } from "wouter";
import { Upload, ChevronLeft, AlertCircle, CheckCircle } from "lucide-react";
import { trpc } from "@/lib/trpc";

interface DetectionResult {
  emotion: string;
  confidence: number;
  stressScore: number;
  stressLevel: string;
  emotionProbabilities: Record<string, number>;
}

export default function UploadPage() {
  const { user, isAuthenticated } = useAuth();
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [results, setResults] = useState<DetectionResult[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  if (!isAuthenticated) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <Card className="w-full max-w-md">
          <CardHeader>
            <CardTitle>Access Denied</CardTitle>
            <CardDescription>Please log in to use this feature</CardDescription>
          </CardHeader>
          <CardContent>
            <Link href="/">
              <Button className="w-full">Go Back Home</Button>
            </Link>
          </CardContent>
        </Card>
      </div>
    );
  }

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // Validate file type
    if (!["image/jpeg", "image/png", "image/gif", "image/bmp"].includes(file.type)) {
      setError("Please select a valid image file (JPG, PNG, GIF, BMP)");
      return;
    }

    // Validate file size (10MB max)
    if (file.size > 10 * 1024 * 1024) {
      setError("File size must be less than 10MB");
      return;
    }

    setSelectedFile(file);
    setError(null);

    // Create preview
    const reader = new FileReader();
    reader.onload = (e) => {
      setPreview(e.target?.result as string);
    };
    reader.readAsDataURL(file);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();

    const file = e.dataTransfer.files?.[0];
    if (file) {
      const input = document.createElement("input");
      input.type = "file";
      Object.defineProperty(input, "files", {
        value: e.dataTransfer.files,
      });
      handleFileSelect({ target: input } as any);
    }
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return;

    setIsProcessing(true);
    setError(null);

    try {
      // Simulate API call - in production, this would call your backend
      // For now, we'll create a mock result
      await new Promise((resolve) => setTimeout(resolve, 2000));

      // Mock results
      const mockResults: DetectionResult[] = [
        {
          emotion: "happy",
          confidence: 0.92,
          stressScore: 15,
          stressLevel: "low",
          emotionProbabilities: {
            angry: 0.02,
            disgust: 0.01,
            fear: 0.01,
            happy: 0.92,
            sad: 0.02,
            surprise: 0.01,
            neutral: 0.01,
          },
        },
      ];

      setResults(mockResults);
    } catch (err) {
      setError("Failed to analyze image. Please try again.");
    } finally {
      setIsProcessing(false);
    }
  };

  const getStressColor = (level: string) => {
    switch (level) {
      case "low":
        return "text-green-600 dark:text-green-400";
      case "moderate":
        return "text-yellow-600 dark:text-yellow-400";
      case "high":
        return "text-orange-600 dark:text-orange-400";
      case "critical":
        return "text-red-600 dark:text-red-400";
      default:
        return "text-slate-600 dark:text-slate-400";
    }
  };

  const getEmotionEmoji = (emotion: string) => {
    const emojis: Record<string, string> = {
      angry: "😠",
      disgust: "🤢",
      fear: "😨",
      happy: "😊",
      sad: "😢",
      surprise: "😲",
      neutral: "😐",
    };
    return emojis[emotion] || "😐";
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-50 to-slate-100 dark:from-slate-950 dark:to-slate-900">
      {/* Header */}
      <div className="border-b border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-950">
        <div className="container mx-auto px-4 py-4 flex items-center gap-4">
          <Link href="/">
            <Button variant="ghost" size="sm" className="gap-2">
              <ChevronLeft className="w-4 h-4" />
              Back
            </Button>
          </Link>
          <h1 className="text-2xl font-bold text-slate-900 dark:text-white">Upload Image</h1>
        </div>
      </div>

      <div className="container mx-auto px-4 py-8">
        <div className="max-w-2xl mx-auto">
          {/* Upload Area */}
          {!results && (
            <Card className="mb-8">
              <CardHeader>
                <CardTitle>Select an Image</CardTitle>
                <CardDescription>
                  Upload an image to analyze emotions and stress levels
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div
                  onDragOver={handleDragOver}
                  onDrop={handleDrop}
                  className="border-2 border-dashed border-slate-300 dark:border-slate-700 rounded-lg p-8 text-center cursor-pointer hover:border-blue-500 dark:hover:border-blue-400 transition-colors"
                >
                  <input
                    type="file"
                    id="file-input"
                    accept="image/*"
                    onChange={handleFileSelect}
                    className="hidden"
                  />
                  <label htmlFor="file-input" className="cursor-pointer">
                    <Upload className="w-12 h-12 mx-auto mb-4 text-slate-400" />
                    <p className="text-lg font-semibold text-slate-900 dark:text-white mb-2">
                      Drag and drop your image here
                    </p>
                    <p className="text-sm text-slate-600 dark:text-slate-400 mb-4">
                      or click to select a file
                    </p>
                    <p className="text-xs text-slate-500 dark:text-slate-500">
                      Supported formats: JPG, PNG, GIF, BMP (Max 10MB)
                    </p>
                  </label>
                </div>

                {error && (
                  <div className="mt-4 p-4 bg-red-50 dark:bg-red-950 border border-red-200 dark:border-red-800 rounded-lg flex gap-3">
                    <AlertCircle className="w-5 h-5 text-red-600 dark:text-red-400 flex-shrink-0 mt-0.5" />
                    <p className="text-sm text-red-700 dark:text-red-300">{error}</p>
                  </div>
                )}

                {selectedFile && preview && (
                  <div className="mt-8">
                    <h3 className="text-sm font-semibold text-slate-900 dark:text-white mb-4">
                      Preview
                    </h3>
                    <img
                      src={preview}
                      alt="Preview"
                      className="max-w-full h-auto rounded-lg border border-slate-200 dark:border-slate-700 mb-4"
                    />
                    <div className="flex gap-4">
                      <Button
                        onClick={handleAnalyze}
                        disabled={isProcessing}
                        className="flex-1"
                      >
                        {isProcessing ? "Analyzing..." : "Analyze Image"}
                      </Button>
                      <Button
                        variant="outline"
                        onClick={() => {
                          setSelectedFile(null);
                          setPreview(null);
                          setError(null);
                        }}
                      >
                        Clear
                      </Button>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          )}

          {/* Results */}
          {results && (
            <>
              <Card className="mb-8">
                <CardHeader>
                  <CardTitle>Analysis Results</CardTitle>
                  <CardDescription>
                    Emotion detection and stress analysis for your image
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <img
                    src={preview!}
                    alt="Analyzed"
                    className="max-w-full h-auto rounded-lg border border-slate-200 dark:border-slate-700 mb-6"
                  />

                  {results.map((result, idx) => (
                    <div key={idx} className="space-y-6">
                      {/* Main Result */}
                      <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-950 dark:to-indigo-950 rounded-lg p-6 border border-blue-200 dark:border-blue-800">
                        <div className="flex items-center gap-4 mb-4">
                          <div className="text-5xl">
                            {getEmotionEmoji(result.emotion)}
                          </div>
                          <div>
                            <p className="text-sm text-slate-600 dark:text-slate-400">
                              Dominant Emotion
                            </p>
                            <p className="text-3xl font-bold text-slate-900 dark:text-white capitalize">
                              {result.emotion}
                            </p>
                            <p className="text-sm text-slate-600 dark:text-slate-400">
                              Confidence: {(result.confidence * 100).toFixed(1)}%
                            </p>
                          </div>
                        </div>

                        <div className="bg-white dark:bg-slate-900 rounded p-4">
                          <div className="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-2">
                            <div
                              className="bg-blue-500 h-2 rounded-full"
                              style={{ width: `${result.confidence * 100}%` }}
                            ></div>
                          </div>
                        </div>
                      </div>

                      {/* Stress Analysis */}
                      <div className="grid md:grid-cols-2 gap-4">
                        <Card>
                          <CardHeader>
                            <CardTitle className="text-lg">Stress Level</CardTitle>
                          </CardHeader>
                          <CardContent>
                            <div className="flex items-center gap-4">
                              <div>
                                <p className={`text-3xl font-bold capitalize ${getStressColor(result.stressLevel)}`}>
                                  {result.stressLevel}
                                </p>
                                <p className="text-sm text-slate-600 dark:text-slate-400 mt-2">
                                  Score: {result.stressScore}/100
                                </p>
                              </div>
                              <div className="flex-1">
                                <div className="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-3">
                                  <div
                                    className={`h-3 rounded-full ${
                                      result.stressScore < 25
                                        ? "bg-green-500"
                                        : result.stressScore < 50
                                          ? "bg-yellow-500"
                                          : result.stressScore < 75
                                            ? "bg-orange-500"
                                            : "bg-red-500"
                                    }`}
                                    style={{ width: `${result.stressScore}%` }}
                                  ></div>
                                </div>
                              </div>
                            </div>
                          </CardContent>
                        </Card>

                        <Card>
                          <CardHeader>
                            <CardTitle className="text-lg">Recommendations</CardTitle>
                          </CardHeader>
                          <CardContent>
                            <ul className="space-y-2 text-sm">
                              {result.stressLevel === "low" && (
                                <>
                                  <li className="flex gap-2">
                                    <CheckCircle className="w-4 h-4 text-green-600 dark:text-green-400 flex-shrink-0 mt-0.5" />
                                    <span>Keep up the good work!</span>
                                  </li>
                                  <li className="flex gap-2">
                                    <CheckCircle className="w-4 h-4 text-green-600 dark:text-green-400 flex-shrink-0 mt-0.5" />
                                    <span>Your stress levels are healthy</span>
                                  </li>
                                </>
                              )}
                              {result.stressLevel === "moderate" && (
                                <>
                                  <li className="flex gap-2">
                                    <AlertCircle className="w-4 h-4 text-yellow-600 dark:text-yellow-400 flex-shrink-0 mt-0.5" />
                                    <span>Consider taking a short break</span>
                                  </li>
                                  <li className="flex gap-2">
                                    <AlertCircle className="w-4 h-4 text-yellow-600 dark:text-yellow-400 flex-shrink-0 mt-0.5" />
                                    <span>Try some deep breathing exercises</span>
                                  </li>
                                </>
                              )}
                              {result.stressLevel === "high" && (
                                <>
                                  <li className="flex gap-2">
                                    <AlertCircle className="w-4 h-4 text-orange-600 dark:text-orange-400 flex-shrink-0 mt-0.5" />
                                    <span>Your stress is elevated</span>
                                  </li>
                                  <li className="flex gap-2">
                                    <AlertCircle className="w-4 h-4 text-orange-600 dark:text-orange-400 flex-shrink-0 mt-0.5" />
                                    <span>Take a break and relax</span>
                                  </li>
                                </>
                              )}
                              {result.stressLevel === "critical" && (
                                <>
                                  <li className="flex gap-2">
                                    <AlertCircle className="w-4 h-4 text-red-600 dark:text-red-400 flex-shrink-0 mt-0.5" />
                                    <span>Stress levels are critical</span>
                                  </li>
                                  <li className="flex gap-2">
                                    <AlertCircle className="w-4 h-4 text-red-600 dark:text-red-400 flex-shrink-0 mt-0.5" />
                                    <span>Seek immediate support</span>
                                  </li>
                                </>
                              )}
                            </ul>
                          </CardContent>
                        </Card>
                      </div>

                      {/* Emotion Probabilities */}
                      <Card>
                        <CardHeader>
                          <CardTitle className="text-lg">All Emotions</CardTitle>
                        </CardHeader>
                        <CardContent>
                          <div className="space-y-3">
                            {Object.entries(result.emotionProbabilities)
                              .sort(([, a], [, b]) => b - a)
                              .map(([emotion, prob]) => (
                                <div key={emotion}>
                                  <div className="flex justify-between items-center mb-1">
                                    <span className="text-sm font-medium text-slate-900 dark:text-white capitalize">
                                      {emotion}
                                    </span>
                                    <span className="text-sm text-slate-600 dark:text-slate-400">
                                      {(prob * 100).toFixed(1)}%
                                    </span>
                                  </div>
                                  <div className="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-2">
                                    <div
                                      className="bg-blue-500 h-2 rounded-full"
                                      style={{ width: `${prob * 100}%` }}
                                    ></div>
                                  </div>
                                </div>
                              ))}
                          </div>
                        </CardContent>
                      </Card>
                    </div>
                  ))}

                  <Button
                    onClick={() => {
                      setResults(null);
                      setSelectedFile(null);
                      setPreview(null);
                    }}
                    className="w-full mt-6"
                  >
                    Analyze Another Image
                  </Button>
                </CardContent>
              </Card>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
