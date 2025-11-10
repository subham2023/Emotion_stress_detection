import { useState, useRef, useEffect, useCallback } from "react";
import { useAuth } from "@/_core/hooks/useAuth";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Link } from "wouter";
import { ChevronLeft, Play, Square, Download, AlertCircle } from "lucide-react";
import { useInference, PredictionResult } from "@/hooks/useInference";
import { Alert, AlertDescription } from "@/components/ui/alert";

interface EmotionData {
  emotion: string;
  confidence: number;
  stressScore: number;
  stressLevel: string;
}

export default function WebcamPage() {
  const { isAuthenticated } = useAuth();
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isActive, setIsActive] = useState(false);
  const [currentEmotion, setCurrentEmotion] = useState<EmotionData | null>(null);
  const [emotionHistory, setEmotionHistory] = useState<EmotionData[]>([]);
  const [sessionStartTime, setSessionStartTime] = useState<Date | null>(null);
  
  // Live tracking with inference hook
  const {
    isConnected,
    isProcessing,
    error: inferenceError,
    sessionId,
    frameCount,
    currentResult,
    connect,
    disconnect,
    startSession,
    predictFrame,
    endSession,
  } = useInference({
    serverUrl: `http://${window.location.hostname}:5000`,
    autoConnect: false,
  });

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

  const startWebcam = async () => {
    try {
      // Connect to inference server
      connect();
      
      // Wait a bit for connection
      await new Promise(resolve => setTimeout(resolve, 500));
      
      // Start session in database and on inference server
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false,
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setIsActive(true);
        setSessionStartTime(new Date());
        setEmotionHistory([]);
        setCurrentEmotion(null);
        
        // Start detection session
        try {
          await startSession('webcam');
        } catch (err) {
          console.error('Failed to start session:', err);
        }
      }
    } catch (err) {
      alert("Failed to access webcam. Please check permissions.");
      disconnect();
    }
  };

  const stopWebcam = async () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const tracks = (videoRef.current.srcObject as MediaStream).getTracks();
      tracks.forEach((track) => track.stop());
      setIsActive(false);
      
      // End session
      try {
        await endSession();
      } catch (err) {
        console.error('Failed to end session:', err);
      }
      
      disconnect();
    }
  };

  const captureFrame = useCallback(async () => {
    if (videoRef.current && canvasRef.current && isConnected && !isProcessing) {
      const ctx = canvasRef.current.getContext("2d");
      if (ctx) {
        ctx.drawImage(videoRef.current, 0, 0);
        
        // Convert canvas to base64
        const frameBase64 = canvasRef.current.toDataURL('image/jpeg', 0.8).split(',')[1];
        
        try {
          // Send frame to inference server via WebSocket
          const result = await predictFrame(frameBase64, Date.now());
          
          if (result && !('error' in result)) {
            const emotionData: EmotionData = {
              emotion: result.dominant_emotion,
              confidence: result.confidence,
              stressScore: result.stress_score,
              stressLevel: result.stress_level,
            };
            
            setCurrentEmotion(emotionData);
            setEmotionHistory((prev) => [...prev, emotionData]);
          }
        } catch (err) {
          console.error('Frame prediction error:', err);
        }
      }
    }
  }, [isConnected, isProcessing, predictFrame]);

  // Update current emotion from inference result
  useEffect(() => {
    if (currentResult && !('error' in currentResult)) {
      const emotionData: EmotionData = {
        emotion: currentResult.dominant_emotion,
        confidence: currentResult.confidence,
        stressScore: currentResult.stress_score,
        stressLevel: currentResult.stress_level,
      };
      setCurrentEmotion(emotionData);
      setEmotionHistory((prev) => [...prev, emotionData]);
    }
  }, [currentResult]);

  // Capture frames periodically when active
  useEffect(() => {
    if (isActive && isConnected) {
      const interval = setInterval(captureFrame, 1000);
      return () => clearInterval(interval);
    }
  }, [isActive, isConnected, captureFrame]);
  
  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (isActive) {
        stopWebcam();
      }
      disconnect();
    };
  }, []);

  const downloadReport = () => {
    if (emotionHistory.length === 0) return;

    const avgStress =
      emotionHistory.reduce((sum, e) => sum + e.stressScore, 0) / emotionHistory.length;
    const duration = sessionStartTime
      ? Math.floor((new Date().getTime() - sessionStartTime.getTime()) / 1000)
      : 0;

    const report = `
Webcam Session Report
=====================
Duration: ${duration} seconds
Total Frames: ${emotionHistory.length}
Average Stress: ${avgStress.toFixed(2)}/100

Emotion Distribution:
${Object.entries(
  emotionHistory.reduce(
    (acc, e) => {
      acc[e.emotion] = (acc[e.emotion] || 0) + 1;
      return acc;
    },
    {} as Record<string, number>
  )
)
  .map(([emotion, count]) => `${emotion}: ${count} (${((count / emotionHistory.length) * 100).toFixed(1)}%)`)
  .join("\n")}

Generated: ${new Date().toLocaleString()}
    `;

    const element = document.createElement("a");
    element.setAttribute("href", "data:text/plain;charset=utf-8," + encodeURIComponent(report));
    element.setAttribute("download", `emotion-report-${Date.now()}.txt`);
    element.style.display = "none";
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
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

  const getStressColor = (level: string) => {
    switch (level) {
      case "low":
        return "bg-green-100 dark:bg-green-950 text-green-900 dark:text-green-100 border-green-300 dark:border-green-700";
      case "moderate":
        return "bg-yellow-100 dark:bg-yellow-950 text-yellow-900 dark:text-yellow-100 border-yellow-300 dark:border-yellow-700";
      case "high":
        return "bg-orange-100 dark:bg-orange-950 text-orange-900 dark:text-orange-100 border-orange-300 dark:border-orange-700";
      case "critical":
        return "bg-red-100 dark:bg-red-950 text-red-900 dark:text-red-100 border-red-300 dark:border-red-700";
      default:
        return "bg-slate-100 dark:bg-slate-800 text-slate-900 dark:text-slate-100";
    }
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
          <h1 className="text-2xl font-bold text-slate-900 dark:text-white">Webcam Detection</h1>
        </div>
      </div>

      <div className="container mx-auto px-4 py-8">
        <div className="max-w-4xl mx-auto">
          <div className="grid lg:grid-cols-3 gap-6">
            {/* Video Feed */}
            <div className="lg:col-span-2">
              <Card>
                <CardHeader>
                  <CardTitle>Live Feed</CardTitle>
                  <CardDescription>Real-time emotion detection from your webcam</CardDescription>
                </CardHeader>
                <CardContent>
                  {inferenceError && (
                    <Alert variant="destructive" className="mb-4">
                      <AlertCircle className="h-4 w-4" />
                      <AlertDescription>
                        {inferenceError}
                      </AlertDescription>
                    </Alert>
                  )}
                  
                  <div className="relative bg-black rounded-lg overflow-hidden mb-4">
                    <video
                      ref={videoRef}
                      autoPlay
                      playsInline
                      className="w-full h-auto"
                      style={{ maxHeight: "480px" }}
                    />
                    <canvas
                      ref={canvasRef}
                      className="hidden"
                      width={640}
                      height={480}
                    />
                    {isProcessing && (
                      <div className="absolute top-2 right-2 bg-blue-500 text-white px-2 py-1 rounded text-sm">
                        Processing...
                      </div>
                    )}
                    {isConnected && (
                      <div className="absolute top-2 left-2 bg-green-500 text-white px-2 py-1 rounded text-sm">
                        Connected
                      </div>
                    )}
                  </div>

                  <div className="flex gap-4">
                    {!isActive ? (
                      <Button onClick={startWebcam} className="flex-1 gap-2">
                        <Play className="w-4 h-4" />
                        Start Webcam
                      </Button>
                    ) : (
                      <Button
                        onClick={stopWebcam}
                        variant="destructive"
                        className="flex-1 gap-2"
                      >
                        <Square className="w-4 h-4" />
                        Stop Webcam
                      </Button>
                    )}
                    {emotionHistory.length > 0 && (
                      <Button
                        onClick={downloadReport}
                        variant="outline"
                        className="gap-2"
                      >
                        <Download className="w-4 h-4" />
                        Report
                      </Button>
                    )}
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Current Emotion & Stats */}
            <div className="space-y-6">
              {/* Current Emotion */}
              {currentEmotion && (
                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg">Current Emotion</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="text-center">
                      <div className="text-6xl mb-4">
                        {getEmotionEmoji(currentEmotion.emotion)}
                      </div>
                      <p className="text-2xl font-bold text-slate-900 dark:text-white capitalize mb-2">
                        {currentEmotion.emotion}
                      </p>
                      <p className="text-sm text-slate-600 dark:text-slate-400 mb-4">
                        Confidence: {(currentEmotion.confidence * 100).toFixed(1)}%
                      </p>
                      <div className="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-2">
                        <div
                          className="bg-blue-500 h-2 rounded-full transition-all"
                          style={{ width: `${currentEmotion.confidence * 100}%` }}
                        ></div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              )}

              {/* Stress Level */}
              {currentEmotion && (
                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg">Stress Level</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div
                      className={`p-4 rounded-lg border text-center ${getStressColor(currentEmotion.stressLevel)}`}
                    >
                      <p className="text-3xl font-bold capitalize mb-2">
                        {currentEmotion.stressLevel}
                      </p>
                      <p className="text-sm">Score: {currentEmotion.stressScore.toFixed(1)}/100</p>
                    </div>
                  </CardContent>
                </Card>
              )}

              {/* Session Stats */}
              {emotionHistory.length > 0 && (
                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg">Session Stats</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <div>
                      <p className="text-sm text-slate-600 dark:text-slate-400">Frames Analyzed</p>
                      <p className="text-2xl font-bold text-slate-900 dark:text-white">
                        {frameCount > 0 ? frameCount : emotionHistory.length}
                      </p>
                    </div>
                    <div>
                      <p className="text-sm text-slate-600 dark:text-slate-400">Avg Stress</p>
                      <p className="text-2xl font-bold text-slate-900 dark:text-white">
                        {(
                          emotionHistory.reduce((sum, e) => sum + e.stressScore, 0) /
                          emotionHistory.length
                        ).toFixed(1)}
                      </p>
                    </div>
                    <div>
                      <p className="text-sm text-slate-600 dark:text-slate-400">Duration</p>
                      <p className="text-2xl font-bold text-slate-900 dark:text-white">
                        {sessionStartTime
                          ? Math.floor(
                              (new Date().getTime() - sessionStartTime.getTime()) / 1000
                            )
                          : 0}
                        s
                      </p>
                    </div>
                  </CardContent>
                </Card>
              )}

              {/* Emotion Distribution */}
              {emotionHistory.length > 0 && (
                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg">Emotions Detected</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-2">
                      {Object.entries(
                        emotionHistory.reduce(
                          (acc, e) => {
                            acc[e.emotion] = (acc[e.emotion] || 0) + 1;
                            return acc;
                          },
                          {} as Record<string, number>
                        )
                      )
                        .sort(([, a], [, b]) => b - a)
                        .map(([emotion, count]) => (
                          <div key={emotion}>
                            <div className="flex justify-between items-center text-sm mb-1">
                              <span className="capitalize">{emotion}</span>
                              <span className="text-slate-600 dark:text-slate-400">
                                {count} ({((count / emotionHistory.length) * 100).toFixed(0)}%)
                              </span>
                            </div>
                            <div className="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-1.5">
                              <div
                                className="bg-blue-500 h-1.5 rounded-full"
                                style={{
                                  width: `${(count / emotionHistory.length) * 100}%`,
                                }}
                              ></div>
                            </div>
                          </div>
                        ))}
                    </div>
                  </CardContent>
                </Card>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
