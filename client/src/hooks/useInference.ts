import { useState, useCallback, useRef, useEffect } from 'react';
import { io, Socket } from 'socket.io-client';
import { trpc } from '@/lib/trpc';

export interface PredictionResult {
  dominant_emotion: string;
  confidence: number;
  stress_score: number;
  stress_level: 'low' | 'moderate' | 'high' | 'critical';
  emotion_probabilities: Record<string, number>;
  faces_detected: number;
}

export interface SessionStats {
  frame_count: number;
  emotion_history: string[];
  stress_history: number[];
}

export interface UseInferenceOptions {
  serverUrl?: string;
  autoConnect?: boolean;
}

export function useInference(options: UseInferenceOptions = {}) {
  const {
    serverUrl = `${window.location.hostname}:5000`,
    autoConnect = false,
  } = options;

  const [isConnected, setIsConnected] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sessionId, setSessionId] = useState<number | null>(null);
  const [frameCount, setFrameCount] = useState(0);
  const [currentResult, setCurrentResult] = useState<PredictionResult | null>(null);
  const [sessionStats, setSessionStats] = useState<SessionStats | null>(null);

  const socketRef = useRef<Socket | null>(null);
  const createSessionMutation = trpc.prediction.createSession.useMutation();
  const storeResultMutation = trpc.prediction.storeResult.useMutation();
  const finalizeSessionMutation = trpc.prediction.finalizeSession.useMutation();

  // Initialize socket connection
  useEffect(() => {
    if (!autoConnect) return;

    const socket = io(serverUrl, {
      reconnection: true,
      reconnectionDelay: 1000,
      reconnectionDelayMax: 5000,
      reconnectionAttempts: 5,
    });

    socket.on('connect', () => {
      setIsConnected(true);
      setError(null);
    });

    socket.on('disconnect', () => {
      setIsConnected(false);
    });

    socket.on('error', (err: any) => {
      setError(err.message || 'Connection error');
    });

    socket.on('prediction_result', (data: any) => {
      setCurrentResult(data.result);
      setFrameCount(data.frame_number);
    });

    socket.on('session_started', (data: any) => {
      setSessionId(data.session_id);
      setFrameCount(0);
    });

    socket.on('session_ended', (data: any) => {
      setSessionStats(data);
    });

    socket.on('session_stats', (stats: any) => {
      setSessionStats(stats);
    });

    socketRef.current = socket;

    return () => {
      socket.disconnect();
    };
  }, [autoConnect, serverUrl]);

  // Connect to inference server
  const connect = useCallback(() => {
    if (socketRef.current?.connected) return;

    const socket = io(serverUrl, {
      reconnection: true,
      reconnectionDelay: 1000,
      reconnectionDelayMax: 5000,
      reconnectionAttempts: 5,
    });

    socket.on('connect', () => {
      setIsConnected(true);
      setError(null);
    });

    socket.on('disconnect', () => {
      setIsConnected(false);
    });

    socket.on('error', (err: any) => {
      setError(err.message || 'Connection error');
    });

    socket.on('prediction_result', (data: any) => {
      setCurrentResult(data.result);
      setFrameCount(data.frame_number);
    });

    socket.on('session_started', (data: any) => {
      setSessionId(data.session_id);
      setFrameCount(0);
    });

    socket.on('session_ended', (data: any) => {
      setSessionStats(data);
    });

    socket.on('session_stats', (stats: any) => {
      setSessionStats(stats);
    });

    socketRef.current = socket;
  }, [serverUrl]);

  // Disconnect from inference server
  const disconnect = useCallback(() => {
    if (socketRef.current?.connected) {
      socketRef.current.disconnect();
      setIsConnected(false);
    }
  }, []);

  // Start a new session
  const startSession = useCallback(
    async (type: 'webcam' | 'video' | 'image') => {
      try {
        setError(null);

        // Create session in database
        const result = await createSessionMutation.mutateAsync({
          sessionType: type,
        });

        if (!result.success) {
          throw new Error('Failed to create session');
        }

        // Start session on inference server
        if (socketRef.current?.connected) {
          socketRef.current.emit('start_session', {
            session_id: result.sessionId,
            session_type: type,
          });
        }

        setSessionId(result.sessionId);
        setFrameCount(0);
        setCurrentResult(null);

        return result.sessionId;
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Failed to start session';
        setError(message);
        throw err;
      }
    },
    [createSessionMutation]
  );

  // Predict from frame
  const predictFrame = useCallback(
    async (frameBase64: string, timestamp?: number) => {
      if (!socketRef.current?.connected) {
        setError('Not connected to inference server');
        return null;
      }

      try {
        setIsProcessing(true);
        setError(null);

        return new Promise<PredictionResult | null>((resolve, reject) => {
          const timeout = setTimeout(() => {
            reject(new Error('Prediction timeout'));
          }, 10000);

          const handleResult = (data: any) => {
            clearTimeout(timeout);
            socketRef.current?.off('prediction_result', handleResult);
            setIsProcessing(false);
            resolve(data.result);
          };

          socketRef.current?.on('prediction_result', handleResult);
          socketRef.current?.emit('predict_frame', {
            image: frameBase64,
            timestamp,
          });
        });
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Prediction failed';
        setError(message);
        setIsProcessing(false);
        return null;
      }
    },
    []
  );

  // Predict image via HTTP
  const predictImage = useCallback(
    async (imageBase64: string) => {
      try {
        setIsProcessing(true);
        setError(null);

        const result = await trpc.prediction.predictImage.mutateAsync({
          imageBase64,
          sessionId: sessionId || undefined,
        });

        if (result.success && result.result) {
          setCurrentResult(result.result);
          setFrameCount((prev) => prev + 1);

          // Store result in database
          if (sessionId && result.result && !('error' in result.result)) {
            await storeResultMutation.mutateAsync({
              sessionId,
              frameNumber: frameCount + 1,
              dominantEmotion: result.result.dominant_emotion,
              emotionConfidence: result.result.confidence,
              stressScore: result.result.stress_score,
              stressLevel: result.result.stress_level,
              emotionProbabilities: result.result.emotion_probabilities,
              facesDetected: result.result.faces_detected,
            });
          }

          return result.result;
        } else {
          const errorMsg = result.result && 'error' in result.result ? result.result.error : 'Prediction failed';
          throw new Error(errorMsg);
        }
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Prediction failed';
        setError(message);
        return null;
      } finally {
        setIsProcessing(false);
      }
    },
    [sessionId, frameCount, storeResultMutation]
  );

  // End session
  const endSession = useCallback(async () => {
    try {
      if (!sessionId) return;

      setError(null);

      // End session on inference server
      if (socketRef.current?.connected) {
        socketRef.current.emit('end_session', {
          session_id: sessionId,
        });
      }

      // Finalize session in database
      let averageStress: number | undefined;
      let dominantEmotion: string | undefined;

      if (sessionStats && 'stress_history' in sessionStats && sessionStats.stress_history.length > 0) {
        averageStress =
          sessionStats.stress_history.reduce((a: number, b: number) => a + b, 0) /
          sessionStats.stress_history.length;
      } else {
        averageStress = currentResult?.stress_score;
      }

      if (sessionStats && 'emotion_history' in sessionStats && sessionStats.emotion_history.length > 0) {
        const emotionCounts: Record<string, number> = {};
        sessionStats.emotion_history.forEach((emotion: string) => {
          emotionCounts[emotion] = (emotionCounts[emotion] || 0) + 1;
        });
        dominantEmotion = Object.keys(emotionCounts).reduce((a, b) =>
          emotionCounts[a] > emotionCounts[b] ? a : b
        );
      } else {
        dominantEmotion = currentResult?.dominant_emotion;
      }

      await finalizeSessionMutation.mutateAsync({
        sessionId,
        totalFrames: frameCount,
        averageStress,
        dominantEmotion,
      });

      setSessionId(null);
      setFrameCount(0);
      setCurrentResult(null);
      setSessionStats(null);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to end session';
      setError(message);
      throw err;
    }
  }, [sessionId, frameCount, currentResult, sessionStats, finalizeSessionMutation]);

  // Get session stats
  const getStats = useCallback(() => {
    if (!socketRef.current?.connected) {
      setError('Not connected to inference server');
      return;
    }

    socketRef.current.emit('get_session_stats');
  }, []);

  return {
    // State
    isConnected,
    isProcessing,
    error,
    sessionId,
    frameCount,
    currentResult,
    sessionStats,

    // Methods
    connect,
    disconnect,
    startSession,
    predictFrame,
    predictImage,
    endSession,
    getStats,
  };
}
