import { useAuth } from "@/_core/hooks/useAuth";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { APP_LOGO, APP_TITLE, getLoginUrl } from "@/const";
import { Link } from "wouter";
import { Smile, Video, BarChart3, Heart, Zap, Shield } from "lucide-react";

export default function Home() {
  const { user, loading, isAuthenticated, logout } = useAuth();

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary"></div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-50 to-slate-100 dark:from-slate-950 dark:to-slate-900">
      {/* Navigation */}
      <nav className="border-b border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-950 sticky top-0 z-50">
        <div className="container mx-auto px-4 py-4 flex justify-between items-center">
          <div className="flex items-center gap-2">
            <img src={APP_LOGO} alt={APP_TITLE} className="h-8 w-8" />
            <h1 className="text-xl font-bold text-slate-900 dark:text-white">{APP_TITLE}</h1>
          </div>
          <div className="flex items-center gap-4">
            {isAuthenticated ? (
              <>
                <span className="text-sm text-slate-600 dark:text-slate-400">
                  Welcome, {user?.name || "User"}
                </span>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => logout()}
                >
                  Logout
                </Button>
              </>
            ) : (
              <Button
                size="sm"
                onClick={() => (window.location.href = getLoginUrl())}
              >
                Login
              </Button>
            )}
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="container mx-auto px-4 py-16">
        <div className="max-w-4xl mx-auto text-center mb-16">
          <h2 className="text-4xl md:text-5xl font-bold text-slate-900 dark:text-white mb-6">
            Facial Emotion & Stress Detection
          </h2>
          <p className="text-xl text-slate-600 dark:text-slate-300 mb-8">
            Advanced AI-powered system to detect emotions and analyze stress levels in real-time.
            Upload images, use your webcam, or process videos to get instant emotional insights.
          </p>
          {isAuthenticated && (
            <div className="flex gap-4 justify-center">
              <Link href="/upload">
                <Button size="lg" className="gap-2">
                  <Smile className="w-5 h-5" />
                  Upload Image
                </Button>
              </Link>
              <Link href="/webcam">
                <Button size="lg" variant="outline" className="gap-2">
                  <Video className="w-5 h-5" />
                  Use Webcam
                </Button>
              </Link>
            </div>
          )}
        </div>

        {/* Features Grid */}
        <div className="grid md:grid-cols-3 gap-6 mb-16">
          <Card className="hover:shadow-lg transition-shadow">
            <CardHeader>
              <Smile className="w-8 h-8 text-blue-500 mb-2" />
              <CardTitle>7 Emotion Detection</CardTitle>
              <CardDescription>
                Detects angry, disgust, fear, happy, sad, surprise, and neutral emotions
              </CardDescription>
            </CardHeader>
          </Card>

          <Card className="hover:shadow-lg transition-shadow">
            <CardHeader>
              <Zap className="w-8 h-8 text-orange-500 mb-2" />
              <CardTitle>Stress Analysis</CardTitle>
              <CardDescription>
                Real-time stress level calculation (Low, Moderate, High, Critical)
              </CardDescription>
            </CardHeader>
          </Card>

          <Card className="hover:shadow-lg transition-shadow">
            <CardHeader>
              <Video className="w-8 h-8 text-purple-500 mb-2" />
              <CardTitle>Real-time Processing</CardTitle>
              <CardDescription>
                Process images, webcam feeds, and videos instantly with high accuracy
              </CardDescription>
            </CardHeader>
          </Card>

          <Card className="hover:shadow-lg transition-shadow">
            <CardHeader>
              <BarChart3 className="w-8 h-8 text-green-500 mb-2" />
              <CardTitle>History & Analytics</CardTitle>
              <CardDescription>
                Track your emotion patterns and stress trends over time
              </CardDescription>
            </CardHeader>
          </Card>

          <Card className="hover:shadow-lg transition-shadow">
            <CardHeader>
              <Heart className="w-8 h-8 text-red-500 mb-2" />
              <CardTitle>Personalized Insights</CardTitle>
              <CardDescription>
                Get recommendations based on your stress levels and emotional state
              </CardDescription>
            </CardHeader>
          </Card>

          <Card className="hover:shadow-lg transition-shadow">
            <CardHeader>
              <Shield className="w-8 h-8 text-indigo-500 mb-2" />
              <CardTitle>Privacy First</CardTitle>
              <CardDescription>
                Your data is secure and only processed for your analysis
              </CardDescription>
            </CardHeader>
          </Card>
        </div>

        {/* How It Works */}
        <div className="max-w-3xl mx-auto mb-16">
          <h3 className="text-2xl font-bold text-slate-900 dark:text-white mb-8 text-center">
            How It Works
          </h3>
          <div className="space-y-6">
            <div className="flex gap-4">
              <div className="flex-shrink-0 w-10 h-10 rounded-full bg-blue-500 text-white flex items-center justify-center font-bold">
                1
              </div>
              <div>
                <h4 className="font-semibold text-slate-900 dark:text-white mb-2">
                  Upload or Capture
                </h4>
                <p className="text-slate-600 dark:text-slate-400">
                  Upload an image, use your webcam, or select a video file to analyze
                </p>
              </div>
            </div>
            <div className="flex gap-4">
              <div className="flex-shrink-0 w-10 h-10 rounded-full bg-blue-500 text-white flex items-center justify-center font-bold">
                2
              </div>
              <div>
                <h4 className="font-semibold text-slate-900 dark:text-white mb-2">
                  Face Detection
                </h4>
                <p className="text-slate-600 dark:text-slate-400">
                  Our AI detects all faces in the image and analyzes facial features
                </p>
              </div>
            </div>
            <div className="flex gap-4">
              <div className="flex-shrink-0 w-10 h-10 rounded-full bg-blue-500 text-white flex items-center justify-center font-bold">
                3
              </div>
              <div>
                <h4 className="font-semibold text-slate-900 dark:text-white mb-2">
                  Emotion Analysis
                </h4>
                <p className="text-slate-600 dark:text-slate-400">
                  Advanced neural networks classify emotions with high confidence scores
                </p>
              </div>
            </div>
            <div className="flex gap-4">
              <div className="flex-shrink-0 w-10 h-10 rounded-full bg-blue-500 text-white flex items-center justify-center font-bold">
                4
              </div>
              <div>
                <h4 className="font-semibold text-slate-900 dark:text-white mb-2">
                  Get Results
                </h4>
                <p className="text-slate-600 dark:text-slate-400">
                  Receive detailed emotion probabilities, stress scores, and recommendations
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* CTA Section */}
        {!isAuthenticated && (
          <div className="bg-blue-50 dark:bg-blue-950 border border-blue-200 dark:border-blue-800 rounded-lg p-8 text-center">
            <h3 className="text-2xl font-bold text-slate-900 dark:text-white mb-4">
              Ready to Get Started?
            </h3>
            <p className="text-slate-600 dark:text-slate-400 mb-6">
              Sign in to access all features and start analyzing emotions today
            </p>
            <Button
              size="lg"
              onClick={() => (window.location.href = getLoginUrl())}
            >
              Sign In Now
            </Button>
          </div>
        )}
      </section>

      {/* Footer */}
      <footer className="border-t border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-950 mt-16">
        <div className="container mx-auto px-4 py-8 text-center text-slate-600 dark:text-slate-400">
          <p>© 2024 {APP_TITLE}. All rights reserved.</p>
          <p className="text-sm mt-2">
            Powered by advanced deep learning and facial recognition technology
          </p>
        </div>
      </footer>
    </div>
  );
}
