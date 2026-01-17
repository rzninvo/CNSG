import { useState, useCallback } from "react";
import { NavigatorHeader } from "@/components/NavigatorHeader";
import { CameraView } from "@/components/CameraView";
import { DestinationInput } from "@/components/DestinationInput";
import { LoadingState } from "@/components/LoadingState";
import { NavigationResult } from "@/components/NavigationResult";
import { ErrorMessage } from "@/components/ErrorMessage";
import { FeedbackSection } from "@/components/FeedbackSection";
import type { FeedbackRatings } from "@/components/FeedbackSection";
import { toast } from "sonner";
import { config } from "@/lib/config";
import { submitFeedbackToGoogleSheets } from "@/services/feedbackService"; // ← AGGIUNGI QUESTA RIGA

// ⚠️ Replace with your actual server URL
const SERVER_URL = config.serverUrl;

const Index = () => {
  const [capturedImage, setCapturedImage] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [feedbackSubmitted, setFeedbackSubmitted] = useState(false);
  const [isSubmittingFeedback, setIsSubmittingFeedback] = useState(false);

  const handleFeedbackSubmit = async (
    ratings: FeedbackRatings,
    comment: string,
  ) => {
    console.log("=== handleFeedbackSubmit called ===");
    console.log("Ratings:", ratings);
    console.log("Comment:", comment);

    setIsSubmittingFeedback(true);

    try {
      console.log("Calling submitFeedbackToGoogleSheets...");
      const result = await submitFeedbackToGoogleSheets(ratings, comment);
      console.log("Result from submitFeedbackToGoogleSheets:", result);

      if (result.success) {
        console.log("✓ Feedback submitted successfully");
        setFeedbackSubmitted(true);
        toast.success("Thanks for your feedback!");
      } else {
        console.error("❌ Feedback submission failed:", result.error);
        throw new Error(result.error || "Failed to submit feedback");
      }
    } catch (error) {
      console.error("❌ Error in handleFeedbackSubmit:", error);
      toast.error("Failed to submit feedback. Please try again.");
    } finally {
      setIsSubmittingFeedback(false);
      console.log("=== handleFeedbackSubmit completed ===");
    }
  };

  const handleCapture = useCallback((imageData: string) => {
    setCapturedImage(imageData);
    setError(null);
    setResult(null);
  }, []);

  const handleRetake = useCallback(() => {
    setCapturedImage(null);
    setError(null);
    setResult(null);
  }, []);

  const handleSubmit = async (destination: string) => {
    if (!capturedImage) {
      toast.error("Please take a photo first");
      return;
    }

    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      // Convert base64 to blob
      const response = await fetch(capturedImage);
      const blob = await response.blob();

      // Create FormData
      const formData = new FormData();
      formData.append("image", blob, "photo.jpg");
      formData.append("user_input", destination);

      console.log("Sending request to server...");
      console.log("FormData contents:", {
        user_input: destination,
        image: blob,
      });

      // Send to server with timeout
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 90000); // 90 seconds

      const serverResponse = await fetch(SERVER_URL, {
        method: "POST",
        body: formData,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!serverResponse.ok) {
        console.log("Server response DIOPORCO:", serverResponse);
        throw new Error(`Server error (${serverResponse.status})`);
      }

      const data = await serverResponse.json();

      console.log("Server response:", data);
      if (data.status === "success" && data.result) {
        setResult(data.result);
        toast.success("Route found!");
      } else {
        throw new Error(data.error || "Invalid response from server");
      }
    } catch (err) {
      console.error("Error:", err);

      let errorMessage = "An unexpected error occurred";

      if (err instanceof Error) {
        if (err.name === "AbortError") {
          errorMessage = "Request timed out. Please try again.";
        } else if (!navigator.onLine) {
          errorMessage = "No internet connection. Please check your network.";
        } else {
          errorMessage = err.message;
        }
      }

      setError(errorMessage);
      toast.error(errorMessage);
    } finally {
      setIsLoading(false);
    }
  };

  const handleNewSearch = () => {
    setCapturedImage(null);
    setResult(null);
    setError(null);
    setFeedbackSubmitted(false);
  };

  return (
    <div className="min-h-screen bg-background flex flex-col">
      {/* Background decorations */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-0 left-1/4 w-96 h-96 bg-primary/5 rounded-full blur-3xl" />
        <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-accent/5 rounded-full blur-3xl" />
        <div className="absolute top-1/2 left-0 w-64 h-64 bg-success/5 rounded-full blur-3xl" />
      </div>

      {/* Main content */}
      <div className="relative flex flex-col flex-1 max-w-lg mx-auto w-full">
        <NavigatorHeader />

        <main className="flex-1 px-4 pb-8 safe-bottom overflow-y-auto">
          <div className="space-y-6">
            {/* Camera section */}
            {!result && (
              <CameraView
                onCapture={handleCapture}
                capturedImage={capturedImage}
                onRetake={handleRetake}
              />
            )}

            {/* Error display */}
            {error && (
              <ErrorMessage message={error} onRetry={() => setError(null)} />
            )}

            {/* Loading state */}
            {isLoading && <LoadingState />}

            {/* Destination input */}
            {capturedImage && !isLoading && !result && (
              <DestinationInput
                onSubmit={handleSubmit}
                isLoading={isLoading}
                disabled={!capturedImage}
              />
            )}

            {/* Result display */}
            {result && (
              <NavigationResult result={result} onNewSearch={handleNewSearch} />
            )}

            {/* Feedback section - appears after result */}
            {result && !feedbackSubmitted && (
              <FeedbackSection onSubmit={handleFeedbackSubmit} />
            )}
          </div>
        </main>
      </div>
    </div>
  );
};

export default Index;
