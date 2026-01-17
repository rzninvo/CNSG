import { useRef, useEffect, useState, useCallback } from "react";
import { Camera, X, RotateCcw, ScanLine } from "lucide-react";
import { Button } from "./ui/button";
import { cn } from "@/lib/utils";

interface CameraViewProps {
  onCapture: (imageData: string) => void;
  capturedImage: string | null;
  onRetake: () => void;
}

export function CameraView({ onCapture, capturedImage, onRetake }: CameraViewProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isCapturing, setIsCapturing] = useState(false);

  const startCamera = useCallback(async () => {
    try {
      setError(null);
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "environment",
          width: { ideal: 1920 },
          height: { ideal: 1080 },
        },
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
      }
      setStream(mediaStream);
    } catch (err) {
      console.error("Camera error:", err);
      setError("Unable to access camera. Please check permissions.");
    }
  }, []);

  const stopCamera = useCallback(() => {
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      setStream(null);
    }
  }, [stream]);

  useEffect(() => {
    if (!capturedImage) {
      startCamera();
    }
    return () => {
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
    };
  }, [capturedImage]);

  const handleCapture = () => {
    if (!videoRef.current || !canvasRef.current) return;
    
    setIsCapturing(true);
    
    const video = videoRef.current;
    const canvas = canvasRef.current;
    
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    const ctx = canvas.getContext("2d");
    if (ctx) {
      ctx.drawImage(video, 0, 0);
      const imageData = canvas.toDataURL("image/jpeg", 0.85);
      
      setTimeout(() => {
        onCapture(imageData);
        stopCamera();
        setIsCapturing(false);
      }, 200);
    }
  };

  const handleRetake = () => {
    onRetake();
    startCamera();
  };

  return (
    <div className="relative w-full animate-fade-in">
      <div className="relative aspect-[4/3] rounded-2xl overflow-hidden glass border-2 border-border/50">
        {/* Scanning overlay effect */}
        {!capturedImage && !error && (
          <div className="absolute inset-0 z-10 pointer-events-none">
            <div className="absolute inset-4 border-2 border-primary/40 rounded-xl">
              <div className="absolute top-0 left-0 w-8 h-8 border-t-2 border-l-2 border-primary rounded-tl-lg" />
              <div className="absolute top-0 right-0 w-8 h-8 border-t-2 border-r-2 border-primary rounded-tr-lg" />
              <div className="absolute bottom-0 left-0 w-8 h-8 border-b-2 border-l-2 border-primary rounded-bl-lg" />
              <div className="absolute bottom-0 right-0 w-8 h-8 border-b-2 border-r-2 border-primary rounded-br-lg" />
            </div>
            <div className="absolute inset-4 overflow-hidden rounded-xl">
              <div 
                className="absolute w-full h-0.5 bg-gradient-to-r from-transparent via-primary to-transparent animate-[scan_2s_ease-in-out_infinite]"
                style={{
                  animation: "scan 2s ease-in-out infinite",
                }}
              />
            </div>
          </div>
        )}

        {error ? (
          <div className="absolute inset-0 flex items-center justify-center bg-card/90 p-6 text-center">
            <div>
              <Camera className="w-12 h-12 mx-auto mb-3 text-muted-foreground" />
              <p className="text-destructive font-medium">{error}</p>
              <Button variant="outline" size="sm" className="mt-4" onClick={startCamera}>
                Try Again
              </Button>
            </div>
          </div>
        ) : capturedImage ? (
          <div className="relative w-full h-full">
            <img 
              src={capturedImage} 
              alt="Captured location" 
              className="w-full h-full object-cover"
            />
            <Button
              variant="destructive"
              size="icon"
              className="absolute top-3 right-3 rounded-full"
              onClick={handleRetake}
            >
              <X className="w-5 h-5" />
            </Button>
          </div>
        ) : (
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            className="w-full h-full object-cover"
          />
        )}
        
        <canvas ref={canvasRef} className="hidden" />

        {/* Capture flash effect */}
        <div 
          className={cn(
            "absolute inset-0 bg-white pointer-events-none transition-opacity duration-200",
            isCapturing ? "opacity-80" : "opacity-0"
          )}
        />
      </div>

      {/* Capture button */}
      {!capturedImage && !error && (
        <div className="flex justify-center mt-4">
          <button
            onClick={handleCapture}
            className="group relative w-20 h-20 rounded-full bg-gradient-to-br from-primary to-accent p-1 transition-transform hover:scale-105 active:scale-95"
          >
            <div className="absolute inset-0 rounded-full animate-pulse-glow" />
            <div className="relative flex items-center justify-center w-full h-full rounded-full bg-card border-4 border-primary/50 group-hover:border-primary transition-colors">
              <Camera className="w-8 h-8 text-primary" />
            </div>
          </button>
        </div>
      )}

      {capturedImage && (
        <Button 
          variant="outline" 
          className="w-full mt-4"
          onClick={handleRetake}
        >
          <RotateCcw className="w-4 h-4" />
          Retake Photo
        </Button>
      )}

      <style>{`
        @keyframes scan {
          0%, 100% { top: 0; }
          50% { top: 100%; }
        }
      `}</style>
    </div>
  );
}
