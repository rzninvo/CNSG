import { AlertTriangle, RefreshCw, WifiOff } from "lucide-react";
import { Button } from "./ui/button";

interface ErrorMessageProps {
  message: string;
  onRetry?: () => void;
}

export function ErrorMessage({ message, onRetry }: ErrorMessageProps) {
  const isNetworkError = message.toLowerCase().includes("network") || 
                         message.toLowerCase().includes("internet") ||
                         message.toLowerCase().includes("timeout");

  return (
    <div className="glass rounded-2xl p-5 border-l-4 border-destructive animate-scale-in">
      <div className="flex items-start gap-3">
        <div className="flex-shrink-0 w-10 h-10 rounded-full bg-destructive/20 flex items-center justify-center">
          {isNetworkError ? (
            <WifiOff className="w-5 h-5 text-destructive" />
          ) : (
            <AlertTriangle className="w-5 h-5 text-destructive" />
          )}
        </div>
        <div className="flex-1">
          <h3 className="font-semibold text-foreground">
            {isNetworkError ? "Connection Error" : "Something went wrong"}
          </h3>
          <p className="text-sm text-muted-foreground mt-1">{message}</p>
          
          {onRetry && (
            <Button
              variant="outline"
              size="sm"
              className="mt-3"
              onClick={onRetry}
            >
              <RefreshCw className="w-4 h-4" />
              Try Again
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}
