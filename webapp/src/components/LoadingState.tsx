import { Compass, MapPin, Route } from "lucide-react";

export function LoadingState() {
  return (
    <div className="flex flex-col items-center justify-center py-12 animate-fade-in">
      {/* Animated compass */}
      <div className="relative mb-6">
        <div className="absolute inset-0 animate-radar rounded-full bg-primary/20" />
        <div className="absolute inset-0 animate-radar rounded-full bg-primary/20" style={{ animationDelay: "0.5s" }} />
        <div className="relative flex items-center justify-center w-20 h-20 rounded-full bg-gradient-to-br from-primary/20 to-accent/20 border-2 border-primary/30">
          <Compass className="w-10 h-10 text-primary animate-spin-slow" />
        </div>
      </div>

      {/* Loading text */}
      <div className="text-center space-y-2">
        <h3 className="text-lg font-semibold text-foreground">Analyzing Location</h3>
        <p className="text-sm text-muted-foreground">Calculating the best route for you...</p>
      </div>

      {/* Progress indicators */}
      <div className="flex items-center gap-4 mt-6">
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <MapPin className="w-4 h-4 text-success animate-pulse" />
          <span>Detecting position</span>
        </div>
        <div className="w-1 h-1 rounded-full bg-border" />
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <Route className="w-4 h-4 text-primary animate-pulse" style={{ animationDelay: "0.3s" }} />
          <span>Planning route</span>
        </div>
      </div>

      {/* Shimmer bar */}
      <div className="w-48 h-1 mt-6 rounded-full overflow-hidden bg-secondary">
        <div 
          className="h-full w-1/2 bg-gradient-to-r from-primary via-accent to-primary animate-shimmer"
          style={{ backgroundSize: "200% 100%" }}
        />
      </div>
    </div>
  );
}
