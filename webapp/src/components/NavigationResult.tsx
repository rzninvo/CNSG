import { CheckCircle2, Navigation, RefreshCw, ArrowRight, MapPin } from "lucide-react";
import { Button } from "./ui/button";

interface NavigationResultProps {
  result: string;
  onNewSearch: () => void;
}

export function NavigationResult({ result, onNewSearch }: NavigationResultProps) {
  // Parse the result into steps if possible
  const steps = result.split(/(?:\d+\.\s*|\n+)/).filter(step => step.trim());

  return (
    <div className="animate-slide-up space-y-4">
      {/* Success header */}
      <div className="glass-strong rounded-2xl p-5 border-l-4 border-success">
        <div className="flex items-start gap-3">
          <div className="flex-shrink-0 w-10 h-10 rounded-full bg-success/20 flex items-center justify-center">
            <CheckCircle2 className="w-5 h-5 text-success" />
          </div>
          <div>
            <h3 className="font-semibold text-foreground flex items-center gap-2">
              Route Found
              <Navigation className="w-4 h-4 text-primary" />
            </h3>
            <p className="text-sm text-muted-foreground mt-0.5">
              Follow these directions to reach your destination
            </p>
          </div>
        </div>
      </div>

      {/* Navigation steps */}
      <div className="glass rounded-2xl p-5 space-y-4">
        <div className="flex items-center gap-2 text-sm font-medium text-muted-foreground mb-4">
          <MapPin className="w-4 h-4 text-primary" />
          Navigation Steps
        </div>

        {steps.length > 1 ? (
          <div className="space-y-3">
            {steps.map((step, index) => (
              <div 
                key={index}
                className="flex items-start gap-3 animate-fade-in"
                style={{ animationDelay: `${index * 100}ms` }}
              >
                <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gradient-to-br from-primary/20 to-accent/20 border border-primary/30 flex items-center justify-center">
                  <span className="text-sm font-semibold text-primary">{index + 1}</span>
                </div>
                <div className="flex-1 pt-1">
                  <p className="text-foreground leading-relaxed">{step.trim()}</p>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-foreground leading-relaxed whitespace-pre-wrap">{result}</p>
        )}

        {/* Destination indicator */}
        <div className="flex items-center gap-2 pt-4 border-t border-border mt-4">
          <ArrowRight className="w-4 h-4 text-success" />
          <span className="text-sm text-success font-medium">You'll arrive at your destination</span>
        </div>
      </div>

      {/* New search button */}
      <Button
        variant="outline"
        size="lg"
        className="w-full"
        onClick={onNewSearch}
      >
        <RefreshCw className="w-4 h-4" />
        Start New Search
      </Button>
    </div>
  );
}
