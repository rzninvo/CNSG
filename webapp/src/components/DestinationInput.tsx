import { useState } from "react";
import { Navigation, Loader2, Sparkles } from "lucide-react";
import { Button } from "./ui/button";
import { cn } from "@/lib/utils";

interface DestinationInputProps {
  onSubmit: (destination: string) => void;
  isLoading: boolean;
  disabled: boolean;
}

const suggestions = [
  "HG E 3",
  "Cafeteria",
  "Exit",
  "HG E 7",
  "Female Bathroom",
  "Male Bathroom",
];

export function DestinationInput({
  onSubmit,
  isLoading,
  disabled,
}: DestinationInputProps) {
  const [destination, setDestination] = useState("");

  const handleSubmit = () => {
    if (destination.trim()) {
      onSubmit(destination.trim());
    }
  };

  const handleSuggestionClick = (suggestion: string) => {
    setDestination(`I want to go to the ${suggestion}`);
  };

  return (
    <div className="space-y-4 animate-slide-up">
      <div className="glass rounded-2xl p-4">
        <label className="flex items-center gap-2 text-sm font-medium text-muted-foreground mb-3">
          <Navigation className="w-4 h-4 text-primary" />
          Where do you want to go?
        </label>

        <textarea
          value={destination}
          onChange={(e) => setDestination(e.target.value)}
          placeholder="E.g.: I want to go to the kitchen, main hall, exit..."
          className={cn(
            "w-full min-h-[100px] p-4 rounded-xl resize-none",
            "bg-secondary/50 border-2 border-border",
            "text-foreground placeholder:text-muted-foreground",
            "focus:outline-none focus:border-primary/50 focus:bg-secondary/70",
            "transition-all duration-300"
          )}
          disabled={disabled || isLoading}
        />

        {/* Quick suggestions */}
        <div className="mt-3">
          <p className="text-xs text-muted-foreground mb-2 flex items-center gap-1">
            <Sparkles className="w-3 h-3" />
            Quick destinations
          </p>
          <div className="flex flex-wrap gap-2">
            {suggestions.map((suggestion) => (
              <button
                key={suggestion}
                onClick={() => handleSuggestionClick(suggestion)}
                disabled={disabled || isLoading}
                className={cn(
                  "px-3 py-1.5 text-xs font-medium rounded-full",
                  "bg-secondary hover:bg-secondary/80 border border-border",
                  "text-muted-foreground hover:text-foreground",
                  "transition-all duration-200 hover:border-primary/30",
                  "disabled:opacity-50 disabled:cursor-not-allowed"
                )}
              >
                {suggestion}
              </button>
            ))}
          </div>
        </div>
      </div>

      <Button
        variant="gradient"
        size="xl"
        className="w-full"
        onClick={handleSubmit}
        disabled={disabled || !destination.trim() || isLoading}
      >
        {isLoading ? (
          <>
            <Loader2 className="w-5 h-5 animate-spin" />
            Calculating Route...
          </>
        ) : (
          <>
            <Navigation className="w-5 h-5" />
            Find My Route
          </>
        )}
      </Button>
    </div>
  );
}
