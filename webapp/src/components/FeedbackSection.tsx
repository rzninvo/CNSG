import { useState } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Star, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";

interface FeedbackSectionProps {
  onSubmit: (ratings: FeedbackRatings, comment: string) => void;
  isSubmitting?: boolean;
}

export interface FeedbackRatings {
  usefulness: number;
  accuracy: number;
  clarity: number;
  response_time: number;
  easeOfUse: number;
}

const questions = [
  {
    key: "usefulness" as keyof FeedbackRatings,
    label: "How useful were the mentioned objects in the experience?",
  },
  {
    key: "accuracy" as keyof FeedbackRatings,
    label:
      "How correct and consistent were the spatial and directional instruction?",
  },
  {
    key: "clarity" as keyof FeedbackRatings,
    label: "How clear and human-like were the instructions provided?",
  },
  {
    key: "response_time" as keyof FeedbackRatings,
    label: "How would you rate the response time of the system?",
  },
  {
    key: "easeOfUse" as keyof FeedbackRatings,
    label: "How smooth was your overall experience with the app?",
  },
];

export function FeedbackSection({
  onSubmit,
  isSubmitting = false,
}: FeedbackSectionProps) {
  const [ratings, setRatings] = useState<FeedbackRatings>({
    accuracy: 0,
    clarity: 0,
    usefulness: 0,
    response_time: 0,
    easeOfUse: 0,
  });
  const [comment, setComment] = useState("");
  const [hoveredStars, setHoveredStars] = useState<{ [key: string]: number }>(
    {}
  );

  const handleRating = (question: keyof FeedbackRatings, rating: number) => {
    setRatings((prev) => ({ ...prev, [question]: rating }));
  };

  const handleSubmit = () => {
    const allRated = Object.values(ratings).every((rating) => rating > 0);
    if (!allRated) {
      return;
    }
    onSubmit(ratings, comment);
  };

  const allRated = Object.values(ratings).every((rating) => rating > 0);

  return (
    <Card className="animate-in slide-in-from-bottom-4 border-2">
      <CardHeader className="pb-4">
        <CardTitle className="text-xl">Help Us Improve</CardTitle>
        <CardDescription>
          Please rate your experience (all questions required)
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        {questions.map(({ key, label }) => (
          <div key={key} className="space-y-2">
            <label className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70">
              {label}
            </label>
            <div className="flex gap-1">
              {[1, 2, 3, 4, 5].map((star) => (
                <button
                  key={star}
                  type="button"
                  onClick={() => handleRating(key, star)}
                  onMouseEnter={() =>
                    setHoveredStars((prev) => ({ ...prev, [key]: star }))
                  }
                  onMouseLeave={() =>
                    setHoveredStars((prev) => ({ ...prev, [key]: 0 }))
                  }
                  disabled={isSubmitting}
                  className="transition-transform hover:scale-110 active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <Star
                    className={cn(
                      "w-8 h-8 transition-colors",
                      (hoveredStars[key] || ratings[key]) >= star
                        ? "fill-yellow-400 text-yellow-400"
                        : "text-gray-300"
                    )}
                  />
                </button>
              ))}
            </div>
          </div>
        ))}

        <div className="space-y-2 pt-2">
          <label htmlFor="comment" className="text-sm font-medium leading-none">
            Additional comments (optional)
          </label>
          <Textarea
            id="comment"
            placeholder="Tell us more about your experience..."
            value={comment}
            onChange={(e) => setComment(e.target.value)}
            rows={4}
            className="resize-none"
            disabled={isSubmitting}
          />
        </div>

        <Button
          onClick={handleSubmit}
          disabled={!allRated || isSubmitting}
          className="w-full"
          size="lg"
        >
          {isSubmitting ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              Submitting...
            </>
          ) : (
            "Submit Feedback"
          )}
        </Button>

        {!allRated && !isSubmitting && (
          <p className="text-xs text-muted-foreground text-center">
            Please rate all questions before submitting
          </p>
        )}
      </CardContent>
    </Card>
  );
}
