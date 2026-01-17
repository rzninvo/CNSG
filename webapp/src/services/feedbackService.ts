import { config } from "@/lib/config";
import type { FeedbackRatings } from "@/components/FeedbackSection";

interface FeedbackData {
  usefulness: number;
  accuracy: number;
  clarity: number;
  response_time: number;
  easeOfUse: number;
  comment: string;
}

export async function submitFeedbackToGoogleSheets(
  ratings: FeedbackRatings,
  comment: string
): Promise<{ success: boolean; error?: string }> {
  console.log("=== START submitFeedbackToGoogleSheets ===");

  // Check 1: Verifica URL
  console.log("Google Sheets URL:", config.googleSheetsUrl);
  console.log("URL type:", typeof config.googleSheetsUrl);
  console.log("URL length:", config.googleSheetsUrl?.length);

  if (!config.googleSheetsUrl || config.googleSheetsUrl.trim() === "") {
    console.error("❌ URL is empty or undefined!");
    return { success: false, error: "Google Sheets URL not configured" };
  }

  try {
    const feedbackData: FeedbackData = {
      accuracy: ratings.accuracy,
      clarity: ratings.clarity,
      easeOfUse: ratings.easeOfUse,
      response_time: ratings.response_time,
      usefulness: ratings.usefulness,
      comment: comment.trim(),
    };

    console.log("Feedback data prepared:", feedbackData);
    console.log("Feedback data JSON:", JSON.stringify(feedbackData));

    // Check 2: Verifica che fetch esista
    if (typeof fetch === "undefined") {
      console.error("❌ fetch is not available!");
      return { success: false, error: "Fetch API not available" };
    }

    console.log("✓ About to send fetch request...");
    console.log("Request details:", {
      url: config.googleSheetsUrl,
      method: "POST",
      mode: "no-cors",
    });

    // Invia la richiesta
    const response = await fetch(config.googleSheetsUrl, {
      method: "POST",
      mode: "no-cors",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(feedbackData),
    });

    console.log("✓ Fetch completed");
    console.log("Response object:", response);
    console.log("Response type:", response.type);
    console.log("Response status:", response.status);

    // Con no-cors, response.type sarà "opaque"
    if (response.type === "opaque") {
      console.log("✓ Request sent successfully (opaque response from no-cors)");
      return { success: true };
    }

    return { success: true };
  } catch (error) {
    console.error("❌ Error in submitFeedbackToGoogleSheets:", error);
    console.error("Error type:", error?.constructor?.name);
    console.error(
      "Error message:",
      error instanceof Error ? error.message : String(error)
    );
    console.error("Error stack:", error instanceof Error ? error.stack : "N/A");

    return {
      success: false,
      error: error instanceof Error ? error.message : "Unknown error",
    };
  } finally {
    console.log("=== END submitFeedbackToGoogleSheets ===");
  }
}
