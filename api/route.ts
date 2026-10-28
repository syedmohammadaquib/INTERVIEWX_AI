import { NextResponse } from "next/server";
import axios from "axios";

interface FeedbackRequest {
  conversation: { type: "user" | "assistant"; content: string }[];
}

export async function POST(request: Request) {
  try {
    const body: FeedbackRequest = await request.json();
    const { conversation } = body;

    if (
      !Array.isArray(conversation) ||
      conversation.length === 0 ||
      conversation.some(
        (message) =>
          !message ||
          !["user", "assistant"].includes(message.type) ||
          typeof message.content !== "string"
      )
    ) {
      return NextResponse.json(
        { isError: true, error: "Invalid or missing conversation data" },
        { status: 400 }
      );
    }

    try {
      const pythonEndpoint =
        process.env.MY_AI_PYTHON_ENDPOINT || "http://localhost:8000/analyze";
      const pythonResponse = await axios.post(
        pythonEndpoint,
        { conversation },
        { timeout: 15000 }
      );
      return NextResponse.json(pythonResponse.data);
    } catch (pythonError: unknown) {
      const message =
        pythonError instanceof Error
          ? pythonError.message
          : "Python analysis service request failed";
      console.error("[MY_AI] Python service failed:", message);
      return NextResponse.json(
        { isError: true, error: "MY_AI analysis service is unavailable" },
        { status: 503 }
      );
    }
  } catch (error: unknown) {
    const errorMessage =
      error instanceof Error ? error.message : "MY_AI service failed";
    console.error("❌ MY_AI ERROR:", error);
    return NextResponse.json(
      { isError: true, error: errorMessage },
      { status: 500 }
    );
  }
}
