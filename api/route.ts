/* eslint-disable @typescript-eslint/no-explicit-any */

import { NextResponse } from "next/server";
import { PromptTemplate } from "@langchain/core/prompts";
import { StructuredOutputParser } from "@langchain/core/output_parsers";
import { z } from "zod";
import axios from 'axios';

interface FeedbackRequest {
  conversation: { type: "user" | "assistant"; content: string }[];
}

// --- Zod Schema (enhanced for detailed feedback) ---
const feedbackSchema = z.object({
  feedback: z.object({
    rating: z.object({
      relevance: z.number().min(0).max(5).describe("Relevance of candidate's background and experience to the job requirements"),
      technicalDepth: z.number().min(0).max(5).describe("Depth of technical knowledge and expertise demonstrated"),
      clarity: z.number().min(0).max(5).describe("Clarity of thought and expression in responses"),
      communicationQuality: z.number().min(0).max(5).describe("Overall quality of communication, including listening and articulation"),
    }),
    summary: z.string().describe("Comprehensive 4-5 line summary highlighting key strengths and areas for improvement"),
    recommendation: z.enum(["Yes", "No", "Maybe"]).describe("Clear hiring recommendation"),
    recommendationMessage: z.string().describe("Detailed justification for the recommendation with specific examples"),
    strengths: z.array(z.string()).describe("List of candidate's key strengths demonstrated during interview"),
    improvements: z.array(z.string()).describe("List of areas where candidate could improve"),
    technicalAssessment: z.string().describe("Specific evaluation of technical skills and knowledge"),
    communicationAssessment: z.string().describe("Evaluation of communication skills and professional demeanor"),
    overallConfidence: z.number().min(0).max(100).describe("Confidence level in the assessment (0-100%)"),
  }),
});

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const parser = StructuredOutputParser.fromZodSchema(feedbackSchema as any);

// --- Simple Prompt Template ---
const promptTemplate = PromptTemplate.fromTemplate(`
You are an expert technical interviewer and hiring manager providing comprehensive feedback on a job interview.

INTERVIEW CONVERSATION:
{conversation}

ANALYSIS INSTRUCTIONS:
Analyze the candidate's performance and provide detailed feedback in the following format:

1. Rate each category on a 0-5 scale:
   - Relevance: How well their background matches the job requirements
   - Technical Depth: Quality of technical knowledge demonstrated
   - Clarity: How clear and structured their responses are
   - Communication Quality: Overall communication skills

2. Provide:
   - A comprehensive 4-5 line summary
   - Clear recommendation (Yes/No/Maybe) with justification
   - List of strengths demonstrated
   - Areas for improvement
   - Technical assessment details
   - Communication assessment details
   - Confidence level (0-100%)

Return ONLY raw JSON. No markdown blocks, no explanations.

{format_instructions}
`);

// --- MY_AI Primary Analysis ---
export async function POST(request: Request) {
  try {
    const body: FeedbackRequest = await request.json();
    const { conversation } = body;

    if (!conversation || conversation.length === 0) {
      return NextResponse.json(
        { isError: true, error: "Missing conversation data" },
        { status: 400 }
      );
    }

    const conversationString = conversation
      .map((m) => `${m.type === "user" ? "User" : "Assistant"}: ${m.content} `)
      .join("\n");

    const input = {
      conversation: conversationString,
      format_instructions: parser.getFormatInstructions(),
    };

    // Call Python service instead of mock response
    console.log('[MY_AI] Calling Python analysis service...');
    
    const pythonEndpoint = process.env.MY_AI_PYTHON_ENDPOINT || 'http://localhost:8000/analyze';
    
    try {
      const pythonResponse = await axios.post(pythonEndpoint, {
        conversation: conversation
      }, {
        timeout: 15000 // 15 second timeout for analysis
      });

      console.log('[MY_AI] Python service analysis successful');
      return NextResponse.json(pythonResponse.data);
      
    } catch (pythonError: any) {
      console.error('[MY_AI] Python service failed:', pythonError.message);
      
      // Return mock response as fallback when Python service is down
      console.log('[MY_AI] Using fallback mock response');
      
      const mockResult = {
        feedback: {
          rating: {
            relevance: 3,
            technicalDepth: 3,
            clarity: 3,
            communicationQuality: 3,
          },
          summary: "Candidate demonstrated average performance during the interview. Showed basic understanding of concepts but needs more depth in technical areas.",
          recommendation: "Maybe",
          recommendationMessage: "Candidate has potential but requires further evaluation to confirm fit for the role.",
          strengths: ["Basic technical knowledge", "Clear communication", "Positive attitude"],
          improvements: ["Deepen technical expertise", "Provide more detailed examples", "Improve problem-solving approach"],
          technicalAssessment: "Candidate shows foundational knowledge but lacks advanced technical depth required for senior roles.",
          communicationAssessment: "Communication is clear and structured, but could benefit from more articulate explanations.",
          overallConfidence: 60,
        },
      };

      return NextResponse.json({ data: mockResult });
    }
    
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : 'MY_AI service failed';
    console.error("❌ MY_AI ERROR:", error);
    return NextResponse.json(
      { isError: true, error: errorMessage, useFallback: true },
      { status: 500 }
    );
  }
}
