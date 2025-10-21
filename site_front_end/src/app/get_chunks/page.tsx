"use client";
import { useState, useEffect } from "react";
import Link from "next/link";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { BASE_URL, GENERIC_ERROR_MESSAGE, safeFetch } from "@/lib/utils";

// Rating component for 0–100 using a number input (clicker)
function Rating({ value, onChange }: { value: number; onChange: (v: number) => void }) {
  return (
    <div className="flex items-center gap-2 justify-center">
      <button
        type="button"
        className="w-8 h-8 rounded border border-gray-300 bg-white text-lg font-bold hover:bg-gray-100 disabled:opacity-50"
        onClick={() => onChange(Math.max(0, value - 1))}
        disabled={value <= 0}
        aria-label="Decrease rating"
      >
        -
      </button>
      <input
        type="number"
        min={0}
        max={100}
        value={value}
        onChange={e => {
          let v = Number(e.target.value);
          if (isNaN(v)) v = 0;
          v = Math.max(0, Math.min(100, v));
          onChange(v);
        }}
        className="w-16 text-center border border-gray-300 rounded h-8 text-base font-semibold bg-white focus:outline-none focus:ring-2 focus:ring-primary"
        aria-label="Rating value"
      />
      <button
        type="button"
        className="w-8 h-8 rounded border border-gray-300 bg-white text-lg font-bold hover:bg-gray-100 disabled:opacity-50"
        onClick={() => onChange(Math.min(100, value + 1))}
        disabled={value >= 100}
        aria-label="Increase rating"
      >
        +
      </button>
    </div>
  );
}
type Chunk = {
  _id: string;
  text: string;
  metadata: {
    chunk_size: number;
    time_start: string;
    time_end: string;
    name_space: string;
  };
  sanity_data: {
    id: string;
    slug: string;
    title: string;
    transcriptURL: string;
    hash: string;
  };
  score: number;
};
export default function Home() {
  // App state
  const [question, setQuestion] = useState("");
  const [embedding_type, setEmbeddingType] = useState("");
  const [chunks, setChunks] = useState<Chunk[]>([]);
  const [ratings, setRatings] = useState<{ [id: string]: number }>({});
  const [overallComment, setOverallComment] = useState<string>("");
  const [step, setStep] = useState<"ask" | "rate" | "done">("ask");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [questions, setQuestions] = useState<string[]>([]);
  const [questionsLoading, setQuestionsLoading] = useState<boolean>(true);
  const [selectedIdx, setSelectedIdx] = useState<number | null>(null);
  const [questionsError, setQuestionsError] = useState<string | null>(null);

  // Load questions from backend
  useEffect(() => {
    let isMounted = true;
    (async () => {
      try {
        setQuestionsLoading(true);
        const res = await safeFetch(`/api/form/questions`);
        const data = await res.json();
        const arr = Array.isArray(data) ? data : (Array.isArray((data as any)?.questions) ? (data as any).questions : []);
        if (isMounted) setQuestions(arr);
        if (arr.length === 0) setQuestionsError(GENERIC_ERROR_MESSAGE);
      } catch (e) {
        console.error("Failed to load questions", e);
        if (isMounted) setQuestionsError(GENERIC_ERROR_MESSAGE);
      } finally {
        if (isMounted) setQuestionsLoading(false);
      }
    })();
    return () => {
      isMounted = false;
    };
  }, []);

  // Handle question submit
  async function handleAsk(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const res = await safeFetch("/api/get_chunks", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });
      const data = await res.json();
      if (data.error) {
        setError(GENERIC_ERROR_MESSAGE);
        setChunks([]);
        return;
      }
      setChunks(data.chunks);
      setEmbeddingType(data.embedding_type);
      setRatings({});
      setStep("rate");
    } catch (err) {
      setError(GENERIC_ERROR_MESSAGE);
    } finally {
      setLoading(false);
    }
  }

  // Handle ratings submit
  async function handleRatings(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      // Optimistic UI: go to done immediately
      setStep("done");
      await safeFetch("/api/upload_ratings", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_question: question,
          embedding_type: embedding_type,
          document: chunks.map((chunk) => [
            chunk, // full chunk object
            ratings[chunk._id] ?? 0 // user rating, 

          ]),
          comments: overallComment || undefined,
        }),
      });
    } catch (err) {
      setError(GENERIC_ERROR_MESSAGE);
      setStep("rate"); // Rollback
    } finally {
      setLoading(false);
    }
  }

  // UI
  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-background p-4">
      <Card className="w-full lg:max-w-4xl p-6 flex flex-col gap-10 shadow-lg">
        <h1 className="text-2xl font-bold mb-2 text-center">Chunk Rater</h1>
        <div className="flex justify-end -mt-2">
          <Link href="/">
            <Button variant="ghost" size="sm">Back to Home</Button>
          </Link>
        </div>
        {/* About Section */}
        {step === "ask" && (
          <div className="mb-4 p-6 bg-yellow-50 border-l-8 border-yellow-400 rounded shadow-md text-base text-muted-foreground">
            <h2 className="font-bold mb-3 text-2xl text-yellow-900">About</h2>
            <p className="text-lg leading-relaxed text-yellow-900">
              This platform is dedicated to exploring and evaluating text chunks extracted by a Retrieval-Augmented Generation system inspired by the teachings of Rabbi Joseph B. Soloveitchik, &quot;The Rav.&quot; Instead of full responses, you’ll engage with selected segments—chunks—derived from embedding his philosophical and halachic works as vectors.

Your role is to rate how well each chunk captures the meaning and nuance of the Rav’s original teachings. This feedback is crucial for testing and improving the embedding strategy and model accuracy in representing the Rav’s transcripts.

Whether you are a scholar, student, or admirer, your participation helps refine this system’s ability to faithfully reflect the Rav’s wisdom through vector embeddings. Thank you for contributing to this ongoing exploration.
            </p>
            <div className="mt-4">
              <ul className="list-disc pl-6 text-yellow-900">
              </ul>
            </div>
          </div>
        )}
        {step === "ask" && (
          <form onSubmit={handleAsk} className="flex flex-col gap-4">
            <div className="font-medium text-base">Select a question to rate:</div>
            {questionsLoading ? (
              <div className="text-sm text-muted-foreground">Loading questions…</div>
            ) : questions.length === 0 ? (
              <div className="text-sm text-red-600">{questionsError || GENERIC_ERROR_MESSAGE}</div>
            ) : (
              <div className="flex flex-col gap-2 max-h-80 overflow-y-auto">
                {questions.map((q, idx) => (
                  <button
                    key={idx}
                    type="button"
                    onClick={() => { setQuestion(q); setSelectedIdx(idx); }}
                    className={`text-left border rounded px-3 py-2 bg-white hover:bg-gray-50 focus:outline-none ${selectedIdx === idx ? "border-primary ring-2 ring-primary/30" : "border-gray-300"}`}
                  >
                    {q}
                  </button>
                ))}
              </div>
            )}
            <Button type="submit" disabled={loading || !question.trim()}>
              {loading ? "Loading..." : "Get Chunks"}
            </Button>
          </form>
        )}
        {step === "rate" && (
          <form onSubmit={handleRatings} className="flex flex-col gap-6">
            <div className="text-lg font-semibold text-center text-primary mb-2">
              {/* Rating instructions */}
              <div className="mb-4 p-3 bg-blue-50 border-l-4 border-blue-400 rounded text-blue-900 text-base">
                <strong>How to rate:</strong> Please rate each chunk based on how relevant it is to the question. Use a scale from 0 to 100, where:<br /><br />
                <div className="pl-2">
                  0 = completely unrelated to the question 
                  <br /><br />
                  50 = somewhat related or tangentially relevant
                  <br /><br />
                  100 = highly relevant — either directly answering the question or providing strong supporting context
                </div>
                <br />
                <em>Note: A chunk doesn’t need to directly answer the question to be valuable. If it’s on a similar topic or helps build context around the question, it may still be highly relevant. Please consider the overall usefulness and thematic connection when assigning a score.</em>
              </div>
              Question: <span className="font-normal text-foreground">{question}</span>
            </div>
            <div className="flex flex-col gap-4">
              {chunks.map((chunk) => (
                <div
                  key={chunk._id}
                  className="flex flex-col gap-2 border rounded-lg bg-muted/60 shadow-inner w-full"
                  style={{ minHeight: 220, maxHeight: 400 }}
                >
                  <div className="p-6 rounded-t-lg whitespace-pre-line leading-relaxed break-words text-base font-medium flex-1"
                    style={{ maxHeight: 320, overflowY: 'auto' }}
                  >
                    {chunk.text}
                  </div>
                  <div className="p-6 pt-4 border-t border-gray-200 rounded-b-lg bg-background flex items-center justify-center">
                    <Rating
                      value={ratings[chunk._id] ?? 50}
                      onChange={(v) => setRatings((r) => ({ ...r, [chunk._id]: v }))}
                    />
                  </div>
                </div>
              ))}
            </div>
            <div className="flex flex-col gap-2">
              <label className="text-sm" htmlFor="overall-comment">Optional: Add any overall comments about these chunks (e.g., clarity, usefulness, off-topic notes).</label>
              <textarea
                id="overall-comment"
                value={overallComment}
                onChange={e => setOverallComment(e.target.value)}
                className="border rounded px-3 py-2 text-base bg-white min-h-[90px]"
                placeholder="Add overall comments (optional)"
              />
            </div>
            <Button
              type="submit"
              disabled={loading || chunks.some((c) => ratings[c._id] === undefined)}
              className="w-full"
            >
              {loading ? "Submitting..." : "Submit Ratings"}
            </Button>
          </form>
        )}
        {step === "done" && (
          <div className="text-center flex flex-col gap-4 items-center">
            <div className="text-green-600 font-semibold text-lg">Thank you for your feedback!</div>
            <div className="text-muted-foreground text-base">Your question: <b>{question}</b></div>
            <Button onClick={() => { setStep("ask"); setQuestion(""); setChunks([]); setRatings({}); setOverallComment(""); }}>
              Rate another question
            </Button>
          </div>
        )}
        {error && <div className="text-red-500 text-center">{error}</div>}
      </Card>
      <footer className="mt-8 text-xs text-muted-foreground text-center opacity-80">
        &copy; {new Date().getFullYear()} Chunk Rater Demo
      </footer>
    </div>
  );
}


