"use client";
import { useState, useEffect } from "react";
import Link from "next/link";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { BASE_URL, GENERIC_ERROR_MESSAGE, safeFetch } from "@/lib/utils";

type TranscriptData = {
  sanity_data: {
    id: string;
    slug: string;
    title: string;
    transcriptURL: string;
    hash: string;
  };
  metadata: {
    chunk_size: number;
    time_start: string;
    time_end: string;
    name_space: string;
  };
  score: number;
};

// Simple 0–100 rating control like chunks page
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

export default function GetFullResponsePage() {
  const [question, setQuestion] = useState("");
  const [responses, setResponses] = useState<{ message: string; transcript_data: TranscriptData[]; prompt_id?: string }[]>([]);
  const [step, setStep] = useState<"ask" | "view" | "done">("ask");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // Comparative rankings: rank 1..N (1 = best). Unique ranks.
  const [ranks, setRanks] = useState<{ [idx: number]: number }>({});
  const [questions, setQuestions] = useState<string[]>([]);
  const [questionsLoading, setQuestionsLoading] = useState<boolean>(true);
  const [selectedIdx, setSelectedIdx] = useState<number | null>(null);
  const [questionsError, setQuestionsError] = useState<string | null>(null);
  const [overallComment, setOverallComment] = useState<string>("");

  const interferenceMeta = [
    { level: 1, title: 'Least LLM interference', detail: 'Direct quotes and minimal paraphrasing.' },
    { level: 2, title: 'Moderate LLM interference', detail: 'Balanced summary with light synthesis.' },
    { level: 3, title: 'Most LLM interference', detail: 'More paraphrasing and synthesis; freer to the source.' },
  ] as const;

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

  async function handleAsk(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const res = await safeFetch("/api/get_full_response", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });
      const data = await res.json();
      if (data.error) {
        setError(GENERIC_ERROR_MESSAGE);
        setResponses([]);
        return;
      }
      setResponses(Array.isArray(data.responses) ? data.responses : []);
      setRanks({});
      setStep("view");
    } catch (err) {
      setError(GENERIC_ERROR_MESSAGE);
    } finally {
      setLoading(false);
    }
  }

  async function handleSubmitRating(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      setStep("done"); // optimistic
      // Compose rankings payload: array of { prompt_id, rank }
      const rankings = responses.slice(0, 3).map((r, idx) => ({
        prompt_id: r.prompt_id,
        rank: ranks[idx],
      }));
      await safeFetch("/api/upload_full_response_rating", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_question: question,
          rankings,
          comments: overallComment || undefined,
        }),
      });
    } catch (err) {
      setError(GENERIC_ERROR_MESSAGE);
      setStep("view");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-background p-4">
      {loading && <LoadingOverlay />}
      <Card className="w-full lg:max-w-4xl p-6 flex flex-col gap-10 shadow-lg">
        <h1 className="text-2xl font-bold mb-2 text-center">Full Response</h1>
        <div className="flex justify-end -mt-2">
          <Link href="/">
            <Button variant="ghost" size="sm">Back to Home</Button>
          </Link>
        </div>

        {step === "ask" && (
          <div className="mb-4 p-6 bg-green-50 border-l-8 border-green-400 rounded shadow-md text-base text-muted-foreground">
            <h2 className="font-bold mb-3 text-2xl text-green-900">About</h2>
            <p className="text-lg leading-relaxed text-green-900">
              Ask a question and receive the full generated response using context from the most relevant transcript segments inspired by the teachings of Rabbi Joseph B. Soloveitchik, "The Rav."
            </p>
            <div className="mt-4">
              <ul className="list-disc pl-6 text-green-900">
              </ul>
            </div>
          </div>
        )}

        {step === "ask" && (
          <form onSubmit={handleAsk} className="flex flex-col gap-4">
            <div className="font-medium text-base">Select a question to ask:</div>
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
              {loading ? "⏳" : "Get Full Response"}
            </Button>
          </form>
        )}

        {step === "view" && (
          <form onSubmit={handleSubmitRating} className="flex flex-col gap-6">
            <div className="text-lg font-semibold text-center text-primary">
              Question: <span className="font-normal text-foreground">{question}</span>
            </div>
            <div className="px-4 py-2 text-sm text-muted-foreground text-center bg-muted/40 rounded">
              Responses differ by LLM interference level: 1 (least) → 3 (most).
            </div>
            <div className="flex flex-col gap-6">
              {responses.slice(0, 3).map((r, idx) => (
                <div key={idx} className="flex flex-col gap-3 border rounded-lg bg-muted/60 shadow-inner p-0">
                  <div className="px-6 pt-4">
                    <span className="inline-flex items-center rounded-full bg-white/80 text-foreground px-2 py-0.5 text-xs font-semibold border">
                      Level {interferenceMeta[idx]?.level ?? idx + 1}
                    </span>
                    <div className="text-sm text-muted-foreground mt-1">
                      {(interferenceMeta[idx]?.title || '') + (interferenceMeta[idx]?.detail ? ` — ${interferenceMeta[idx].detail}` : '')}
                    </div>
                  </div>
                  <div className="p-6 leading-relaxed break-words text-base">
                    <div
                      className="prose prose-sm max-w-none"
                      dangerouslySetInnerHTML={{ __html: simpleMarkdownToHtml(r.message) }}
                    />
                  </div>
                  {Array.isArray(r.transcript_data) && r.transcript_data.length > 0 && (
                    <div className="text-sm text-muted-foreground px-6 pb-4">
                      <div className="font-semibold mb-2">Context Sources</div>
                      <ul className="list-disc pl-6 space-y-1">
                        {r.transcript_data.slice(0, 5).map((t, i) => (
                          <li key={i}>
                            {t.sanity_data?.title || t.sanity_data?.slug || 'Source'}
                            {t.metadata?.time_start && t.metadata?.time_end ? (
                              <span className="ml-2 opacity-80">({t.metadata.time_start} - {t.metadata.time_end})</span>
                            ) : null}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                  <div className="px-6 pb-6 flex flex-col items-start gap-2">
                    <div className="mb-1 p-3 bg-blue-50 border-l-4 border-blue-400 rounded text-blue-900 text-sm w-full">
                    <strong>Rank these responses:</strong> Assign a unique rank from <strong>1 - Best</strong> to <strong>3 - Worst</strong> based on how helpful you find them.
                    </div>
                    <label className="text-sm">Rank</label>
                    <select
                      value={ranks[idx] ?? ''}
                      onChange={e => setRanks(prev => ({ ...prev, [idx]: Number(e.target.value) }))}
                      className="border rounded px-3 py-2 text-base bg-white"
                      required
                    >
                      <option value="" disabled>Select rank</option>
                      {[1,2,3].map(n => (
                        <option key={n} value={n} disabled={Object.values(ranks).includes(n) && ranks[idx] !== n}>{n}</option>
                      ))}
                    </select>
                  </div>
                </div>
              ))}
            </div>
            <div className="flex flex-col gap-2">
              <label className="text-sm" htmlFor="overall-comment">Optional: Your comments about the responses are helpful. For example, you can note if the answer was clear, useful, or off-topic, e.g., “The explanation was detailed and easy to follow.”</label>
              <textarea
                id="overall-comment"
                value={overallComment}
                onChange={e => setOverallComment(e.target.value)}
                className="border rounded px-3 py-2 text-base bg-white min-h-[90px]"
                placeholder="Add any comments about the responses (optional)"
              />
            </div>
            <Button type="submit" disabled={loading || !areRanksCompleteAndUnique(ranks, Math.min(3, responses.length))} className="w-full">{loading ? "⏳" : "Submit Rankings"}</Button>
            <div className="flex gap-3">
              <Button type="button" onClick={() => { setStep("ask"); setQuestion(""); setResponses([]); setRanks({}); setOverallComment(""); }}>
                Ask another question
              </Button>
              <Link href="/get_chunks">
                <Button type="button" variant="secondary">Rate chunks instead</Button>
              </Link>
            </div>
          </form>
        )}

        {step === "done" && (
          <div className="text-center flex flex-col gap-4 items-center">
            <div className="text-green-600 font-semibold text-lg">Thank you for your feedback!</div>
            <div className="text-muted-foreground text-base">Your question: <b>{question}</b></div>
            <Button onClick={() => { setStep("ask"); setQuestion(""); setResponses([]); setRanks({}); setOverallComment(""); }}>
              Rate another response
            </Button>
          </div>
        )}

        {error && <div className="text-red-500 text-center">{error}</div>}
      </Card>
      <footer className="mt-8 text-xs text-muted-foreground text-center opacity-80">
        &copy; {new Date().getFullYear()} Full Response Demo
      </footer>
    </div>
  );
}


function LoadingOverlay() {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-background/80 backdrop-blur-sm">
      <Card className="w-full max-w-md p-6 flex flex-col items-center justify-center shadow-xl">
        <div className="text-4xl" aria-hidden>⏳</div>
      </Card>
    </div>
  );
}

// very small markdown-to-HTML for bold, italics, blockquotes, and line breaks
function simpleMarkdownToHtml(input: string): string {
  if (!input) return '';
  let html = input;
  html = html.replace(/^>\s?(.*)$/gm, '<blockquote>$1</blockquote>');
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');
  html = html.replace(/\n/g, '<br />');
  return html;
}

function areRanksCompleteAndUnique(ranks: { [idx: number]: number }, count: number): boolean {
  const values = Array.from({ length: count }, (_, i) => ranks[i]);
  if (values.some(v => typeof v !== 'number')) return false;
  const set = new Set(values);
  if (set.size !== count) return false;
  for (const v of values) {
    if (v < 1 || v > count) return false;
  }
  return true;
}

