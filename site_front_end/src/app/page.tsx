"use client";
import Link from "next/link";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";

export default function Home() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-background to-muted/30">
      <main className="mx-auto max-w-5xl px-4 py-12 md:py-16">
        {/* Hero */}
        <section className="text-center mb-10 md:mb-14">
          <div className="inline-flex items-center rounded-full border bg-background px-3 py-1 text-xs text-muted-foreground shadow-sm">
            Retrieval-Augmented Evaluation
          </div>
          <h1 className="mt-4 text-4xl md:text-5xl font-bold tracking-tight">
            The Rav RAG Evaluation
          </h1>
          <p className="mx-auto mt-4 max-w-2xl text-base md:text-lg text-muted-foreground">
            Explore the teachings of Rabbi Joseph B. Soloveitchik through an AI system
            grounded in curated texts. Help us measure and improve retrieval and
            response quality.
          </p>
          <div className="mt-6 flex flex-col sm:flex-row gap-3 justify-center">
            <Link href="/get_full_response">
              <Button size="lg">Start with Full Response</Button>
            </Link>
            <Link href="/get_chunks">
              <Button size="lg" variant="outline">Try Chunk Evaluation</Button>
            </Link>
          </div>
        </section>

        {/* About */}
        <section className="mb-10 md:mb-14">
          <Card className="p-6 md:p-8">
            <div className="space-y-3 md:space-y-4">
              <h2 className="text-xl md:text-2xl font-semibold">About the Project</h2>
              <p className="text-base leading-relaxed text-muted-foreground">
                This platform uses Retrieval-Augmented Generation (RAG) to surface relevant
                passages from The Rav’s philosophical and halakhic writings, pairing them with
                generated analysis. Your evaluations directly inform how we tune our
                retrieval pipeline and model behavior.
              </p>
              <ul className="grid gap-2 text-sm md:text-base text-muted-foreground md:grid-cols-3">
                <li className="flex items-start gap-2"><span>✅</span><span>Measure retrieval precision and coverage</span></li>
                <li className="flex items-start gap-2"><span>✅</span><span>Assess end-to-end answer quality</span></li>
                <li className="flex items-start gap-2"><span>✅</span><span>Ensure fidelity to source texts</span></li>
              </ul>
            </div>
          </Card>
        </section>

        {/* Modes */}
        <section className="mb-10 md:mb-14">
          <h2 className="text-xl md:text-2xl font-semibold mb-4">Choose an Evaluation Mode</h2>
          <div className="grid gap-4 md:grid-cols-2">
            <Card className="p-5 md:p-6">
              <div className="space-y-2">
                <h3 className="text-lg font-semibold">📄 Chunk Evaluation</h3>
                <p className="text-sm text-muted-foreground">Best for testing retrieval quality</p>
                <ul className="list-disc pl-5 text-sm md:text-base space-y-1">
                  <li>Submit a question from our curated list</li>
                  <li>Receive up to five relevant text chunks</li>
                  <li>Score each chunk’s relevance on a 0–100 scale</li>
                </ul>
                <div className="pt-2">
                  <Link href="/get_chunks">
                    <Button>Go to Chunk Evaluation</Button>
                  </Link>
                </div>
              </div>
            </Card>
            <Card className="p-5 md:p-6">
              <div className="space-y-2">
                <h3 className="text-lg font-semibold">🔍 Full Response Evaluation</h3>
                <p className="text-sm text-muted-foreground">Best for overall answer quality</p>
                <ul className="list-disc pl-5 text-sm md:text-base space-y-1">
                  <li>Ask about The Rav’s thought and teachings</li>
                  <li>Review three responses with varying model guidance</li>
                  <li>Rank them 1 (best), 2, and 3 (worst)</li>
                </ul>
                <div className="pt-2">
                  <Link href="/get_full_response">
                    <Button variant="outline">Go to Full Response</Button>
                  </Link>
                </div>
              </div>
            </Card>
          </div>
        </section>

        {/* Evaluator guidance */}
        <section>
          <Card className="p-6 md:p-8">
            <h2 className="text-xl md:text-2xl font-semibold mb-3">How Your Feedback Helps</h2>
            <ul className="list-disc pl-6 text-sm md:text-base text-muted-foreground space-y-1">
              <li>Improve retrieval ranking and chunk selection</li>
              <li>Refine response structure, tone, and faithfulness to sources</li>
              <li>Guide future dataset curation and evaluation criteria</li>
            </ul>
            <p className="mt-3 text-sm md:text-base">
              Ready to begin? Choose a mode above and start evaluating.
            </p>
          </Card>
        </section>

        {/* Footer note */}
        <p className="mt-8 text-center text-xs text-muted-foreground">
          Built with RAG for respectful engagement with The Rav’s writings.
        </p>
      </main>
    </div>
  );
}


