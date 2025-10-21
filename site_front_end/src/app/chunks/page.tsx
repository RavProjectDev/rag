"use client";
import Link from "next/link";
import { Card } from "@/components/ui/card";

export default function ChunksInfoPage() {
  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-background p-4">
      <Card className="w-full lg:max-w-3xl p-6 flex flex-col gap-6 shadow-lg">
        <h1 className="text-2xl font-bold text-center">About: chunks</h1>
        <div className="text-base text-muted-foreground leading-relaxed">
          <p>
            The <code>/chunks</code> page is an informational alias of <code>/get_chunks</code>. It describes how the
            project surfaces relevant text chunks for a given question, enabling users to rate
            their relevance and help optimize retrieval and embedding strategies.
          </p>
          <ul className="list-disc pl-6 mt-4">
            <li><strong>Same purpose</strong> as <code>/get_chunks</code>, presented as an alias.</li>
            <li><strong>Guidance</strong>: Use the home page to choose a question and start rating.</li>
          </ul>
        </div>
        <div className="flex gap-3 justify-center">
          <Link href="/" className="underline">Back to Home</Link>
          <Link href="/get_chunks" className="underline">Learn about /get_chunks</Link>
        </div>
      </Card>
    </div>
  );
}


