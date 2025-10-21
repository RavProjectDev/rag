import { NextRequest, NextResponse } from 'next/server';
import { BASE_URL, GENERIC_ERROR_MESSAGE } from '@/lib/utils';

// Fetch a full generated response for a given question
export async function POST(req: NextRequest) {
  try {
    const { question } = await req.json();
    if (!question || typeof question !== 'string') {
      return NextResponse.json({ error: GENERIC_ERROR_MESSAGE }, { status: 400 });
    }

    // Explicitly call the backend full-response endpoint via GET with path param
    const url = `${BASE_URL}/form/full/${encodeURIComponent(question)}`;
    const response = await fetch(url, { method: 'GET' });
    if (!response.ok) {
      return NextResponse.json({ error: GENERIC_ERROR_MESSAGE }, { status: 502 });
    }
    const data = await response.json();

    // Normalize to a simple shape { responses: [{ message, transcript_data, prompt_id }] }
    type Normalized = { message: string; transcript_data: unknown[]; prompt_id?: string | number };
    const normalize = (item: unknown): Normalized => {
      if (typeof item !== 'object' || item === null) {
        return { message: '', transcript_data: [] };
      }
      const obj = item as Record<string, unknown>;
      const message =
        typeof obj.llm_response === 'string' ? obj.llm_response :
        (typeof obj.message === 'string' ? obj.message : '');
      const transcript_data = Array.isArray(obj.transcript_data) ? obj.transcript_data : [];
      const promptIdRaw = obj.prompt_id;
      const prompt_id = (typeof promptIdRaw === 'number' || typeof promptIdRaw === 'string') ? promptIdRaw : undefined;
      return { message, transcript_data, prompt_id };
    };

    let responses: Array<{ message: string; transcript_data: unknown[]; prompt_id?: string }> = [];
    if (Array.isArray(data)) {
      const normalized = data.map(normalize);
      responses = normalized.map((r) => ({ ...r, prompt_id: r.prompt_id != null ? String(r.prompt_id) : undefined }));
    } else if (typeof data === 'object' && data !== null && Array.isArray((data as Record<string, unknown>).responses)) {
      const normalized = (data as Record<string, unknown>).responses as unknown[];
      responses = normalized.map(normalize).map((r) => ({ ...r, prompt_id: r.prompt_id != null ? String(r.prompt_id) : undefined }));
    } else if (typeof data === 'object' && data) {
      const r = normalize(data);
      responses = [{ ...r, prompt_id: r.prompt_id != null ? String(r.prompt_id) : undefined }];
    }

    return NextResponse.json({ responses });
  } catch (error) {
    console.error('Error in get_full_response:', error);
    return NextResponse.json({ error: GENERIC_ERROR_MESSAGE }, { status: 500 });
  }
}

