import { NextRequest, NextResponse } from 'next/server';
import { BASE_URL, GENERIC_ERROR_MESSAGE } from '@/lib/utils';

// Upload rating for a full generated response
export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { user_question } = body || {};

    if (!user_question || typeof user_question !== 'string') {
      return NextResponse.json({ error: GENERIC_ERROR_MESSAGE }, { status: 400 });
    }

    // If a rankings array is provided, send a single request with list rankings
    if (Array.isArray(body.rankings)) {
      const url = `${BASE_URL}/form/upload_ratings/full`;
      const rankings = body.rankings
        .filter((r: any) => r && (typeof r.prompt_id === 'string' || typeof r.prompt_id === 'number') && typeof r.rank === 'number')
        .map((r: any) => ({ prompt_id: String(r.prompt_id), rank: r.rank as number }));

      if (rankings.length === 0) {
        return NextResponse.json({ error: GENERIC_ERROR_MESSAGE }, { status: 400 });
      }

      const payload = { user_question, rankings, comments: typeof body?.comments === 'string' ? body.comments : undefined } as const;
      const res = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        return NextResponse.json({ error: GENERIC_ERROR_MESSAGE }, { status: 502 });
      }
      const data = await res.json();
      return NextResponse.json(data);
    }

    // Otherwise, fall back to single-response rating payload
    const { message, transcript_data, rating, prompt_id } = body || {};
    if (typeof rating !== 'number' || rating < 0 || rating > 100) {
      return NextResponse.json({ error: GENERIC_ERROR_MESSAGE }, { status: 400 });
    }

    const singlePayload = {
      user_question,
      message,
      transcript_data,
      rating,
      prompt_id,
    } as const;

    // Support single ranking if provided directly
    if ((typeof body?.prompt_id === 'string' || typeof body?.prompt_id === 'number') && typeof (body as any)?.rank === 'number') {
      const url = `${BASE_URL}/form/upload_ratings/full`;
      const payload = {
        user_question,
        rankings: [{
          prompt_id: String(body.prompt_id),
          rank: (body as any).rank,
        }],
        comments: typeof body?.comments === 'string' ? body.comments : undefined,
      } as const;
      const res = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        return NextResponse.json({ error: GENERIC_ERROR_MESSAGE }, { status: 502 });
      }
      const data = await res.json();
      return NextResponse.json(data);
    }

    // Legacy fallback
    const legacyUrl = `${BASE_URL}/form/upload_full_response_rating`;
    const res = await fetch(legacyUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(singlePayload),
    });
    if (!res.ok) {
      return NextResponse.json({ error: GENERIC_ERROR_MESSAGE }, { status: 502 });
    }
    const data = await res.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Error in upload_full_response_rating:', error);
    return NextResponse.json({ error: GENERIC_ERROR_MESSAGE }, { status: 500 });
  }
}


