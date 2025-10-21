import { NextRequest, NextResponse } from 'next/server';
import { BASE_URL, GENERIC_ERROR_MESSAGE } from '@/lib/utils';

export async function POST(req: NextRequest) {
  try {
    const { user_question, document, embedding_type, comments } = await req.json();
    if (!user_question || typeof user_question !== 'string') {
      return NextResponse.json({ error: GENERIC_ERROR_MESSAGE }, { status: 400 });
    }
    // Forward to FastAPI
    const response = await fetch(`${BASE_URL}/form/upload_ratings/chunk`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        "user_question": user_question,
        "data": document,
        "embedding_type": embedding_type,
        ...(typeof comments === 'string' && comments.trim() ? { comments } : {}),
      }),
    });

    if (!response.ok) {
      return NextResponse.json({ error: GENERIC_ERROR_MESSAGE }, { status: 502 });
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Error in upload_ratings:', error);
    return NextResponse.json({ error: GENERIC_ERROR_MESSAGE }, { status: 500 });
  }
} 