import { NextRequest, NextResponse } from 'next/server';
import { BASE_URL, GENERIC_ERROR_MESSAGE } from '@/lib/utils';

export async function POST(req: NextRequest) {
  try {
    // Parse the question from the request body (not used in mock)
    const { question } = await req.json();
    const response = await fetch(`${BASE_URL}/form/${question}`, {
      method: 'GET',
    });
    if (!response.ok) {
      console.error(`Upstream error: ${response.status}`);
      return NextResponse.json({ error: GENERIC_ERROR_MESSAGE, chunks: [] }, { status: 502 });
    }
    const data = await response.json();
    // Map the first 5 items from the response to chunks
    const documents = data.documents;
    const chunks = Array.isArray(documents) ? documents.slice(0, 5).map(item => ({ ...item })) : [];
    const embedding_type = data.embedding_type;
    return NextResponse.json({ chunks, embedding_type });
  } catch (error) {
    console.error('Error in get_chunks:', error);
    return NextResponse.json({ error: GENERIC_ERROR_MESSAGE, chunks: [] }, { status: 500 });
  }
}
