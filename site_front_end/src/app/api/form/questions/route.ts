import { NextResponse } from 'next/server';
import { BASE_URL, GENERIC_ERROR_MESSAGE } from '@/lib/utils';

export async function GET() {
  try {
    const response = await fetch(`${BASE_URL}/form/data/get_all_questions`, { cache: 'no-store' });
    if (!response.ok) {
      return NextResponse.json({ error: GENERIC_ERROR_MESSAGE }, { status: 502 });
    }
    const data = await response.json();
    return NextResponse.json(Array.isArray(data) ? data : (Array.isArray(data?.questions) ? data.questions : []));
  } catch (error) {
    console.error('Error fetching /form/questions:', error);
    return NextResponse.json({ error: GENERIC_ERROR_MESSAGE }, { status: 502 });
  }
}


