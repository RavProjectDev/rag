import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}
export const MAIN_URL = "http://3.17.36.74:8000";
export const LOCAL_URL = "http://localhost:8000";

export const BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ||
  (process.env.NODE_ENV === "development" ? LOCAL_URL : MAIN_URL);

// User-friendly generic message shown for all unexpected errors
export const GENERIC_ERROR_MESSAGE = "Something went wrong. Please try again later.";

// Wrapper around fetch that throws a friendly Error on network/HTTP failures
export async function safeFetch(input: RequestInfo | URL, init?: RequestInit) {
  try {
    const response = await fetch(input as any, init);
    if (!response.ok) {
      // Swallow upstream details and throw a friendly message
      throw new Error(GENERIC_ERROR_MESSAGE);
    }
    return response;
  } catch (_) {
    // Network errors or other exceptions
    throw new Error(GENERIC_ERROR_MESSAGE);
  }
}