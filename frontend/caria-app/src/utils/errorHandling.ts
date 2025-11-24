/**
 * Error handling utilities for consistent error handling across the app
 */

export interface ApiError {
  message: string;
  detail?: string;
  status?: number;
  name?: string;
}

/**
 * Safely extract error message from unknown error type
 */
export function getErrorMessage(error: unknown): string {
  if (error instanceof Error) {
    return error.message;
  }
  if (typeof error === 'string') {
    return error;
  }
  if (error && typeof error === 'object' && 'message' in error) {
    return String(error.message);
  }
  return 'An unexpected error occurred';
}

/**
 * Check if error is an AbortError (cancelled request)
 */
export function isAbortError(error: unknown): boolean {
  return error instanceof Error && error.name === 'AbortError';
}

/**
 * Check if error is an authentication error
 */
export function isAuthError(error: unknown): boolean {
  const message = getErrorMessage(error).toLowerCase();
  return message.includes('401') || message.includes('403') || message.includes('session expired');
}

/**
 * Check if error is a network/connection error
 */
export function isNetworkError(error: unknown): boolean {
  const message = getErrorMessage(error).toLowerCase();
  return (
    message.includes('failed to fetch') ||
    message.includes('network') ||
    message.includes('connection') ||
    message.includes('timeout')
  );
}
