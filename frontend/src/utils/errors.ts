export interface AppError {
  userMessage: string
  devMessage: string
}

export function parseApiError(err: unknown, status?: number): AppError {
  if (status === 422) {
    return {
      userMessage: 'Invalid passenger data. Please check your inputs and try again.',
      devMessage: `422 Unprocessable Entity — validation failed on /predict`,
    }
  }

  if (status !== undefined) {
    return {
      userMessage: 'Sorry, something went wrong. Please try again later or contact support if the issue persists.',
      devMessage: `Unexpected API response: HTTP ${status}`,
    }
  }

  if (err instanceof TypeError && err.message.includes('fetch')) {
    return {
      userMessage: 'Unable to reach the prediction service. Please check your connection and try again. Contact support if the issue persists.',
      devMessage: `Failed to connect to the prediction API: ${err.message}. Make sure the backend is running.`,
    }
  }

  return {
    userMessage: 'Sorry, something went wrong. Please try again later or contact support if the issue persists.',
    devMessage: `Unknown error: ${err instanceof Error ? err.message : String(err)}`,
  }
}