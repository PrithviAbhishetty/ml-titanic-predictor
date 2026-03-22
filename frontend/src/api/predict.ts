import type { PassengerInput, PredictionOutput } from '../types/passenger'
import { parseApiError } from '../utils/errors'
export type { AppError } from '../utils/errors'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export async function predictSurvival(
  passenger: PassengerInput
): Promise<PredictionOutput> {
  let response: Response

  try {
    response = await fetch(`${API_URL}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(passenger),
    })
  } catch (err) {
    throw parseApiError(err)
  }

  if (!response.ok) {
    throw parseApiError(null, response.status)
  }

  return response.json()
}