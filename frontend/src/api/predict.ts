import type { PassengerInput, PredictionOutput } from '../types/passenger'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export async function predictSurvival(
  passenger: PassengerInput
): Promise<PredictionOutput> {
  const response = await fetch(`${API_URL}/predict`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(passenger),
  })

  if (!response.ok) {
    throw new Error(`API error: ${response.status}`)
  }

  return response.json()
}