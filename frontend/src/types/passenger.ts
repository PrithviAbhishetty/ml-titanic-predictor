export interface PassengerInput {
  pclass: number
  sex: 'male' | 'female'
  age: number
  sibsp: number
  parch: number
  fare: number
  embarked: 'S' | 'C' | 'Q'
}

export interface PredictionOutput {
  survived: boolean
  survival_probability: number
}