# frontend/

React + TypeScript + Vite frontend for the Titanic survival predictor. Provides a passenger input form and renders the predicted survival outcome and probability returned by the FastAPI backend.

---

## Structure

```
frontend/
├── src/
│   ├── App.tsx                    # Root component — state management, layout
│   ├── components/
│   │   ├── PredictionForm.tsx     # Passenger input form + submit handler
│   │   └── PredictionResult.tsx  # Survival outcome display
│   ├── api/
│   │   └── predict.ts            # fetch wrapper for POST /predict
│   ├── utils/
│   │   └── errors.ts             # Error normalisation — userMessage / devMessage
│   ├── types/
│   │   └── passenger.ts          # PassengerInput and PredictionOutput types
│   └── index.css                 # Global styles and CSS variables
└── public/
    └── titanic.svg
```

---

## Error Handling

API errors are normalised in `utils/errors.ts` into an `AppError` shape with two fields:

- `userMessage` — displayed in the UI (friendly, no technical detail)
- `devMessage` — logged to `console.error` (status codes, raw error messages)

This separation ensures implementation details never surface to users while keeping developer-relevant context accessible in the browser console.

---

## Environment Variables

| Variable | Value |
|----------|-------|
| `VITE_API_URL` | URL of the FastAPI backend |

For local development, create `frontend/.env.local`:
```
VITE_API_URL=http://localhost:8000
```

On Vercel:
- Production: `https://ml-titanic-predictor.onrender.com`
- Preview: `https://ml-titanic-predictor-staging.onrender.com`

---

## Running Locally

```bash
npm install --prefix frontend
npm run dev --prefix frontend
```

Requires `frontend/.env.local` with `VITE_API_URL` set. Backend must be running for predictions to work.

---

## Build

```bash
npm run build --prefix frontend
```

Output goes to `frontend/dist/`. Vercel handles this automatically on deploy.