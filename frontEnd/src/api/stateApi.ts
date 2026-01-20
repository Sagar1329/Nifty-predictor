// src/api/stateApi.ts
// src/api/stateApi.ts
import type { LiveState } from "../types/state";

/**
 * MOCK implementation of /state.
 * This will later be replaced by a real fetch().
 */

export async function fetchLiveState(): Promise<LiveState> {
    await new Promise((resolve) => setTimeout(resolve, 300));

    const mock: LiveState = {
        status: "live",
        phase: "prediction",
        timestamp: "2026-01-19 14:40:00",
        signal: "UP",
        confidence_level: "MEDIUM",
        probabilities: {
            UP: 0.56,
            SIDEWAYS: 0.28,
            DOWN: 0.16,
        },
    };

    return mock;
}
