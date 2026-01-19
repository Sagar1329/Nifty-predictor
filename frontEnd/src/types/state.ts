export type MarketPhase =
    | "waiting_for_next_candle"
    | "prediction"
    | "market_closed"
    | "no_data"
    | "error";

export type ConfidenceLevel = "HIGH" | "MEDIUM" | "LOW";

export interface Probabilities {
    UP: number;
    SIDEWAYS: number;
    DOWN: number;
}

export interface LiveState {
    status: "live" | "stopped" | "error";

    phase: MarketPhase;

    timestamp?: string;       
    last_candle_time?: string; 

    signal?: "UP" | "DOWN" | "SIDEWAYS" | "UNCERTAIN";
    confidence_level?: ConfidenceLevel;

    probabilities?: Probabilities;

    message?: string;          
}
