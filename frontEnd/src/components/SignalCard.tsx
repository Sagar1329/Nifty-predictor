import {
    Paper,
    Typography,
    Box,
    LinearProgress,
    Chip,
    Divider,
} from "@mui/material";
import type { LiveState } from "../types/state";
import type { JSX } from "react";
import { CircularProgress } from "@mui/material";
import { useAutoRefresh } from "../hooks/useAutoRefresh";

interface Props {
    state: LiveState;
    refresh_interval: number;
}

const signalColors: Record<string, string> = {
    UP: "#2e7d32",
    DOWN: "#c62828",
    UNCERTAIN: "#6d6d6d",
};

const confidenceColors: Record<string, "success" | "warning" | "error"> = {
    HIGH: "success",
    MEDIUM: "warning",
    LOW: "error",
};

function ProbabilityBar({
    label,
    value,
    color,
}: {
    label: string;
    value: number;
    color: string;
}) {

    return (
        <Box sx={{ mb: 1.5 }}>
            <Box
                sx={{
                    display: "flex",
                    justifyContent: "space-between",
                    mb: 0.5,
                }}
            >
                <Typography variant="body2">{label}</Typography>
                <Typography variant="body2">
                    {(value * 100).toFixed(1)}%
                </Typography>
            </Box>

            <LinearProgress
                variant="determinate"
                value={value * 100}
                sx={{
                    height: 10,
                    borderRadius: 5,
                    backgroundColor: "#eeeeee",
                    [`& .MuiLinearProgress-bar`]: {
                        backgroundColor: color,
                    },
                }}
            />
        </Box>
    );
}

export default function SignalCard({ state, refresh_interval }: Props): JSX.Element {
    if (!state.signal || !state.probabilities) {
        return (
            <Paper sx={{ p: 4 }}>
                <Typography>No prediction available</Typography>
            </Paper>
        );
    }

    const signalColor = signalColors[state.signal] ?? "#333";
    const secondsLeft = useAutoRefresh(refresh_interval);


    return (
        <Paper
            elevation={6}
            sx={{
                p: 4,
                maxWidth: 480,
                borderRadius: 4,
                background: "linear-gradient(180deg, #ffffff, #fafafa)",
            }}
        >
            {/* Header */}
            <Box
                sx={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                    mb: 2,
                }}
            >
                <Typography variant="h6">Market Signal</Typography>
                <Box
                sx={{
                    display:'flex',
                        justifyContent: "space-between",
                        alignItems: "center",
                        gap:1

                }}
                >
                    <Typography variant="body2">Confidence</Typography>
                    <Chip
                        label={state.confidence_level}
                        color={confidenceColors[state.confidence_level as string]}
                        size="small"
                    />
                </Box>

               
            </Box>

            <Divider sx={{ mb: 3 }} />
            <Box
                sx={{
                    display: "flex",
                    justifyContent: "space-between",
                    mb: 2,
                }}
            >
                <Typography variant="body2" color="text.secondary">
                    Last updated:
                </Typography>

                <Typography variant="body2" fontWeight={500}>
                    {state.timestamp
                        ? new Date(state.timestamp).toLocaleTimeString()
                        : "—"}
                </Typography>
            </Box>


            {/* Main Signal */}
            <Box sx={{ textAlign: "center", mb: 3 }}>
                <Typography
                    variant="h3"
                    sx={{
                        fontWeight: 700,
                        color: signalColor,
                        letterSpacing: 1,
                    }}
                >
                    {state.signal}
                </Typography>

                {state.timestamp && (
                    <Typography
                        variant="body2"
                        color="text.secondary"
                        sx={{ mt: 1 }}
                    >
                        Candle close: {state.timestamp}
                    </Typography>
                )}
            </Box>

            {/* Probabilities */}
            <Box>
                <Typography
                    variant="subtitle2"
                    sx={{ mb: 1, fontWeight: 600 }}
                >
                    Model Probabilities
                </Typography>

                <ProbabilityBar
                    label="UP"
                    value={state.probabilities.UP}
                    color="#2e7d32"
                />
                <ProbabilityBar
                    label="SIDEWAYS"
                    value={state.probabilities.SIDEWAYS}
                    color="#ed6c02"
                />
                <ProbabilityBar
                    label="DOWN"
                    value={state.probabilities.DOWN}
                    color="#c62828"
                />
            </Box>

            <Box
                sx={{
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "start",
                    mt: 3,
                }}
            >
                <CircularProgress size={16} />
                <Typography variant="caption" color="text.secondary">
                    Refreshing in {secondsLeft}s
                </Typography>

               
            </Box>

        </Paper>
    );
}
