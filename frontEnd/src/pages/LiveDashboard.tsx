import { useEffect, useState, type JSX } from "react";
import { Box, Typography, Paper } from "@mui/material";

import type { LiveState } from "../types/state";
import { fetchLiveState } from "../api/stateApi";
import SignalCard from "../components/SignalCard";

export default function LiveDashboard(): JSX.Element {
    const [state, setState] = useState<LiveState | null>(null);

    useEffect(() => {
        fetchLiveState().then(setState);
    }, []);

    if (!state) {
        return <Typography>Loading…</Typography>;
    }

    return <SignalCard state={state} />;
}
