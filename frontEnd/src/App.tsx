import { Container, Typography } from "@mui/material";
import type { JSX } from "react";
import LiveDashboard from "./pages/LiveDashboard";

function App(): JSX.Element {
  return (
    <Container maxWidth="md" sx={{ mt: 4 }}>
      <Typography variant="h4" gutterBottom>
        Nifty Predictor – Live
      </Typography>

      
      <LiveDashboard/>
    </Container>
  );
}

export default App;
