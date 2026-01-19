import { Container, Typography } from "@mui/material";
import type { JSX } from "react";

function App(): JSX.Element {
  return (
    <Container maxWidth="md" sx={{ mt: 4 }}>
      <Typography variant="h4" gutterBottom>
        Nifty Predictor – Live
      </Typography>

      <Typography color="text.secondary">
        Frontend initialized successfully (TypeScript).
      </Typography>
    </Container>
  );
}

export default App;
