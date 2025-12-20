import { useState } from "react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardHeader,
  CardTitle,
  CardContent,
} from "@/components/ui/card";

import { predictionMockResponse } from "@/mockData/predictionMock";

type PredictionResponse = {
  timestamp: string;
  signal: "UP" | "DOWN" | "SIDEWAYS";
  confidence_level: "LOW" | "MEDIUM" | "HIGH";
  prediction: "UP" | "DOWN" | "SIDEWAYS";
  probabilities: {
    UP: number;
    DOWN: number;
    SIDEWAYS: number;
  };
};

function App() {
  const [data, setData] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);

  const fetchPrediction = () => {
    setLoading(true);
    setTimeout(() => {
      setData(predictionMockResponse as PredictionResponse);
      setLoading(false);
    }, 800);
  };

  // helper to convert decimal to %
  const formatPercentage = (value: number) =>
    `${Math.round(value * 100)}%`;

  // return (
  //   <div className="min-h-screen bg-blue-400 flex flex-col items-center justify-center gap-6 p-6">

  //     {/* Title */}
  //     <h1 className="text-4xl font-bold text-gray-800">
  //       Nifty Predictor 
  //     </h1>

  //     {/* Button */}
  //     <Button onClick={fetchPrediction} disabled={loading}>
  //       {loading ? "Fetching..." : "Get Prediction"}
  //     </Button>

  //     {/* Card */}
  //     {data && (
  //       <Card className="w-full max-w-md shadow-lg">
  //         <CardHeader>
  //           <CardTitle className="text-center text-xl">
  //             Market Prediction
  //           </CardTitle>
  //         </CardHeader>

  //         <CardContent className="space-y-3 text-gray-700">
  //           <p>
  //             <strong>Signal:</strong>{" "}
  //             <span className="font-semibold text-red-600">
  //               {data.signal}
  //             </span>
  //           </p>

  //           <p>
  //             <strong>Confidence Level:</strong>{" "}
  //             {data.confidence_level}
  //           </p>

  //           <p>
  //             <strong>Prediction:</strong>{" "}
  //             {data.prediction}
  //           </p>

  //           <div>
  //             <strong>Probabilities:</strong>
  //             <ul className="mt-2 space-y-1">
  //               <li>
  //                 DOWN: {formatPercentage(data.probabilities.DOWN)}
  //               </li>
  //               <li>
  //                 SIDEWAYS: {formatPercentage(data.probabilities.SIDEWAYS)}
  //               </li>
  //               <li>
  //                 UP: {formatPercentage(data.probabilities.UP)}
  //               </li>
  //             </ul>
  //           </div>
  //         </CardContent>
  //       </Card>
  //     )}
  //   </div>
  // );

  return (
    <div className="min-h-screen bg-blue-300 flex flex-col items-center justify-center gap-6 px-4 py-8">

      <h1 className="text-white font-extrabold text-center
      text-[clamp(2rem,5vw,3.5rem)]">
        Nifty Predictor
      </h1>


      <Button
        onClick={fetchPrediction}
        disabled={loading}
        className="
        px-6 py-4
        text-[clamp(1rem,2.5vw,1.25rem)]
        font-semibold
      "
      >
        {loading ? "Fetching..." : "Get Prediction"}
      </Button>

      {data && (
        <Card
          className="
          w-full
          max-w-[clamp(20rem,90vw,36rem)]
          rounded-2xl
          shadow-2xl
        "
        >
          <CardHeader>
            <CardTitle
              className="
              text-center font-bold
              text-[clamp(1.4rem,3vw,2rem)]
            "
            >
              Market Prediction
            </CardTitle>
          </CardHeader>

          <CardContent
            className="
            space-y-5
            text-[clamp(1rem,2.5vw,1.2rem)]
          "
          >
            <div className="grid grid-cols-[140px_1fr] items-center">
              <span className="text-gray-700 font-semibold">
                Signal
              </span>
              <span className="inline-block px-3 py-1 rounded-md
                   text-red-600 bg-red-100
                   font-bold text-lg w-fit">
                {data.signal}
              </span>
            </div>

            <div className="grid grid-cols-[140px_1fr] items-center">
              <span className="text-gray-700 font-semibold">
                Confidence
              </span>
              <span className="inline-block px-3 py-1 rounded-md
                   bg-gray-200 text-gray-900
                   font-semibold w-fit">
                {data.confidence_level}
              </span>
            </div>

            <div className="grid grid-cols-[140px_1fr] items-center">
              <span className="text-gray-700 font-semibold">
                Prediction
              </span>
              <span className="inline-block px-3 py-1 rounded-md
                   bg-gray-200 text-gray-900
                   font-semibold w-fit">
                {data.prediction}
              </span>
            </div>

            <div className="border-t pt-4">
              <h3 className="font-bold mb-3
              text-[clamp(1.2rem,3vw,1.6rem)]">
                Probabilities
              </h3>

              <div className="space-y-2">
                {Object.entries(data.probabilities).map(([key, value]) => (
                  <div key={key} className="grid grid-cols-[140px_1fr] items-center">
                    <span className="text-gray-700 font-semibold">{key}</span>
                    <span className="inline-block px-3 py-1 rounded-md
                   bg-gray-200 text-gray-900
                   font-semibold w-fit">
                      {formatPercentage(value)}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );

}

export default App;
