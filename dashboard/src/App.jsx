import { useState, useEffect } from "react";
import { DataProvider, useData } from "./DataContext";
import SlideResearchAssets from "./components/SlideResearchAssets";
import SlidePrologue from "./components/SlidePrologue";
import SlideBaseline from "./components/SlideBaseline";
import SlideSentiment from "./components/SlideSentiment";
import SlideComparison from "./components/SlideComparison";
import SlideSpilloverStocks from "./components/SlideSpilloverStocks";

function SlideDeck() {
  const { comparison, enhancedRows, foldResults } = useData();
  const [slide, setSlide] = useState(0);
  const totalSlides = 6;

  const navigate = (dir) => {
    const next = slide + dir;
    if (next >= 0 && next < totalSlides) {
      if (document.startViewTransition) {
        document.startViewTransition(() => setSlide(next));
      } else {
        setSlide(next);
      }
    }
  };

  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === "ArrowRight") navigate(1);
      if (e.key === "ArrowLeft") navigate(-1);
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [slide]);

  // Make sure data is loaded
  if (!comparison.baseline) {
    return <div className="deck-container">Loading Models...</div>;
  }

  const slides = [
    <SlideResearchAssets key="0" />,
    <SlidePrologue key="1" />,
    <SlideBaseline key="2" />,
    <SlideSentiment key="3" />,
    <SlideComparison key="4" />,
    <SlideSpilloverStocks key="5" />
  ];

  return (
    <>
      <div className="nav-region left" onClick={() => navigate(-1)} />
      <div className="nav-region right" onClick={() => navigate(1)} />
      
      <div className="deck-container">
        {slides[slide]}
      </div>

      <div className="deck-controls">
        {[0, 1, 2, 3, 4, 5].map((idx) => (
          <button 
            key={idx} 
            className={`deck-dot ${slide === idx ? "active" : ""}`}
            onClick={() => {
              if (document.startViewTransition) {
                document.startViewTransition(() => setSlide(idx));
              } else {
                setSlide(idx);
              }
            }}
            aria-label={`Go to slide ${idx + 1}`}
          />
        ))}
      </div>
    </>
  );
}

export default function App() {
  return (
    <DataProvider>
      <SlideDeck />
    </DataProvider>
  );
}
