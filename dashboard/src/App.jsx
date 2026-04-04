import { useState, useEffect } from "react";
import { DataProvider, useData } from "./DataContext";
import SlidePrologue from "./components/SlidePrologue";
import SlideBaseline from "./components/SlideBaseline";
import SlideSentiment from "./components/SlideSentiment";
import SlideComparison from "./components/SlideComparison";

function SlideDeck() {
  const { comparison, enhancedRows, foldResults } = useData();
  const [slide, setSlide] = useState(0);
  const totalSlides = 4;

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
    <SlidePrologue key="0" />,
    <SlideBaseline key="1" />,
    <SlideSentiment key="2" />,
    <SlideComparison key="3" />
  ];

  return (
    <>
      <div className="nav-region left" onClick={() => navigate(-1)} />
      <div className="nav-region right" onClick={() => navigate(1)} />
      
      <div className="deck-container">
        {slides[slide]}
      </div>

      <div className="deck-controls">
        {[0, 1, 2, 3].map((idx) => (
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
