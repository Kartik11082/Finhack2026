import { useEffect, useState } from "react";
import actualSentiment from "../assets/actualSentiment.png";
import flowDiag from "../assets/FlowDiag.jpg";

export default function SlideResearchAssets() {
  const [activeAsset, setActiveAsset] = useState(null);

  useEffect(() => {
    const onKeyDown = (event) => {
      if (event.key === "Escape") {
        setActiveAsset(null);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  const assets = {
    flow: {
      src: flowDiag,
      alt: "Flow diagram showing the project workflow and data pipeline",
      title: "Workflow view",
      description: "High-level process diagram for data ingestion, modeling, and output generation.",
    },
    sentiment: {
      src: actualSentiment,
      alt: "Sentiment visualization used in the project analysis",
      title: "Signal view",
      description: "Visual reference for the sentiment pattern that feeds the prediction logic.",
    },
  };

  return (
    <div className="slide slide-research-assets">
      <div className="research-assets-header">
        <div className="module-title">Opening Context</div>
        <h1 className="hero-title">Two visuals that frame the whole deck.</h1>
        <p className="hero-subtitle">
          The flow diagram shows the modeling logic and data path. The sentiment image anchors the signal we are trying
          to extract from transcripts and related text inputs.
        </p>
      </div>

      <div className="research-assets-grid">
        <button
          type="button"
          className="research-asset-card research-asset-button research-asset-large"
          onClick={() => setActiveAsset(assets.flow)}
        >
          <img src={assets.flow.src} alt={assets.flow.alt} />
          <figcaption>
            <strong>{assets.flow.title}</strong>
            <span>{assets.flow.description}</span>
            <em>Click to expand</em>
          </figcaption>
        </button>

        <button
          type="button"
          className="research-asset-card research-asset-button research-asset-small"
          onClick={() => setActiveAsset(assets.sentiment)}
        >
          <img src={assets.sentiment.src} alt={assets.sentiment.alt} />
          <figcaption>
            <strong>{assets.sentiment.title}</strong>
            <span>{assets.sentiment.description}</span>
            <em>Click to expand</em>
          </figcaption>
        </button>
      </div>

      {activeAsset && (
        <div className="research-lightbox" role="dialog" aria-modal="true" aria-label={activeAsset.title}>
          <button
            type="button"
            className="research-lightbox-backdrop"
            aria-label="Close expanded image"
            onClick={() => setActiveAsset(null)}
          />
          <div className="research-lightbox-card">
            <button
              type="button"
              className="research-lightbox-close"
              aria-label="Close expanded image"
              onClick={() => setActiveAsset(null)}
            >
              Close
            </button>
            <div className="research-lightbox-media">
              <img src={activeAsset.src} alt={activeAsset.alt} />
            </div>
            <div className="research-lightbox-copy">
              <strong>{activeAsset.title}</strong>
              <span>{activeAsset.description}</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
